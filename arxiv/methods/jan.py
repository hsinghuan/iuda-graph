from typing import Optional, Sequence
from copy import deepcopy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from .grl import GradientReverseLayer
from pytorch_adapt.validators import IMValidator

class JAN():
    def __init__(self, encoder, classifier, src_data, device="cpu"):
        self.device = device
        self.set_encoder_classifier(encoder, classifier)
        thetas = None  # none adversarial
        self.jmmd_loss = JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=(
                [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
                (GaussianKernel(sigma=0.92, track_running_stats=False),)
            ),
            linear=False, thetas=thetas
        ).to(self.device)
        self.src_data = src_data.to(self.device)
        self.validator = IMValidator()

    def _adapt_train_epoch(self, encoder, classifier, tgt_data, optimizer, lambda_coeff):
        encoder.train()
        classifier.train()
        self.jmmd_loss.train()

        src_y, src_f = classifier(encoder(self.src_data.x, self.src_data.edge_index))
        tgt_y, tgt_f = classifier(encoder(tgt_data.x, tgt_data.edge_index))
        cls_loss = F.nll_loss(F.log_softmax(src_y[self.src_data.train_mask], dim=1), self.src_data.y[self.src_data.train_mask])
        transfer_loss = self._minibatch_jmmd(src_f[self.src_data.train_mask], src_y[self.src_data.train_mask],
                                       tgt_f[tgt_data.train_mask], tgt_y[tgt_data.train_mask])
        loss = cls_loss + transfer_loss * lambda_coeff

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), cls_loss.item(), transfer_loss.item(), src_y[self.src_data.train_mask], tgt_y[tgt_data.train_mask]

    @torch.no_grad()
    def _adapt_test_epoch(self, encoder, classifier, tgt_data, lambda_coeff):
        encoder.eval()
        classifier.eval()
        self.jmmd_loss.eval()

        src_y, src_f = classifier(encoder(self.src_data.x, self.src_data.edge_index))
        tgt_y, tgt_f = classifier(encoder(tgt_data.x, tgt_data.edge_index))
        cls_loss = F.nll_loss(F.log_softmax(src_y[self.src_data.val_mask], dim=1), self.src_data.y[self.src_data.val_mask])
        transfer_loss = self._minibatch_jmmd(src_f[self.src_data.val_mask], src_y[self.src_data.val_mask],
                                       tgt_f[tgt_data.val_mask], tgt_y[tgt_data.val_mask])
        loss = cls_loss + transfer_loss * lambda_coeff

        return loss.item(), cls_loss.item(), transfer_loss.item(), src_y[self.src_data.val_mask], tgt_y[tgt_data.val_mask]

    def _adapt_train_test(self, tgt_data, lambda_coeff, args):
        encoder = deepcopy(self.encoder)
        classifier = deepcopy(self.classifier)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.adapt_lr)

        best_val_loss = np.inf
        best_val_score = None
        best_encoder, best_classifier = None, None
        patience = 20
        staleness = 0
        for e in range(1, args.adapt_epochs):
            train_loss, train_cls_loss, train_transfer_loss, train_src_logits, train_tgt_logits = self._adapt_train_epoch(encoder, classifier, tgt_data, optimizer, lambda_coeff)
            val_loss, val_cls_loss, val_transfer_loss, val_src_logits, val_tgt_logits = self._adapt_test_epoch(encoder, classifier, tgt_data, lambda_coeff)
            val_score = self.negative_entropy_from_logits(val_tgt_logits) # self.validator(target_train={'logits': val_tgt_logits})
            val_src_score = self.negative_entropy_from_logits(val_src_logits) # self.validator(target_train={'logits': val_src_logits})
            train_src_score = self.negative_entropy_from_logits(train_src_logits) # self.validator(target_train={'logits': train_src_logits})
            train_score = self.negative_entropy_from_logits(train_tgt_logits) # self.validator(target_train={'logits': train_tgt_logits})
            # train_entropy = self.entropy(train_tgt_logits)
            # val_entropy = self.entropy(val_tgt_logits)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_score = val_score
                best_encoder = deepcopy(encoder)
                best_classifier = deepcopy(classifier)
                staleness = 0
            else:
                staleness += 1
            print(
                f'Lambda {lambda_coeff} Epoch {e}/{args.adapt_epochs} Train Loss: {round(train_loss, 3)} Train Cls Loss: {round(train_cls_loss, 3)} Train Transfer Loss: {round(train_transfer_loss, 3)} \n Val Loss: {round(val_loss, 3)} Val Cls Loss: {round(val_cls_loss, 3)} Val Transfer Loss: {round(val_transfer_loss, 3)}')

            self.writer.add_scalar("Total Loss/train", train_loss, e)
            self.writer.add_scalar("Total Loss/val", val_loss, e)
            self.writer.add_scalar("Source Label Loss/train", train_cls_loss, e)
            self.writer.add_scalar("Source Label Loss/val", val_cls_loss, e)
            self.writer.add_scalar("Transfer Loss/train", train_transfer_loss, e)
            self.writer.add_scalar("Transfer Loss/val", val_transfer_loss, e)
            self.writer.add_scalar("Target Score/val", val_score, e)
            self.writer.add_scalar("Target Score/train", train_score, e)
            self.writer.add_scalar("Source Score/train", train_src_score, e)
            self.writer.add_scalar("Source Score/val", val_src_score, e)

            if staleness > patience:
                break

        encoder = deepcopy(best_encoder)
        classifier = deepcopy(best_classifier)
        return encoder, classifier, best_val_score

    def adapt(self, tgt_data, lambda_coeff_list, stage, args):
        tgt_data = tgt_data.to(self.device)
        performance_dict = dict()
        for lambda_coeff in lambda_coeff_list:
            run_name = args.method + "_" + str(lambda_coeff).replace(".","") + "_" + str(args.model_seed)
            self.writer = SummaryWriter(os.path.join(args.log_dir, str(stage[0]) + "_" + str(stage[1]) + "_" + str(stage[2]), run_name))
            encoder, classifier, val_score = self._adapt_train_test(tgt_data, lambda_coeff, args)
            performance_dict[lambda_coeff] = {'tgt_encoder': encoder, 'tgt_classifier': classifier, 'tgt_val_score': val_score}

        best_val_score = -np.inf
        best_encoder, best_classifier = None, None
        for lambda_coeff, perf_dict in performance_dict.items():
            if perf_dict['tgt_val_score'] > best_val_score:
                best_val_score = perf_dict['tgt_val_score']
                best_encoder = perf_dict['tgt_encoder']
                best_classifier = perf_dict['tgt_classifier']

        self.set_encoder_classifier(best_encoder, best_classifier)

    def set_encoder_classifier(self, encoder, classifier):
        self.encoder = deepcopy(encoder).to(self.device)
        self.classifier = deepcopy(classifier).to(self.device)

    def get_encoder_classifier(self):
        return self.encoder, self.classifier

    def _minibatch_jmmd(self, src_f, src_y, tgt_f, tgt_y, batch_size=32):
        src_loader = torch.utils.data.DataLoader(tuple(zip(list(src_f), list(src_y))), batch_size=batch_size,
                                                 shuffle=True)
        tgt_loader = torch.utils.data.DataLoader(tuple(zip(list(tgt_f), list(tgt_y))), batch_size=batch_size,
                                                 shuffle=True)
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        len_dataloader = min(len(src_loader), len(tgt_loader))

        total_transfer_loss = 0
        for i in range(len_dataloader):
            src_f, src_y = src_iter.next()
            tgt_f, tgt_y = tgt_iter.next()
            if src_f.shape[0] != tgt_f.shape[0]:
                break

            total_transfer_loss += self.jmmd_loss((src_f, F.softmax(src_y, dim=1)), (tgt_f, F.softmax(tgt_y, dim=1)))

        return total_transfer_loss / len_dataloader

    def negative_entropy_from_logits(self, logits):
        neg_entropies = torch.sum(
            torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1), dim=1
        )
        return torch.mean(neg_entropies)


class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix
    Gaussian Kernel k is defined by
    .. math::
        k(x_1, x_2) = \exp \left( - \dfrac{\| x_1 - x_2 \|^2}{2\sigma^2} \right)
    where :math:`x_1, x_2 \in R^d` are 1-d tensors.
    Gaussian Kernel Matrix K is defined on input group :math:`X=(x_1, x_2, ..., x_m),`
    .. math::
        K(X)_{i,j} = k(x_i, x_j)
    Also by default, during training this layer keeps running estimates of the
    mean of L2 distances, which are then used to set hyperparameter  :math:`\sigma`.
    Mathematically, the estimation is :math:`\sigma^2 = \dfrac{\alpha}{n^2}\sum_{i,j} \| x_i - x_j \|^2`.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and use a fixed :math:`\sigma` instead.
    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``
    Inputs:
        - X (tensor): input group :math:`X`
    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


class JointMultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""
    Args:
        kernels (tuple(tuple(torch.nn.Module))): kernel functions, where `kernels[r]` corresponds to kernel :math:`k^{\mathcal{L}[r]}`.
        linear (bool): whether use the linear version of JAN. Default: False
        thetas (list(Theta): use adversarial version JAN if not None. Default: None
    Inputs:
        - z_s (tuple(tensor)): multiple layers' activations from the source domain, :math:`z^s`
        - z_t (tuple(tensor)): multiple layers' activations from the target domain, :math:`z^t`
    Shape:
        - :math:`z^{sl}` and :math:`z^{tl}`: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar
    .. note::
        Activations :math:`z^{sl}` and :math:`z^{tl}` must have the same shape.
    .. note::
        The kernel values will add up when there are multiple kernels for a certain layer.
    Examples::
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> layer1_kernels = (GaussianKernel(alpha=0.5), GaussianKernel(1.), GaussianKernel(2.))
        >>> layer2_kernels = (GaussianKernel(1.), )
        >>> loss = JointMultipleKernelMaximumMeanDiscrepancy((layer1_kernels, layer2_kernels))
        >>> # layer1 features from source domain and target domain
        >>> z1_s, z1_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # layer2 features from source domain and target domain
        >>> z2_s, z2_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss((z1_s, z2_s), (z1_t, z2_t))
    """

    def __init__(self, kernels: Sequence[Sequence[nn.Module]], linear: Optional[bool] = True,
                 thetas: Sequence[nn.Module] = None):
        super(JointMultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear
        if thetas:
            self.thetas = thetas
        else:
            self.thetas = [nn.Identity() for _ in kernels]

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        batch_size = int(z_s[0].size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s[0].device)

        kernel_matrix = torch.ones_like(self.index_matrix)
        for layer_z_s, layer_z_t, layer_kernels, theta in zip(z_s, z_t, self.kernels, self.thetas):
            layer_features = torch.cat([layer_z_s, layer_z_t], dim=0)
            layer_features = theta(layer_features)
            kernel_matrix *= sum(
                [kernel(layer_features) for kernel in layer_kernels])  # Add up the matrix of each kernel

        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1)) # s1, s2
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1)) # t1, t2
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size) # s1, t2
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size) # s2, t1
    return index_matrix


class Theta(nn.Module):
    """
    maximize loss respect to :math:`\theta`
    minimize loss respect to features
    """

    def __init__(self, dim: int):
        super(Theta, self).__init__()
        self.grl1 = GradientReverseLayer()
        self.grl2 = GradientReverseLayer()
        self.layer1 = nn.Linear(dim, dim)
        nn.init.eye_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.grl1(features)
        return self.grl2(self.layer1(features))

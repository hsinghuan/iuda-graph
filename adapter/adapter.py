from copy import deepcopy
from torch_geometric.loader import DataLoader


class FullModelSinglegraphAdapter():
    def __init__(self, model, src_data=None, device="cpu"):
        self.device = device
        self.set_model(model)
        self.src_data = src_data
        if self.src_data:
            self.src_data = self.src_data.to(self.device)


    def _adapt_train_epoch(self):
        raise NotImplementedError("Subclass should implement this")

    def _adapt_test_epoch(self):
        raise NotImplementedError("Subclass should implement this")

    def _adapt_train_test(self):
        raise NotImplementedError("Subclass should implement this")

    def adapt(self):
        raise NotImplementedError("Subclass should implement this")

    def set_model(self, model):
        self.model = deepcopy(model).to(self.device)

    def get_model(self):
        return self.model

class FullModelMultigraphAdapter():
    def __init__(self, model, src_train_loader=None, src_val_loader=None, device="cpu"):
        self.device = device
        self.set_model(model)
        self.src_train_loader = src_train_loader
        self.src_val_loader = src_val_loader


    def _adapt_train_epoch(self):
        raise NotImplementedError("Subclass should implement this")

    def _adapt_test_epoch(self):
        raise NotImplementedError("Subclass should implement this")

    def _adapt_train_test(self):
        raise NotImplementedError("Subclass should implement this")

    def adapt(self):
        raise NotImplementedError("Subclass should implement this")

    def set_model(self, model):
        self.model = deepcopy(model).to(self.device)

    def get_model(self):
        return self.model


class DecoupledSinglegraphAdapter():
    def __init__(self, encoder, classifier, src_data=None, device="cpu"):
        self.device = device
        self.set_encoder_classifier(encoder, classifier)
        self.src_data = src_data
        if self.src_data:
            self.src_data = self.src_data.to(self.device)

    def _adapt_train_epoch(self):
        raise NotImplementedError("Subclass should implement this")

    def _adapt_test_epoch(self):
        raise NotImplementedError("Subclass should implement this")

    def _adapt_train_test(self):
        raise NotImplementedError("Subclass should implement this")

    def adapt(self):
        raise NotImplementedError("Subclass should implement this")

    def set_encoder_classifier(self, encoder, classifier):
        self.encoder = deepcopy(encoder).to(self.device)
        self.classifier = deepcopy(classifier).to(self.device)

    def get_encoder_classifier(self):
        return self.encoder, self.classifier


class DecoupledMultigraphAdapter():
    def __init__(self, encoder, classifier, src_train_loader=None, src_val_loader=None, device="cpu"):
        self.device = device
        self.set_encoder_classifier(encoder, classifier)
        self.src_train_loader = src_train_loader
        self.src_val_loader = src_val_loader

    def _adapt_train_epoch(self):
        raise NotImplementedError("Subclass should implement this")

    def _adapt_test_epoch(self):
        raise NotImplementedError("Subclass should implement this")

    def _adapt_train_test(self):
        raise NotImplementedError("Subclass should implement this")

    def adapt(self):
        raise NotImplementedError("Subclass should implement this")

    def set_encoder_classifier(self, encoder, classifier):
        self.encoder = deepcopy(encoder).to(self.device)
        self.classifier = deepcopy(classifier).to(self.device)

    def get_encoder_classifier(self):
        return self.encoder, self.classifier

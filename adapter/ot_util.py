import numpy as np
import time
import ot

def pushforward(X_S, X_T, plan, t):
    print(f'Pushforward to t={t}')
    assert 0 <= t <= 1
    nonzero_indices = np.argwhere(plan > 0)
    weights = plan[plan > 0]
    assert len(nonzero_indices) == len(weights)
    x_t= ( 1-t ) * X_S[nonzero_indices[: ,0]] + t* X_T[nonzero_indices[:, 1]]

    return x_t, weights


def get_OT_plan(X_S, X_T, solver='sinkhorn', weights_S=None, weights_T=None, Y_S=None, numItermax=1e7,
                entropy_coef=1, entry_cutoff=0):

    n, m = len(X_S), len(X_T)
    a = np.ones(n) / n if weights_S is None else weights_S
    b = np.ones(m) / m if weights_T is None else weights_T
    print(f'{n} source data, {m} target data. ')
    dist_mat = ot.dist(X_S, X_T) # .detach().numpy()
    t = time.time()
    if solver == 'emd':
        plan = ot.emd(a, b, dist_mat, numItermax=int(numItermax))
    elif solver == 'sinkhorn':
        plan = ot.sinkhorn(a, b, dist_mat, reg=entropy_coef, numItermax=int(numItermax), stopThr=10e-9)
    elif solver == 'lpl1':
        plan = ot.sinkhorn_lpl1_mm(a, b, Y_S, dist_mat, reg=entropy_coef, numItermax=int(numItermax), stopInnerThr=10e-9)

    if entry_cutoff > 0:
        avg_val = 1 / (n * m)
        print(f'Zero out entries with value < {entry_cutoff}*{avg_val}')
        plan[plan < avg_val * entry_cutoff] = 0

    elapsed = round(time.time() - t, 2)
    print(f"Time for OT calculation: {elapsed}s")
    plan = plan * n

    return plan


def generate_domains(n_inter, feat_s, feat_t, plan=None, entry_cutoff=0):
    print("------------Generate Intermediate domains----------")
    all_domains = []

    if plan is None:
        plan = get_OT_plan(feat_s, feat_t, solver='sinkhorn', entry_cutoff=entry_cutoff)


    for i in range(1, n_inter+1):
        x, weights = pushforward(feat_s, feat_t, plan, i / (n_inter +1))
        all_domains.append(x)
    all_domains.append(feat_t)

    return all_domains
import os
import glob
import numpy as np
import pandas as pd
from numpy import load
from collections import defaultdict

def mse(x, y):
    return np.mean(np.square(x-y))

def truncate_by_g(attribute, g, level=0.01):
    keep_these = np.logical_and(g >= level, g <= 1.-level)
    return attribute[keep_these]

def truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level=0.05):
    orig_g = np.copy(g)

    q_t0 = truncate_by_g(np.copy(q_t0), orig_g, truncate_level)
    q_t1 = truncate_by_g(np.copy(q_t1), orig_g, truncate_level)
    g = truncate_by_g(np.copy(g), orig_g, truncate_level)
    t = truncate_by_g(np.copy(t), orig_g, truncate_level)
    y = truncate_by_g(np.copy(y), orig_g, truncate_level)

    return q_t0, q_t1, g, t, y

def psi_naive(q_t0, q_t1, g, t, y, truncate_level=0.):
    ite = (q_t1 - q_t0)
    return np.mean(truncate_by_g(ite, g, level=truncate_level))

def psi_tmle_cont_outcome(q_t0, q_t1, g, t, y, eps_hat=None, truncate_level=0.05):
    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    g_loss = mse(g, t)
    h = t * (1.0/g) - (1.0-t) / (1.0 - g)
    full_q = (1.0-t)*q_t0 + t*q_t1

    if eps_hat is None:
        eps_hat = np.sum(h*(y-full_q)) / np.sum(np.square(h))

    def q1(t_cf):
        h_cf = t_cf * (1.0 / g) - (1.0 - t_cf) / (1.0 - g)
        full_q = (1.0 - t_cf) * q_t0 + t_cf * q_t1
        return full_q + eps_hat * h_cf

    ite = q1(np.ones_like(t)) - q1(np.zeros_like(t))
    psi_tmle = np.mean(ite)

    ic = h*(y-q1(t)) + ite - psi_tmle
    psi_tmle_std = np.std(ic) / np.sqrt(t.shape[0])
    initial_loss = np.mean(np.square(full_q-y))
    final_loss = np.mean(np.square(q1(t)-y))

    return psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss


def load_data(
        split,
        replication,
        npz_path
        ):
    
    data = load(npz_path + '{}_{}.npz'.format(replication, split))
    q_t0 = data['q_t0'].reshape(-1, 1)
    q_t1 = data['q_t1'].reshape(-1, 1)
    g = data['g'].reshape(-1, 1)
    t = data['t'].reshape(-1, 1)
    y = data['y'].reshape(-1, 1)

    return q_t0, q_t1, g, t, y

def ate(folder, split):
    full_path = os.path.abspath(folder)
    dir_path = os.path.join(full_path, folder, 'processed')

    dict = defaultdict(int)
    tmle_dict = defaultdict(int)

    ufids = sorted(glob.glob("{}/*".format(dir_path)))
    for model in ['baseline', 'targeted_regularization']:
        npz_path = os.path.join(dir_path, ufid, model)
        ufid_simple = pd.Series(np.zeros(len(ufids)))
        ufid_tmle = pd.Series(np.zeros(len(ufids)))
        for j in range(len(ufids)):
            ufid = os.path.basename(ufids[j])

            #get for the _cf file version the ground truth

            all_psi_n, all_psi_tmle = [], []
            for rep in range(25):
                q_t0, q_t1, g, t, y = load_data(split, rep, npz_path)
                


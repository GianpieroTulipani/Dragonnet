import os
import glob
import argparse
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

def psi_naive(q_t0, q_t1, g, truncate_level):
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

def get_estimate(q_t0, q_t1, g, t, y_dragon, truncate_level=0.01):
    psi_n = psi_naive(q_t0, q_t1, g, truncate_level=truncate_level)
    psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(q_t0, q_t1, g, t,
                                                                                              y_dragon,
                                                                                              truncate_level=truncate_level)
    return psi_n, psi_tmle, initial_loss, final_loss, g_loss

def load_truth(scaling_path, ufid):
        cf_suffix = "_cf"
        file_extension = ".csv"
        
        ufid_cf = ufid + cf_suffix + file_extension
        df = pd.read_csv(os.path.join(scaling_path, ufid_cf), index_col='sample_id', header=0, sep=',')

        y0 = df['y0'].values
        y1 = df['y1'].values

        diff = y1 - y0
        return np.mean(diff)

def load_data(
        split,
        npz_path
        ):
    
    data = load(os.path.join(npz_path, f'{split}.npz'))
    q_t0 = data['q_t0'].reshape(-1, 1)
    q_t1 = data['q_t1'].reshape(-1, 1)
    g = data['g'].reshape(-1, 1)
    t = data['t'].reshape(-1, 1)
    y = data['y'].reshape(-1, 1)

    return q_t0, q_t1, g, t, y

def ate(folder, split):
    full_path = os.path.abspath(folder)
    scaling_path = os.path.join(full_path, 'raw', 'train_scaling')
    processed_path = os.path.join(full_path, 'processed')

    mae_dict = defaultdict(float)
    rmse_dict = defaultdict(float)

    mae_tmle_dict = defaultdict(float)
    rmse_tmle_dict = defaultdict(float)

    ufids = sorted(glob.glob(f"{processed_path}/*"))
    for model in ['baseline', 'targeted_regularization']:

        mae_simple = pd.Series(np.zeros(len(ufids)))
        rmse_simple = pd.Series(np.zeros(len(ufids)))
        mae_tmle = pd.Series(np.zeros(len(ufids)))
        rmse_tmle = pd.Series(np.zeros(len(ufids)))

        for j, ufid_path in enumerate(ufids):
            ufid = os.path.basename(ufid_path)
            npz_path = os.path.join(processed_path, ufid, model)

            ground_truth = load_truth(scaling_path, ufid)

            q_t0, q_t1, g, t, y = load_data(split, npz_path)

            psi_n, psi_tmle, initial_loss, final_loss, g_loss = get_estimate(
                q_t0, q_t1, g, t, y, truncate_level=0.01
            )

            mae_simple[j] = abs(psi_n - ground_truth)
            mae_tmle[j] = abs(psi_tmle - ground_truth)

            rmse_simple[j] = (psi_n - ground_truth)**2
            rmse_tmle[j] = (psi_tmle - ground_truth)**2

        """
        print(f'MAE results for model: {model}\n')
        print(f'mae_simple: {mae_simple}')
        print(f'mae_tmle: {mae_tmle}\n')
        print('RMSE results for model: {model}\n')
        print(f'rmse_simple: {rmse_simple}')
        print(f'rmse_tmle: {rmse_tmle}\n')
        """

        mae_dict[model] = mae_simple.mean()
        rmse_dict[model] = np.sqrt(rmse_simple.mean())

        mae_tmle_dict[model] = mae_tmle.mean()
        rmse_tmle_dict[model] = np.sqrt(rmse_tmle.mean())

    return mae_dict, mae_tmle_dict, rmse_dict, rmse_tmle_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute ATE estimates")

    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['train', 'val', 'test'],
        help="Dataset split to evaluate (train | val | test)"
    )

    parser.add_argument(
        '--folder',
        type=str,
        default='data',
        help="Root data folder (default: data)"
    )

    args = parser.parse_args()

    mae_naive, mae_tmle, rmse_naive, rmse_tmle = ate(args.folder, args.split)

    print("Naive ATE metrics:")
    for k in mae_naive:
        print(f"  {k} -> MAE: {mae_naive[k]:.4f}, RMSE: {rmse_naive[k]:.4f}")

    print("\nTMLE ATE metrics:")
    for k in mae_tmle:
        print(f"  {k} -> MAE: {mae_tmle[k]:.4f}, RMSE: {rmse_tmle[k]:.4f}")
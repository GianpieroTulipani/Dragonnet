import os
import gc
import glob
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .dragonnet import DatasetACIC, Dragonnet, dragonnet_loss, tarreg_loss, regression_loss

def load_and_format_covariates(file_path):
    df = pd.read_csv(file_path, index_col='sample_id', header=0, sep=',')
    return df

def load_treatment_and_outcome(covariates, file_path, standardize=True):
    output = pd.read_csv(file_path, index_col='sample_id',header=0, sep=',')

    dataset = covariates.join(output, how='inner')
    t = dataset['z'].values
    y = dataset['y'].values
    x = dataset.values[:,:-2]
    if standardize:
        normal_scalar = StandardScaler()
        x = normal_scalar.fit_transform(x)
    return t.reshape(-1,1), y.reshape(-1,1), x

def _split_output(yt_hat, t, y, y_scaler, x):
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].copy())
    g = yt_hat[:, 2].copy()

    if yt_hat.shape[1] == 4:
        eps = yt_hat[:, 3][0]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    y = y_scaler.inverse_transform(y.copy())

    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'eps': eps}

def compute_test_mse(model, loader, device):
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch, t_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            t_batch = t_batch.to(device)

            concat_pred = model(x_batch)
            concat_true = torch.cat([y_batch, t_batch], dim=1)

            mse = regression_loss(concat_true, concat_pred)
            total_mse += mse.item()
            total_samples += x_batch.size(0)

    return total_mse / total_samples

def train_and_predict(
        t,
        y_unscaled,
        x,
        targeted_regularization=True,
        ratio=1.,
        val_split=0.2,
        batch_size=512,
        num_epochs=300,
        patience=40,
        runs=25,
        device=None
):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y_unscaled)

    train_outputs = []
    val_outputs = []
    test_outputs = []

    for run in range(runs):
        np.random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed(run)
        torch.cuda.manual_seed_all(run)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        num_workers=os.cpu_count()
        
        t_trainval, t_test, y_trainval, y_test, x_trainval, x_test = train_test_split(t, y, x, test_size=val_split, random_state=run)
        t_train, t_val, y_train, y_val, x_train, x_val = train_test_split(t_trainval, y_trainval, x_trainval, test_size=val_split, random_state=run)

        train_dataset = DatasetACIC(x_train, t_train, y_train)
        val_dataset = DatasetACIC(x_val, t_val, y_val)
        test_dataset = DatasetACIC(x_test, t_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = Dragonnet(x_train.shape[1]).to(device)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=1e-5,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.01
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            threshold=0.0,
            cooldown=0,
            min_lr=0.0
        )

        loss_fn = tarreg_loss if targeted_regularization else dragonnet_loss

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            train_samples = 0
            train_mse = 0.0
            train_loss = 0.0

            for x_batch, y_batch, t_batch in tqdm(train_loader, desc=f"Run {run + 1}, Epoch {epoch+1}", leave=False):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                t_batch = t_batch.to(device)

                optimizer.zero_grad()
                concat_pred = model(x_batch)

                concat_true = torch.cat([y_batch, t_batch], dim=1)
                loss = loss_fn(ratio, concat_true, concat_pred) if targeted_regularization else loss_fn(concat_true, concat_pred)
                mse = regression_loss(concat_true, concat_pred)
                train_samples += x_batch.size(0)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_mse += mse.item()
            
            avg_train_loss = train_loss / train_samples
            avg_train_mse = train_mse / train_samples
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1} - Average train loss per sample: {avg_train_loss:.4f}, Average mse per sample: {avg_train_mse:.4f},  LR = {current_lr:.2e}")

            model.eval()
            val_mse=0.0
            val_loss=0.0
            val_samples = 0

            with torch.no_grad():
                for x_batch, y_batch, t_batch in tqdm(val_loader, desc=f"Run {run+1} Epoch {epoch+1} Val", leave=False):
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    t_batch = t_batch.to(device)

                    concat_pred = model(x_batch)
                    concat_true = torch.cat([y_batch, t_batch], dim=1)

                    loss = loss_fn(ratio, concat_true, concat_pred) if targeted_regularization else loss_fn(concat_true, concat_pred)
                    mse = regression_loss(concat_true, concat_pred)
                    val_mse += mse.item()
                    val_loss += loss.item()
                    val_samples += x_batch.size(0)
                
            avg_val_loss = val_loss / val_samples
            avg_val_mse = val_mse / val_samples

            print(f"Epoch {epoch+1} - Average val loss per sample: {avg_val_loss:.4f}, Average val mse: {avg_val_mse:.4f}")

            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        model.load_state_dict(best_model_state)
        model.eval()

        test_mse = compute_test_mse(model, test_loader, device)
        print(f"Run {run+1} - Test regression MSE: {test_mse:.4f}")

        def predict_loader(loader):
            outputs = []
            with torch.no_grad():
                for x_batch, _, _ in loader:
                    x_batch = x_batch.to(device)
                    outputs.append(model(x_batch))
            return torch.cat(outputs, dim=0)
        
        yt_hat_train = predict_loader(train_loader)
        yt_hat_val = predict_loader(val_loader)
        yt_hat_test = predict_loader(test_loader)

        train_outputs.append(_split_output(
            yt_hat_train.cpu().numpy(), t_train, y_train, y_scaler, x_train)
        )
        val_outputs.append(_split_output(
            yt_hat_val.cpu().numpy(), t_val, y_val, y_scaler, x_val)
        )
        test_outputs.append(_split_output(
            yt_hat_test.cpu().numpy(), t_test, y_test, y_scaler, x_test)
        )

        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    return train_outputs, val_outputs, test_outputs
                
def run_acic(
    folder,
    ratio=1.
):
    full_path = os.path.abspath(folder)
    covariate_csv = os.path.join(full_path, 'raw', 'x.csv')
    x_raw = load_and_format_covariates(covariate_csv)

    simulation_dir = os.path.join(full_path, 'raw', 'train_scaling')
    simulation_files = sorted(glob.glob("{}/*".format(simulation_dir)))

    for simulation_file in simulation_files:
        cf_suffix = "_cf"
        file_extension = ".csv"
        if simulation_file.endswith(cf_suffix + file_extension):
            continue

        ufid = os.path.basename(simulation_file)[:-4]
        
        t, y, x = load_treatment_and_outcome(x_raw, simulation_file)

        for is_targeted_regularization in [True, False]:
            print("Is targeted regularization: {}".format(is_targeted_regularization))

            train_outputs, val_outputs, test_outputs = train_and_predict(t, y, x,        
                                                                         targeted_regularization=is_targeted_regularization,
                                                                         ratio=ratio,
                                                                         val_split=0.2,
                                                                         batch_size=512
                                                                         )
            if is_targeted_regularization:
                output_dir = os.path.join(full_path, 'processed', ufid, "targeted_regularization")
            else:
                output_dir = os.path.join(full_path, 'processed', ufid, "baseline")
            
            os.makedirs(output_dir, exist_ok=True)
            for num, output in test_outputs:
                np.savez_compressed(os.path.join(output_dir, "{}_test.npz".format(num)),
                                    **output)
            
            for num, output in enumerate(train_outputs):
                np.savez_compressed(os.path.join(output_dir, "{}_train.npz".format(num)),
                                    **output)
                
            for num, output in enumerate(val_outputs):
                np.savez_compressed(os.path.join(output_dir, "{}_val.npz".format(num)),
                                    **output)

#$env:KMP_DUPLICATE_LIB_OK="TRUE"

if __name__ == '__main__':
    run_acic('data')
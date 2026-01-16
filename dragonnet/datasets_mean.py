import os
import glob
import pandas as pd

def load_and_format_covariates(file_path):
    df = pd.read_csv(file_path, index_col='sample_id', header=0, sep=',')
    return df

def load_outcome_mean(covariates, file_path, standardize=True):
    output = pd.read_csv(file_path, index_col='sample_id',header=0, sep=',')

    dataset = covariates.join(output, how='inner')

    return dataset['y'].mean()

def compute_dataset_mean(folder):
    datsets_mean = []
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

        datsets_mean.append(load_outcome_mean(x_raw, simulation_file))

    return datsets_mean

if __name__ == '__main__':
    datasets_mean = compute_dataset_mean('data')
    for i, d in enumerate(datasets_mean):
        print(f'{i}: {d}')
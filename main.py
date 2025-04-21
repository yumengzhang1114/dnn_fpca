from data_generate import *
from fit import Iter_train, Iter_train_data
import pickle
from utils import write_text, load_pickle, save_pickle
import time
import pandas as pd
import numpy as np

from irrnn import fit_irrnn, get_coord
from irrnn import compute_threshold, get_ols_est, fit

def main():
    args = get_args()
    if args.seed is not None:
        set_seed(args.seed)

    device = args.device

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X = pd.read_pickle("/Users/yumeng/Desktop/ADNI_MRI_data/ADNI_MRI/x_centered.pkl")
    X = X.to_numpy()
    print(X.shape)
    R = args.rank
    VV = X.shape[1]
    coord = pd.read_csv("/Users/yumeng/Desktop/ADNI_MRI_data/ADNI_MRI/coord.csv")
    coord = coord[["x", "y", "z"]]
    coord = coord.to_numpy()


    data = pd.read_pickle("/Users/yumeng/Desktop/ADNI_MRI_data/ADNI_MRI/svd_results.pickle")
    top3_eigenfunctions = data["top3_eigenfunctions"]
    phi_temp = top3_eigenfunctions[1].reshape(1, -1)

    print("first eigenfunctions shape:", phi_temp.shape)
    xi_temp = X @ phi_temp.T


    print(xi_temp.shape)
    print(phi_temp.shape)

    xi_hat_save, phi_hat_save, xi_loss, phi_loss = Iter_train_data(xi_temp, phi_temp, X, args, coord)
    result = {"xi_hat_save": xi_hat_save, "phi_hat_save": phi_hat_save,
              "xi_loss": xi_loss, "phi_loss": phi_loss}
    save_pickle(result, f'result_true_initial.pickle')


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
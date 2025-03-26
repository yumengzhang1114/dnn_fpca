from data_generate import *
from fit import Iter_train,Iter_train_data
import pickle
from utils import write_text, load_pickle, save_pickle
import time
import pandas as pd
import numpy as np


def main():
    x = pd.read_pickle("x_centered.pkl")
    x = x.to_numpy()
    print("x shape:", x.shape)

    X_centered = x - np.mean(x, axis=0)
    # svd
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
    top3_eigenfunctions = VT[:3, :]
    top100_eigenfunctions = VT[:100, :]
    print("Shape of top3_eigenfunctions:", top3_eigenfunctions.shape)
    print("Shape of top100_eigenfunctions:", top100_eigenfunctions.shape)
    singularvalues = S
    singularvalue1, singularvalue2, singularvalue3 = S[:3]
    N = X_centered.shape[0]
    eigenvalue1 = singularvalue1 ** 2 / (N - 1)
    eigenvalue2 = singularvalue2 ** 2 / (N - 1)
    eigenvalue3 = singularvalue3 ** 2 / (N - 1)
    print("Singular values:", singularvalue1, singularvalue2, singularvalue3)
    print("Eigenvalues:", eigenvalue1, eigenvalue2, eigenvalue3)
    svd_results = {
        "top3_eigenfunctions": top3_eigenfunctions,   # shape: (3, V)
        "top100_eigenfunctions": top100_eigenfunctions, # shape: (100, V)
        "singular_values": (singularvalue1, singularvalue2, singularvalue3),
        "eigenvalues": (eigenvalue1, eigenvalue2, eigenvalue3),
        "singularvalues":singularvalues
    }
    save_pickle(svd_results,f'svd_results.pickle')

if __name__ == "__main__":
    main()
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import os
import copy
from data_generate import *
from fit import Iter_train
import pickle
from utils import write_text, load_pickle, save_pickle
import time

def Simulation(args, seed):
    V = args.n_voxels
    VV = V ** 3
    R = args.rank
    N = args.n_indivs

    xi, phi, X = data_gen(N, V, R, seed)
    truth = {'xi': xi, 'phi': phi, 'X': X}
    save_pickle(truth, f'{args.prefix}_true.pickle')
    mu = X.mean(axis=0)
    X_temp = X - mu

    H = np.random.randn(VV, R)
    q, r = np.linalg.qr(H, mode="reduced")
    phi_temp = q.T
    xi_temp = X_temp @ phi_temp.T

    xi_hat_save, phi_hat_save, xi_loss, phi_loss = Iter_train(xi_temp, phi_temp, X_temp, args)

    return (xi_hat_save, phi_hat_save, xi_loss, phi_loss)


def main():
    args = get_args()
    start_time = time.time()

    if args.seed is not None:
        set_seed(args.seed)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    
    seed_list = [(2 * i - 1) ** 25 for i in range(1, 51)]

    #seed_list = [6, 7, 8, 9, 10]
        
    with ProcessPoolExecutor(max_workers=len(seed_list)) as executor:
        futures = {}
        for i, seed in enumerate(seed_list):
            args_copy = copy.deepcopy(args)
            args_copy.prefix = f"Simulation{i}"
            futures[executor.submit(Simulation, args_copy, seed)] = args_copy.prefix

        for future in as_completed(futures):
            prefix = futures[future]
            try:
                xi_hat_save, phi_hat_save, xi_loss, phi_loss = future.result()
                result = {
                    "xi_hat_save": xi_hat_save,
                    "phi_hat_save": phi_hat_save,
                    "xi_loss": xi_loss,
                    "phi_loss": phi_loss
                }
                save_pickle(result, f'{prefix}_result.pickle')
                print(f"{prefix} completed.")
            except Exception as exc:
                print(f'{prefix} generated an exception: {exc}')

    end_time = time.time()
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()


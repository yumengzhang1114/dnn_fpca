from data_generate import *
from fit import Iter_train,Iter_train_data
import pickle
from utils import write_text, load_pickle, save_pickle
import time
import pandas as pd
import numpy as np

from irrnn import fit_irrnn, get_coord
from irrnn import compute_threshold, get_ols_est, fit




def cal_population_mean(x,coord):
    args = get_args()
    if args.seed is not None:
        set_seed(args.seed)
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_widths = (args.width,) * args.depth
    pred = fit(
        value=x, coord=coord, img_shape=None,
        hidden_widths=hidden_widths,
        activation=args.activation,
        lr=args.lr, batch_size=args.batch_size,
        epochs=args.epochs,
        prefix=args.prefix,
        device=args.device)
    x_mean_pred = pred
    return x_mean_pred


def main():

    x = pd.read_csv("subject_means.csv")
    coord = pd.read_csv("coord.csv")
    coord = coord[["x","y","z"]]
    x = x.to_numpy()
    coord =  coord.to_numpy()
    start_time = time.time()  # Record the start time
    x_mean_pred = cal_population_mean(x,coord)
    end_time = time.time()
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")
    df = pd.DataFrame(x_mean_pred)
    df.to_csv("x_mean_pred.csv", index=False)



if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')
    main()

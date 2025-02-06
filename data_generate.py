import argparse
import random
import numpy as np
import torch

def get_coord(shape):
    coord = [np.linspace(0, 1, n) for n in shape]
    coord = np.stack(np.meshgrid(*coord, indexing='ij'), -1)
    coord = coord.reshape(-1, coord.shape[-1])
    return coord

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def data_gen(N,V,R,seed):

    set_seed(seed)
    img_shape=(V,V,V)
    VV=V**3
    s=get_coord(img_shape)

    STS=s.T@s
    U,eigenvalues, beta = np.linalg.svd(STS)

    # generate phi
    phi=beta@s.T

    for i in range(R):
        phi[i,:]=phi[i,:]/np.sqrt(phi[i,:]@phi[i,:].T)

    #generate xi
    xi=np.random.normal(loc=0.0, scale=1.0, size=(N, R))

    #(4,2,1)
    xi=xi*np.array([2,np.sqrt(2),1])

    xi=xi-xi.mean(axis=0)
    xi=xi*np.sqrt(VV)

    ### With noise N(0,1)
    Noise= np.random.normal(loc=0.0, scale=1, size=(N, VV))
    X=xi@phi+Noise
    return(xi,phi,X)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,default='Synthetic')
    parser.add_argument('--prefix', type=str,default='Simulation')
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--activation', type=str, default='leaky')
    parser.add_argument('--alpha-threshold', type=float, default=0.05)
    parser.add_argument('--n-permute', type=int, default=100)
    parser.add_argument('--max-iter', type=int, default=10)
    parser.add_argument('--n-indivs', type=int, default=10)
    parser.add_argument('--n-voxels', type=int, default=16)
    parser.add_argument('--rank', type=int, default=3)
    parser.add_argument('--n-states', type=int, default=11)
    parser.add_argument('--alpha-states', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--n-jobs', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-m', '--message', type=str, default=None)
    parser.add_argument('-f')
    args = parser.parse_args()
    return args


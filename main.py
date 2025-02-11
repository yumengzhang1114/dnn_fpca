
from data_generate import *
from fit import Iter_train
import pickle
from utils import write_text, load_pickle, save_pickle
import time

def Simulation(args,seed):

    V=args.n_voxels
    VV=V**3
    R=args.rank
    N=args.n_indivs


    # generate data
    xi, phi, X = data_gen(N, V,R,seed)
    truth={'xi': xi, 'phi': phi, 'X': X}
    save_pickle(truth, f'{args.prefix}_true.pickle')
    
    
    mu=X.mean(axis=0)
    X_temp=X-mu

    #initialise xi and phi randomly
    H=np.random.randn(VV, VV)
    q,r=np.linalg.qr(H)
    phi_temp=q[:R,:]
    xi_temp=X_temp@phi_temp.T

    xi_hat_save, phi_hat_save, xi_loss, phi_loss = Iter_train(xi_temp, phi_temp, X_temp, args)
    

    return (xi_hat_save,phi_hat_save,xi_loss,phi_loss)



def main():
    args = get_args()
    start_time = time.time()  # Record the start time
    if args.seed is not None:
        set_seed(args.seed)

    device = args.device

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)
    seed_list = [6,7,8,9,10]
    for i in range(5):
        args.prefix="Simulation"+str(i)
        xi_hat_save,phi_hat_save,xi_loss,phi_loss= Simulation(args,seed=seed_list[i])
        result={"xi_hat_save":xi_hat_save,"phi_hat_save":phi_hat_save,
            "xi_loss":xi_loss,"phi_loss":phi_loss}
        save_pickle(result,f'{args.prefix}_result.pickle')
    end_time = time.time()
    print(f"Total execution time: {(end_time - start_time)/60:.2f} minutes")

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')
    main()

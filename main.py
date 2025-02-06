
from data_generate import *
from fit import Iter_train
import pickle

def Simulation(args,seed):

    V=args.n_voxels
    VV=V**3
    R=args.rank

    # generate data
    xi, phi, X = data_gen(10, V,R,seed)
    mu=X.mean(axis=0)
    X_temp=X-mu

    #initialise xi and phi randomly
    H=np.random.randn(VV, VV)
    q,r=np.linalg.qr(H)
    phi_temp=q[:R,:]
    xi_temp=X_temp@phi_temp.T

    xi_hat_save, phi_hat_save, xi_loss, phi_loss = Iter_train(xi_temp, phi_temp, X_temp, args)
    print("xi_hat_save: ",xi_hat_save)
    print("phi_hat_save: ",phi_hat_save)
    print("xi_loss: ",xi_loss)
    print("phi_loss: ",phi_loss)

    return (xi_hat_save,phi_hat_save,xi_loss,phi_loss)



def main():
    args = get_args()
    seed_list = [6,7,8,9,10]
    simulation_result = []
    for i in range(len(seed_list)):
        print("seed_list: ",seed_list[i])
        print("simulation" + str(i))
        simulation = Simulation(args,seed_list[i])
        simulation_result.append(simulation)
    return simulation_result


if __name__ == '__main__':
    simulation_result = main()

    with open("list.pkl", "wb") as f:
        pickle.dump(simulation_result, f)

    # Load the data back
    with open("list.pkl", "rb") as f:
        loaded_list = pickle.load(f)

    print(loaded_list)

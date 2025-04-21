from data_generate import get_coord
import numpy as np

from irrnn import fit_irrnn, get_coord
from irrnn import compute_threshold, get_ols_est, fit


def get_orthog_phi(xi, phi):
    # SVD for xi
    xi_U, D_xi, xi_Vt = np.linalg.svd(xi, full_matrices=False)
    r_xi = np.count_nonzero(D_xi)
    xi_U, xi_Vt, xi_D = xi_U[:, :r_xi], xi_Vt[:r_xi, :], np.diag(D_xi[:r_xi])

    # SVD for phi
    phi_U, D_phi, phi_Vt = np.linalg.svd(phi, full_matrices=False)
    r_phi = np.count_nonzero(D_phi)
    phi_U, phi_Vt, phi_D = phi_U[:, :r_phi], phi_Vt[:r_phi, :], np.diag(D_phi[:r_phi])

    # SVD for M
    M = xi_D @ xi_Vt @ phi_U @ phi_D
    U, D_M, Vt = np.linalg.svd(M, full_matrices=False)
    r_M = np.count_nonzero(D_M)

    U, Vt, D3 = U[:, :r_M], Vt[:r_M, :], np.diag(D_M[:r_M])
    return xi_U @ U @ D3, Vt @ phi_Vt

def Iter_train_data(xi_temp,phi_temp,X,args,coord):
    iteration=0
    R=args.rank
    N=X.shape[0]
    VV=X.shape[1]
    hidden_widths = (args.width,) * args.depth

    # keep xi_hat, phi_hat, xi_loss and phi loss for each iteration
    xi_hat_save=[]
    phi_hat_save=[]
    xi_loss=[]
    phi_loss=[]

    xi_hat = xi_temp.copy()
    phi_hat = phi_temp.copy()

    phi_loss_temp = 100000
    xi_loss_temp = 100000

    while iteration < args.max_iter and (
            iteration < args.min_iter or
            phi_loss_temp > 0.005 or xi_loss_temp > 0.005):

        print(f"Iteration {iteration}")

        if iteration == 0:
            cur_epochs = 100
            warm_start_f = False
        else:
            cur_epochs = 30
            warm_start_f = True

        for r in range(R):
            y = X
            xi_temp[:, r] = (y @ phi_temp[r, :].T) / np.sum(phi_temp[r, :] ** 2)
            x = (xi_temp[:, r]).reshape((N, 1))
            print(x.shape)
            beta_temp = (x.T @ y) / (x.T @ x)
            print(beta_temp.shape)

            pred = fit(
                value=beta_temp.T, coord=coord, img_shape=None,
                hidden_widths=hidden_widths,
                activation=args.activation,
                lr=args.lr, batch_size=args.batch_size,
                epochs=cur_epochs,
                prefix=args.prefix,device = args.device, warm_start = warm_start_f)
            phi_temp[r, :] = pred.T


    ## Get orthogonal phi
        #xi_temp,phi_temp=get_orthog_phi(xi_temp,phi_temp)
        iteration=iteration + 1
        phi_temp = phi_temp/np.linalg.norm(phi_temp)
        print(np.linalg.norm(phi_temp))

        phi_loss_temp=np.sum((phi_hat-phi_temp)**2)/(np.sum(phi_hat**2)+1)
        phi_loss.append(phi_loss_temp)

        xi_loss_temp=np.sum((xi_hat-xi_temp)**2)/(np.sum(xi_hat**2)+1)
        xi_loss.append(xi_loss_temp)
        print("phi loss" + str(phi_loss_temp))
        print("xi loss" + str(xi_loss_temp))

        phi_hat_save.append(phi_temp.copy())
        xi_hat_save.append(xi_temp.copy())

        phi_hat=phi_temp.copy()
        xi_hat=xi_temp.copy()

        if iteration==args.max_iter:
            print("Max iteration reaches")

    return(xi_hat_save,phi_hat_save,xi_loss,phi_loss)

def Iter_train(xi_temp,phi_temp,X, args):
    iteration=0
    R=args.rank
    N=args.n_indivs
    V=args.n_voxels
    VV=V**3
    hidden_widths = (args.width,) * args.depth
    img_shape=(V,V,V)
    s=get_coord(img_shape)

    # keep xi_hat, phi_hat, xi_loss and phi loss for each iteration
    xi_hat_save=[]
    phi_hat_save=[]
    xi_loss=[]
    phi_loss=[]

    xi_hat = xi_temp
    phi_hat = phi_temp

    phi_loss_temp = 100000
    xi_loss_temp = 100000

    while iteration < args.max_iter and (iteration < args.min_iter or phi_loss_temp > 0.005 or xi_loss_temp > 0.005):

        for r in range(R):
            y = X-np.delete(xi_temp, r, axis=1)@(np.delete(phi_temp, r, axis=0))
            xi_temp[:,r]=(y@phi_temp[r,:].T)/np.sum(phi_temp[r,:]**2)

            x=(xi_temp[:,r]).reshape((N,1))
            beta_temp=np.linalg.inv(x.T@x)@x.T@y

            pred= fit(
                value=beta_temp.T, coord=s, img_shape=img_shape,
                hidden_widths=hidden_widths,
                activation=args.activation,
                lr=args.lr, batch_size=args.batch_size,
                epochs=args.epochs,
                prefix=args.prefix,
                device=args.device)
            phi_temp[r,:]=pred.T

    ## Get orthogonal phi
        #xi_temp,phi_temp=get_orthog_phi(xi_temp,phi_temp)
        iteration=iteration + 1

     ## flip sign
        for r in range(R):
            if (xi_temp[:,r]).T@xi_hat[:,r]<0:
                xi_temp[:,r] = -xi_temp[:,r]
                phi_temp[r,:] = -phi_temp[r,:]

        phi_loss_temp=np.sum((phi_hat-phi_temp)**2)/(np.sum(phi_hat**2)+1)
        phi_loss.append(phi_loss_temp)

        xi_loss_temp=np.sum((xi_hat-xi_temp)**2)/(np.sum(xi_hat**2)+1)
        xi_loss.append(xi_loss_temp)
        print(phi_loss_temp)
        print(xi_loss_temp)
        phi_hat_save.append(phi_temp)
        xi_hat_save.append(xi_temp)

        phi_hat=phi_temp
        xi_hat=xi_temp

        if iteration==args.max_iter:
            print("Max iteration reaches")

    return(xi_hat_save,phi_hat_save,xi_loss,phi_loss)




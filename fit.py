from data_generate import get_coord
import numpy as np

from irrnn import fit_irrnn, get_coord
from irrnn import compute_threshold, get_ols_est, fit

def get_orthog_phi(xi,phi):

    xi_U, D, xi_V=np.linalg.svd(xi)
    ind=np.nonzero(D)
    xi_D=np.diag(D[ind])
    r=(D[ind]).shape[0]
    xi_U=xi_U[:,:r]
    xi_V=xi_V[:r,:]

    phi_U,D,phi_V=np.linalg.svd(phi)
    ind=np.nonzero(D)
    phi_D=np.diag(D[ind])
    r=(D[ind]).shape[0]
    phi_U=phi_U[:,:r]
    phi_V=phi_V[:r,:]


    M=xi_D@xi_V@phi_U@phi_D
    U,D,V=np.linalg.svd(M)
    ind=np.nonzero(D)
    D3=np.diag(D[ind])
    r=(D[ind]).shape[0]
    U=U[:,:r]
    V=V[:r,:]

    return (xi_U@U@D3   ,V@phi_V )

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
        xi_temp,phi_temp=get_orthog_phi(xi_temp,phi_temp)
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

        phi_hat_save.append(phi_temp)
        xi_hat_save.append(xi_temp)

        phi_hat=phi_temp
        xi_hat=xi_temp

        if iteration==args.max_iter:
            print("Max iteration reaches")

    return(xi_hat_save,phi_hat_save,xi_loss,phi_loss)
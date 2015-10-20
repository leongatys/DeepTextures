import numpy as np
import pdb

def gram_mse_loss(activations, target_gram_matrix, weight=1., linear_transform=None):
    '''
    This function computes an elementwise mean squared distance between the gram matrices of the source and the generated image.

    :param activations: the network activations in response to the image that is generated
    :param target_gram_matrix: gram matrix in response to the source image
    :param weight: scaling factor for the loss function
    :param linear_transform: linear transform that is applied to the feature vector at all positions before gram matrix computation
    :return: mean squared distance between normalised gram matrices and gradient wrt activations
    '''

    N = activations.shape[1]
    fm_size = np.array(activations.shape[2:])
    M = np.prod(fm_size)
    G_target = target_gram_matrix
    if linear_transform == None:
        F = activations.reshape(N,-1) 
        G = np.dot(F,F.T) / M
        loss = float(weight)/4 * ((G - G_target)**2).sum() / N**2
        gradient = (weight * np.dot(F.T, (G - G_target)).T / (M * N**2)).reshape(1, N, fm_size[0], fm_size[1])
    else: 
        F = np.dot(linear_transform, activations.reshape(N,-1))
        G = np.dot(F,F.T) / M
        loss = float(weight)/4 * ((G - G_target)**2).sum() / N**2
        gradient = (weight * np.dot(linear_transform.T, np.dot(F.T, (G - G_target)).T) / (M * N**2)).reshape(1, N, fm_size[0], fm_size[1])
        
    return [loss, gradient]

def meanfm_mse_loss(activations, target_activations, weight=1., linear_transform=None):
    '''
    This function computes an elementwise mean squared distance between the mean feature maps of the source and the generated image.

    :param activations: the network activations in response to the image that is generated
    :param target_activations: the network activations in response to the source image 
    :param weight: scaling factor for the loss function
    :param linear_transform: linear transform that is applied to the feature vector at all positions before gram matrix computation
    :return: mean squared distance between mean feature maps and gradient wrt activations
    '''

    N = activations.shape[1]
    fm_size = np.array(activations.shape[2:])
    M = np.prod(fm_size)
    
    target_fm_size = np.array(target_activations.shape[2:])
    M_target = np.prod(target_fm_size)
    if linear_transform==None:
        target_mean_fm = target_activations.reshape(N,-1).sum(1) / M_target
        mean_fm = activations.reshape(N,-1).sum(1) / M 
        f_val = float(weight)/2 * ((mean_fm - target_mean_fm)**2).sum() / N 
        f_grad = weight * (np.tile((mean_fm - target_mean_fm)[:,None],(1,M)) / (M * N)).reshape(1,N,fm_size[0],fm_size[1])
    else:
        target_mean_fm = np.dot(linear_transform, target_activations.reshape(N,-1)).sum(1) / M_target
        mean_fm = np.dot(linear_transform, activations.reshape(N,-1)).sum(1) / M 
        f_val = float(weight)/2 * ((mean_fm - target_mean_fm)**2).sum() / N 
        f_grad = weight * (np.dot(linear_transform.T ,np.tile((mean_fm - target_mean_fm)[:,None],(1,M))) / (M * N)).reshape(1,N,fm_size[0],fm_size[1])
    return [f_val,f_grad]


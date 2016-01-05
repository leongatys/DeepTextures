import numpy as np
import scipy
import caffe
import matplotlib.pyplot as plt
from IPython.display import display,clear_output

class constraint(object):
    '''
    Object that contains the constraints on a particular layer for the image synthesis.
    '''

    def __init__(self, loss_functions, parameter_lists):
        self.loss_functions = loss_functions
        self.parameter_lists = parameter_lists
  
def get_indices(net, constraints):
    '''
    Helper function to pick the indices of the layers included in the loss function from all layers of the network.
    
    :param net: caffe.Classifier object defining the network
    :param contraints: dictionary where each key is a layer and the corresponding entry is a constraint object
    :return: list of layers in the network and list of indices of the loss layers in descending order
    '''

    indices = [ndx for ndx,layer in enumerate(net.blobs.keys()) if layer in constraints.keys()]
    return net.blobs.keys(),indices[::-1]

def show_progress(x, net, title=None, handle=False):
    '''
    Helper function to show intermediate results during the gradient descent.

    :param x: vectorised image on which the gradient descent is performed
    :param net: caffe.Classifier object defining the network
    :param title: optional title of figuer
    :param handle: obtional return of figure handle
    :return: figure handle (optional)
    '''

    disp_image = (x.reshape(*net.blobs['data'].data.shape)[0].transpose(1,2,0)[:,:,::-1]-x.min())/(x.max()-x.min())
    clear_output()
    plt.imshow(disp_image)
    if title != None:
        ax = plt.gca()
        ax.set_title(title)
    f = plt.gcf()
    display()
    plt.show()    
    if handle:
        return f
   
def get_bounds(images, im_size):
    '''
    Helper function to get optimisation bounds from source image.

    :param images: a list of images 
    :param im_size: image size (height, width) for the generated image
    :return: list of bounds on each pixel for the optimisation
    '''

    lowerbound = np.min([im.min() for im in images])
    upperbound = np.max([im.max() for im in images])
    bounds = list()
    for b in range(im_size[0]*im_size[1] * 3):
        bounds.append((lowerbound,upperbound))
    return bounds 

def test_gradient(function, parameters, eps=1e-6):
    '''
    Simple gradient test for any loss function defined on layer output

    :param function: function to be tested, must return function value and gradient
    :param parameters: input arguments to function passed as keyword arguments
    :param eps: step size for numerical gradient evaluation 
    :return: numerical gradient and gradient from function
    '''

    i,j,k,l = [np.random.randint(s) for s in parameters['activations'].shape]
    f1,_ = function(**parameters)
    parameters['activations'][i,j,k,l] += eps
    f2,g = function(**parameters)
    
    return [(f2-f1)/eps,g[i,j,k,l]]

def gram_matrix(activations):
    '''
    Gives the gram matrix for feature map activations in caffe format with batchsize 1. Normalises by spatial dimensions.

    :param activations: feature map activations to compute gram matrix from
    :return: normalised gram matrix
    '''

    N = activations.shape[1]
    F = activations.reshape(N,-1)
    M = F.shape[1]
    G = np.dot(F,F.T) / M
    return G
    
def disp_img(img):
    '''
    Returns rescaled image for display with imshow
    '''
    disp_img = (img - img.min())/(img.max()-img.min())
    return disp_img  

def uniform_hist(X):
    '''
    Maps data distribution onto uniform histogram
    
    :param X: data vector
    :return: data vector with uniform histogram
    '''

    Z = [(x, i) for i, x in enumerate(X)]
    Z.sort()
    n = len(Z)
    Rx = [0]*n
    start = 0 # starting mark
    for i in range(1, n):
        if Z[i][0] != Z[i-1][0]:
            for j in range(start, i):
                Rx[Z[j][1]] = float(start+1+i)/2.0;
            start = i
    for j in range(start, n):
        Rx[Z[j][1]] = float(start+1+n)/2.0;
    return np.asarray(Rx) / float(len(Rx))

def histogram_matching(org_image, match_image, grey=False, n_bins=100):
    '''
    Matches histogram of each color channel of org_image with histogram of match_image

    :param org_image: image whose distribution should be remapped
    :param match_image: image whose distribution should be matched
    :param grey: True if images are greyscale
    :param n_bins: number of bins used for histogram calculation
    :return: org_image with same histogram as match_image
    '''

    if grey:
        hist, bin_edges = np.histogram(match_image.ravel(), bins=n_bins, density=True)
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
        inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
        r = np.asarray(uniform_hist(org_image.ravel()))
        r[r>cum_values.max()] = cum_values.max()    
        matched_image = inv_cdf(r).reshape(org_image.shape) 
    else:
        matched_image = np.zeros_like(org_image)
        for i in range(3):
            hist, bin_edges = np.histogram(match_image[:,:,i].ravel(), bins=n_bins, density=True)
            cum_values = np.zeros(bin_edges.shape)
            cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
            inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
            r = np.asarray(uniform_hist(org_image[:,:,i].ravel()))
            r[r>cum_values.max()] = cum_values.max()    
            matched_image[:,:,i] = inv_cdf(r).reshape(org_image[:,:,i].shape)
        
    return matched_image

def load_image(file_name, im_size, net_model, net_weights, mean, show_img=False):
    '''
    Loads and preprocesses image into caffe format by constructing and using the appropriate network.

    :param file_name: file name of the image to be loaded
    :param im_size: size of the image after preprocessing if float that the original image is rescaled to contain im_size**2 pixels
    :param net_model: file name of the prototxt file defining the network model
    :param net_weights: file name of caffemodel file defining the network weights
    :param mean: mean values for each color channel (bgr) which are subtracted during preprocessing
    :param show_img: if True shows the loaded image before preprocessing
    :return: preprocessed image and caffe.Classifier object defining the network
    '''

    img = caffe.io.load_image(file_name)
    if show_img:
        plt.imshow(img)
    if isinstance(im_size,float):
        im_scale = np.sqrt(im_size**2 /np.prod(np.asarray(img.shape[:2])))
        im_size = im_scale * np.asarray(img.shape[:2])
    batchSize = 1
    with open(net_model,'r+') as f:
        data = f.readlines() 
    data[2] = "input_dim: %i\n" %(batchSize)
    data[4] = "input_dim: %i\n" %(im_size[0])
    data[5] = "input_dim: %i\n" %(im_size[1])
    with open(net_model,'r+') as f:
        f.writelines(data)
    net_mean =  np.tile(mean[:,None,None],(1,) + tuple(im_size.astype(int)))
    #load pretrained network
    net = caffe.Classifier( 
    net_model, net_weights,
    mean = net_mean,
    channel_swap=(2,1,0),
    input_scale=255,)
    img_pp = net.transformer.preprocess('data',img)[None,:]
    return[img_pp, net]

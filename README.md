# DeepTextures
Code to synthesise textures using convolutional neural networks as described in the paper "Texture Synthesis Using Convolutional Neural Networks" (Gatys et al., NIPS 2015) (http://arxiv.org/abs/1505.07376).
More examples of synthesised textures can be found at http://bethgelab.org/deeptextures/.

The IPythonNotebook Example.ipynb contains the code to synthesise the pebble texture shown in Figure 3A (177k parameters) of the revised version of the paper. In the notebook I additionally match the pixel histograms in each colorchannel of the synthesised and original texture, which is not done in the figures in the paper.
#Prerequisites
* To run the code you need a recent version of the [Caffe](https://github.com/BVLC/caffe) deep learning framework and its dependencies (tested with master branch at commit 20c474fe40fe43dee68545dc80809f30ccdbf99b).
* The images in the paper were generated using a normalised version of the [19-layer VGG-Network](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)
described in the work by [Simonyan and Zisserman](http://arxiv.org/abs/1409.1556). The weights in the normalised network are scaled
such that the mean activation of each filter over images and positions is equal to 1.
**The normalised network can be downloaded [here](http://bethgelab.org/media/uploads/deeptextures/vgg_normalised.caffemodel) and has to be copied into the Models/ folder.**

# Disclaimer
This software is published for academic and non-commercial use only. 

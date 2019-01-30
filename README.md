# ImageDepthPredictionKeras
I have abandoned this approach because the up-projection code was taking too long and moved to pytorch, some of this code works. See https://github.com/simonmeister/pytorch-mono-depth for a good off the shelf depth prediction network running in pytorch.


A Keras implementation of the supervised half of the monocular depth prediction network from "Semi-supervised Deep Learning for Monocular Depth Map Prediction". As of November 2018 this is the state of the art approach to depth map prediction from a single image. This repo only implements the supervised half of their network.

#!/bin/bash

# verbose
set -x
###################
# Update items below for each train/test
###################

# training params

wsize=5
hiddenDim=100
reg=0.001
alpha=0.01
nepoch=10
batchSize=20

python two_layer_NN.py --wsize $wsize --hiddenDim $hiddenDim --reg $reg \
                --alpha $alpha --nepoch $nepoch --batchSize $batchSize

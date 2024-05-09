#!/usr/bin/env bash

#CONFIG=$1

# export NCCL_P2P_DISABLE=1

python setup.py develop --no_cuda_ext

CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt Options/LLIE.yml 

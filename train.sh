python setup.py develop --no_cuda_ext

CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt Options/LOL_v2_synthetic.yml 

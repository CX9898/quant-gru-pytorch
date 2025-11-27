export LD_LIBRARY_PATH=/home/zcx/CLionProjects/gru-pytorch/pytouch/lib:$LD_LIBRARY_PATH

python setup.py install

python test_pytorch_gru.py

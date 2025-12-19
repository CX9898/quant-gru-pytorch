mkdir -p ../build
cd ../build
cmake ..
make -j 10
cd ../pytorch

python setup.py build_ext --inplace

python test_quant_gru.py

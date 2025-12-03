cd ..
mkdir build
cd build
cmake ..
make -j 10
cd ../pytouch

python setup.py build_ext --inplace

python test_custom_gru_quantization.py
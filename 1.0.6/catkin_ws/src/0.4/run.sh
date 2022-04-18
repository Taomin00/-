

rm -rf build/
mkdir build
cd build
cmake ..
make -j8
cd ..
clear
bin/run_vo config/default.yaml

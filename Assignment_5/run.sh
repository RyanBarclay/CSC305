#!/bin/bash
mkdir build;
cd build; cmake -DCMAKE_BUILD_TYPE=Debug ..
make; ./assignment5
cd ..
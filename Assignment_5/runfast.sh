#!/bin/bash
mkdir build;
cd build; cmake -DCMAKE_BUILD_TYPE=Release ..
make; ./assignment5

cd ..
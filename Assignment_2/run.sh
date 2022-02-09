#!/bin/bash

cd build; cmake -DCMAKE_BUILD_TYPE=Debug ..
make; ./assignment2
cd ..
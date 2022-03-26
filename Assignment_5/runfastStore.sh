#!/bin/bash
mkdir build;
mkdir output_img;
cd build; cmake -DCMAKE_BUILD_TYPE=Release ..
make; ./assignment5;
cd ..;

vared -p "What would you like to name this file?: " -c filename
mv build/raytrace.png output_img/$filename.png
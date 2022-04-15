#!/bin/bash zsh
mkdir build;
mkdir output_img;
cd build; cmake -DCMAKE_BUILD_TYPE=Release ..
make; ./assignment5;
cd ..;
vared -p "What would you like to name this file?: " -c filename

echo "Would you like to store flat shading (y/n): "
read REPLY
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mv build/flat_shading.png output_img/flat_shading_$filename.png
fi

echo "Would you like to store pv shading (y/n): "
read REPLY
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mv build/pv_shading.png output_img/pv_shading_$filename.png
fi

echo "Would you like to store simple (y/n): "
read REPLY
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mv build/simple.png output_img/simple_$filename.png
fi

echo "Would you like to store wireframe (y/n): "
read REPLY
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mv build/wireframe.png output_img/wireframe_$filename.png
fi

echo "Would you like to store wireframe gif (y/n): "
read REPLY
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mv build/wireframe_render.gif output_img/wireframe_$filename.gif
fi  

echo "Would you like to store flat shading gif (y/n): "
read REPLY
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mv build/flat_shading.gif output_img/flat_shading_$filename.gif
fi

echo "Would you like to store pv shading gif (y/n): "
read REPLY
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mv build/pv_shading.gif output_img/pv_shading_$filename.gif
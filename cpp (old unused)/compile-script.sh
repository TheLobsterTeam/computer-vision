#!/bin/bash

# Script used to compile and run the code.
cmake -Bbuild -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/computer-vision

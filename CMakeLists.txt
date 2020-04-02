cmake_minimum_required(VERSION 3.15)

#SET(OpenCV_DIR /home/noeim/Documents/installation/OpenCV-master/lib/cmake/opencv4)
#find_package( OpenCV REQUIRED )
#include_directories( ${OpenCV_DIR} )
project(Tiny3DRenderer)

set(CMAKE_CXX_STANDARD 20)

add_executable(Tiny3DRenderer main.cpp tgaimage.cpp tgaimage.h main.h model.cpp model.h geometry.h dimension2.cpp dimension2.h dimension3.cpp dimension3.h )
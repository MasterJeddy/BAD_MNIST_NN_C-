cmake_minimum_required(VERSION 3.27)
project(untitled1)

set(CMAKE_CXX_STANDARD 17)

find_package(SFML COMPONENTS graphics audio system window network REQUIRED)
add_executable(untitled1 main.cpp
        MNIST_EigenLoader.cpp
        MNIST_EigenLoader.h
        Neural_Network.cpp
        Neural_Network.h)
target_link_libraries(untitled1 sfml-graphics sfml-audio sfml-system sfml-window sfml-network)

set(CMAKE_CXX_FLAGS "-O3")
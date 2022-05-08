cmake_minimum_required(VERSION 3.4.0)
project(MNIST_Reader)

file (GLOB DATA
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte

    train-images-idx3-ubyte
    train-labels-idx1-ubyte
)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE mnist_reader.hpp)

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

install(FILES ${DATA} DESTINATION build)
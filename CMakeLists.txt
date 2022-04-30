cmake_minimum_required(VERSION 3.0.0)
project(NeuralNetwork VERSION 0.1.0)

file (GLOB SRCS *.cpp *.h)

add_executable(${PROJECT_NAME} ${SRCS})
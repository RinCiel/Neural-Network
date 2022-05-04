cmake_minimum_required(VERSION 3.0.0)
project(neural_network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_GNUtoMS ON)

file (GLOB SOURCES
    src/neural_network.cpp
    src/vector_utils.cpp
)

file (GLOB INCLUDES 
    include/neural_network.h
    include/vector_utils.h
)

add_library(${PROJECT_NAME} ${SOURCES} ${INCLUDES})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS ${PROJECT_NAME} DESTINATION bin/lib)
install(FILES ${INCLUDES} DESTINATION bin/include)
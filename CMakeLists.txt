cmake_minimum_required(VERSION 3.0.0)
project(Neural Network Example VERSION 0.1.0)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(neural_network)

add_executable(${PROJECT_NAME} example.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC neural_network)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
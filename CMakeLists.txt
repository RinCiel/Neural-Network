cmake_minimum_required(VERSION 3.4.0)
project(NeuralNetwork VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(neural_network)
add_subdirectory(mnist)

add_executable(${PROJECT_NAME} example.cpp)

if (MSVC)
	set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "/wd4018")
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC neural_network)
target_link_libraries(${PROJECT_NAME} PUBLIC MNIST_Reader)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION build)
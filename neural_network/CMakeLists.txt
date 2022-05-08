cmake_minimum_required(VERSION 3.4.0)
project(neural_network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (WIN32)
	set(CMAKE_GNUtoMS ON)
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

file (GLOB SOURCES
    src/neural_network.cpp
    src/vector_utils.cpp
)

file (GLOB INCLUDES 
    include/neural_network.h
    include/vector_utils.h
)

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${INCLUDES})

if (MSVC) 
	set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "/wd4244 /wd4267")
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(TARGETS ${PROJECT_NAME} DESTINATION bin/lib)
install(FILES ${INCLUDES} DESTINATION bin/include)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION build)
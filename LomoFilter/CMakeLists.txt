cmake_minimum_required(VERSION 3.17)
project(LomoFilter)

set(CMAKE_CXX_STANDARD 14)

# Where to find CMake modules and OpenCV
set(OpenCV_DIR "D:\\OpenCV\\mingwbuild")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

set(OpenCV_STATIC ON)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(LomoFilter main.cpp)

# add libs you need
set(OpenCV_LIBS opencv_core opencv_highgui)

# linking
target_link_libraries(LomoFilter ${OpenCV_LIBS} "--static")
cmake_minimum_required(VERSION 3.17)
project(SkinFilter)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXE_LINKER_FLAGS -static)

# Where to find CMake modules and OpenCV
set(OpenCV_DIR "D:\\OpenCV\\mingwbuild")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SkinFilter main.cpp)

# add libs you need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_objdetect)

# linking
target_link_libraries(SkinFilter ${OpenCV_LIBS} "-static")
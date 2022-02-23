cmake_minimum_required(VERSION 3.19)
project(yolov5)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#set(OpenCV_DIR "/usr/local/Cellar/opencv/4.3.0_5/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

set(ncnn_DIR "/home/ubuntu/opt/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(WARNING "ncnn NOT FOUND!  Please set ncnn_DIR environment variable")
else()
    message("NCNN FOUND")
endif()


include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(
        yolov5
        main.cpp
        yolov5.cpp
        yolov5.h)


target_link_libraries(
        yolov5
        ncnn
        ${OpenCV_LIBS}
)
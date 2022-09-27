cmake_minimum_required(VERSION 3.10)

project(video_cuda LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package (OpenMP REQUIRED)

SET(CUDA_SEPARABLE_COMPILATION ON)

# add_executable(guidedfilter_cpu ./src/guidedfilter.cpp)
# target_include_directories(guidedfilter_cpu PUBLIC ./src ${OpenMP_CXX_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(guidedfilter_cpu PUBLIC ${OpenCV_LIBRARIES} ${OpenMP_CXX_LIBRARIES} opencv_videoio)
# target_compile_options(guidedfilter_cpu PUBLIC ${OpenMP_CXX_FLAGS})

cuda_add_executable(guidedfilter_gpu src/main_gpu.cu src/guidedfilter.cu OPTIONS --expt-extended-lambda -O2)
target_include_directories(guidedfilter_gpu PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(guidedfilter_gpu ${OpenCV_LIBRARIES} opencv_videoio)
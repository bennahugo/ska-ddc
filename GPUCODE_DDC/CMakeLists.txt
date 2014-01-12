project(gpu_ddc)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
#declare executable

FIND_PACKAGE(CUDA)
if(CUDA_FOUND)
    SET(CUDA_NVCC_FLAGS "-O3 -Xcompiler -fopenmp -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30")
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()

cuda_add_executable(gpu_ddc ddc.cu)
install(TARGETS gpu_ddc RUNTIME DESTINATION bin)
#link external libraries
SET(CMAKE_REQUIRED_FLAGS "-Wl,--no-undefined")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
target_link_libraries(gpu_ddc gomp)

set(CMAKE_BUILD_TYPE Release)

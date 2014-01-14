project(output_correlator)

add_executable(output_correlator timer.cpp main.cpp)

find_package(rt)
target_link_libraries(output_correlator rt)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
target_link_libraries(output_correlator gomp)

install(TARGETS output_correlator RUNTIME DESTINATION bin)

set( CMAKE_CXX_FLAGS "-Wall -m64 -O3 -fopenmp" )
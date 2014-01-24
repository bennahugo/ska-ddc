project(inv_pfb)

add_executable(inv_pfb inv_pfb.c)

find_package(rt)
target_link_libraries(inv_pfb rt)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
target_link_libraries(inv_pfb gomp)

install(TARGETS inv_pfb RUNTIME DESTINATION bin)

set( CMAKE_CXX_FLAGS "-Wall -m64 -O3 -fopenmp" )

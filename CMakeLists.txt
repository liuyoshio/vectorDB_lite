cmake_minimum_required(VERSION 3.18)
project(vector_db LANGUAGES CXX)

option(VDB_ENABLE_CUDA "Enable CUDA acceleration" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable SIMD (AVX2/SSE4) for distance computations
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

find_package(OpenMP)

add_library(vdb
  src/dataset.cpp
  src/distance.cpp
  src/index.cpp
  src/ivf.cpp
  src/hnsw.cpp
  src/utils.cpp
  src/io.cpp
)

target_include_directories(vdb PUBLIC include)

if(OpenMP_CXX_FOUND)
  target_link_libraries(vdb PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(vdb PUBLIC VDB_USE_OPENMP)
endif()

if(VDB_ENABLE_CUDA)
  enable_language(CUDA)
  target_compile_definitions(vdb PUBLIC VDB_USE_CUDA)
  target_sources(vdb PRIVATE src/gpu.cu)
  set_target_properties(vdb PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
  target_sources(vdb PRIVATE src/gpu_stub.cpp)
endif()

add_executable(vdb_demo src/main.cpp)
add_executable(vdb_cli src/cli.cpp)

target_link_libraries(vdb_demo PRIVATE vdb)
target_link_libraries(vdb_cli PRIVATE vdb)

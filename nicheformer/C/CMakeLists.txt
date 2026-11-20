cmake_minimum_required(VERSION 3.18)
project(nicheformer_embed CXX)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(nicheformer_embed nicheformer.cpp)
target_link_libraries(nicheformer_embed "${TORCH_LIBRARIES}")
set_property(TARGET nicheformer_embed PROPERTY CXX_STANDARD 17)

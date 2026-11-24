cmake_minimum_required(VERSION 3.22.1)
project("nova_engine")

set(LLAMA_DIR ${CMAKE_SOURCE_DIR}/llama.cpp)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama.cpp)

add_library(nova_engine SHARED native-lib.cpp)

target_link_libraries(nova_engine
    llama
    android
    log
)

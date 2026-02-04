cmake_minimum_required(VERSION 3.16)

project(second_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(second_lab main.cpp)

if (MSVC)
    target_compile_options(second_lab PRIVATE
            /arch:AVX2
            /W4
            $<$<CONFIG:Release>:/O2>
    )
else()
    target_compile_options(second_lab PRIVATE
            -mavx2
            -mfma
            -Wall
            -Wextra
            -Wpedantic
            $<$<CONFIG:Release>:-O3>
    )
endif()


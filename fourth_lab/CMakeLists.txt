cmake_minimum_required(VERSION 3.16)

project(fourth_lab LANGUAGES CXX)

# ===== C++ =====
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ===== Исполняемый файл =====
add_executable(fourth_lab
        main.cpp
        mod_ops.cpp
        num_threads.cpp
        performance.cpp
        randomize.cpp
        test.cpp
        vector_mod.cpp
)

# ===== Заголовки =====
target_include_directories(fourth_lab PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# ===== OpenMP =====
find_package(OpenMP REQUIRED)
target_link_libraries(fourth_lab PRIVATE OpenMP::OpenMP_CXX)

# ===== Флаги компилятора =====
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
        CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

    target_compile_options(fourth_lab PRIVATE
            -O3
            -Wall
            -Wextra
            -Wpedantic
    )

elseif (MSVC)
    target_compile_options(fourth_lab PRIVATE
            /O2
            /W4
    )
endif()

# ===== Тип сборки =====
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

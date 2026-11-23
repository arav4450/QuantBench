cmake_minimum_required(VERSION 3.16)
project(QuantBench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OPT_FLAG "-O3 -march=native" CACHE STRING "Compiler optimization flags (e.g. -O0, -O2, -O3 -march=native)")
separate_arguments(OPT_FLAGS UNIX_COMMAND "${OPT_FLAG}")

add_executable(quantbench main.cpp)
target_compile_options(quantbench PRIVATE ${OPT_FLAGS})

add_executable(day2 day2.cpp)
target_compile_options(day2 PRIVATE ${OPT_FLAGS})

add_executable(day3 day3.cpp)
target_compile_options(day3 PRIVATE ${OPT_FLAGS})

add_executable(day4 day4.cpp)
target_compile_options(day4 PRIVATE ${OPT_FLAGS})

add_executable(day5 day5.cpp)
target_compile_options(day5 PRIVATE ${OPT_FLAGS})

add_executable(day6 day6.cpp)
target_compile_options(day6 PRIVATE ${OPT_FLAGS})

add_executable(day7 day7.cpp)
target_compile_options(day7 PRIVATE ${OPT_FLAGS})

add_executable(day8 day8.cpp)
target_compile_options(day8 PRIVATE ${OPT_FLAGS})

cmake_minimum_required(VERSION 3.16)
project(pufferfish VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
    add_compile_options(/W4 /WX)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /DNDEBUG)
    else()
        add_compile_options(/Od /Zi)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG -march=native)
    else()
        add_compile_options(-O0 -g)
    endif()
endif()

# =============================================================================
# Library: pufferfish_core
# =============================================================================
add_library(pufferfish_core STATIC
    src/position.cpp
    src/attack.cpp
    src/movegen.cpp
    src/perft.cpp
)
target_include_directories(pufferfish_core PUBLIC ${CMAKE_SOURCE_DIR}/src)

# =============================================================================
# Executable: perft
# =============================================================================
add_executable(perft apps/perft_main.cpp)
target_link_libraries(perft PRIVATE pufferfish_core)

# =============================================================================
# Tests (optional, enabled with -DBUILD_TESTS=ON)
# =============================================================================
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_runner tests/test_main.cpp)
    target_link_libraries(test_runner PRIVATE pufferfish_core)
    add_test(NAME unit_tests COMMAND test_runner)
endif()

message(STATUS "Pufferfish Chess Engine v${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

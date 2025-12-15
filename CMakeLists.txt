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
    src/zobrist.cpp
    src/transposition_table.cpp
    src/search.cpp
    src/timer.cpp
    src/uci.cpp
    src/nnue.cpp
)
target_include_directories(pufferfish_core PUBLIC ${CMAKE_SOURCE_DIR}/src)

# =============================================================================
# Executable: perft
# =============================================================================
add_executable(perft apps/perft_main.cpp)
target_link_libraries(perft PRIVATE pufferfish_core)

# =============================================================================
# Executable: zobrist_test
# =============================================================================
add_executable(zobrist_test apps/zobrist_test.cpp)
target_link_libraries(zobrist_test PRIVATE pufferfish_core)

# =============================================================================
# Executable: transposition_table_test
# =============================================================================
add_executable(transposition_table_test apps/transposition_table_test.cpp)
target_link_libraries(transposition_table_test PRIVATE pufferfish_core)

# =============================================================================
# Executable: search_test
# =============================================================================
add_executable(search_test apps/search_test.cpp)
target_link_libraries(search_test PRIVATE pufferfish_core)

# =============================================================================
# Executable: iterative_deepening_test
# =============================================================================
add_executable(iterative_deepening_test apps/iterative_deepening_test.cpp)
target_link_libraries(iterative_deepening_test PRIVATE pufferfish_core)

# =============================================================================
# Executable: uci_test
# =============================================================================
# Executable: uci_main (UCI Engine for CuteChess/Lichess)
# =============================================================================
add_executable(pufferfish apps/uci_main.cpp)
target_link_libraries(pufferfish PRIVATE pufferfish_core)

# =============================================================================
# Executable: uci_test (UCI Protocol Tests)
# =============================================================================
add_executable(uci_test apps/uci_test.cpp)
target_link_libraries(uci_test PRIVATE pufferfish_core)

# =============================================================================
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

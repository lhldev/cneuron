cmake_minimum_required(VERSION 3.10)
project(cneuron C CXX)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type: Debug or Release" FORCE)
endif()

message(STATUS "Building in ${CMAKE_BUILD_TYPE} mode")

enable_testing()

find_package(GTest REQUIRED)

file(GLOB_RECURSE SRC_FILES src/*.c EXCLUDE "src/main.c")
file(GLOB_RECURSE TEST_FILES test/*.cpp)

set(ASAN_FLAGS -fsanitize=address -fno-omit-frame-pointer -g)
set(WARNING_FLAGS -Wall -Wextra -Wpedantic)
set(RELEASE_FLAGS -O3 -DNDEBUG)

function(apply_target_flags target)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(${target} PRIVATE ${WARNING_FLAGS})

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${target} PRIVATE ${ASAN_FLAGS})
        target_link_options(${target} PRIVATE ${ASAN_FLAGS})
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${target} PRIVATE ${RELEASE_FLAGS})
        target_link_options(${target} PRIVATE -O3)
    endif()
endfunction()

# libcneuron library
add_library(libcneuron ${SRC_FILES})
apply_target_flags(libcneuron)
target_link_libraries(libcneuron PRIVATE m)

# Main executable
add_executable(${PROJECT_NAME} src/main.c)
apply_target_flags(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PRIVATE m libcneuron)

# Test executable
add_executable(tests ${TEST_FILES})
apply_target_flags(tests)
target_link_libraries(tests PRIVATE m libcneuron GTest::gtest_main)

add_test(NAME tests COMMAND tests)

message(STATUS "Checking and copying data and input.py to the build directory...")
file(COPY ${CMAKE_SOURCE_DIR}/data ${CMAKE_SOURCE_DIR}/input.py DESTINATION ${CMAKE_BINARY_DIR})
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output)

add_custom_target(run
    COMMAND ${CMAKE_BINARY_DIR}/${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME} tests
)

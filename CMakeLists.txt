cmake_minimum_required(VERSION 3.10)
project(cneuron C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type: Debug or Release" FORCE)
endif()

message(STATUS "Building in ${CMAKE_BUILD_TYPE} mode")

option(ENABLE_TESTING "Enable C++ and testing features" ON)

enable_testing()

if(NOT DEFINED BLA_VENDOR)
    set(BLA_VENDOR OpenBLAS)
    add_compile_definitions(USE_OPENBLAS)
endif()
message(STATUS "Using BLAS vendor: ${BLA_VENDOR}")
find_package(BLAS REQUIRED)

if(ENABLE_TESTING)
    find_package(GTest REQUIRED)
endif()

file(GLOB_RECURSE SRC_FILES src/*.c EXCLUDE "src/main.c")
file(GLOB_RECURSE TEST_FILES test/*.cpp)

set(SANITIZE_FLAGS -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
set(WARNING_FLAGS -Wall -Wextra -Wpedantic)
set(RELEASE_FLAGS -O3 -DNDEBUG -march=native)
set(PROFILE_FLAG -O3 -p -g)

set(INCLUDE_FOLDER ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/external ${CMAKE_SOURCE_DIR}/src)

option(USE_THREADING "Enable multithreading support for batch training" ON)
if(USE_THREADING)
    add_compile_definitions(USE_THREADING)
endif()

add_compile_options(${WARNING_FLAGS})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(${SANITIZE_FLAGS} -g)
    add_link_options(${SANITIZE_FLAGS})
elseif(CMAKE_BUILD_TYPE STREQUAL "Profile")
    add_compile_options(${PROFILE_FLAG})
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(${RELEASE_FLAGS})
endif()

# libcneuron library
add_library(cneuron ${SRC_FILES})
target_link_libraries(cneuron PRIVATE ${BLAS_LIBRARIES} m)
target_include_directories(cneuron PRIVATE ${INCLUDE_FOLDER})

# Main executable
add_executable(cneuron_cli src/main.c)
target_link_libraries(cneuron_cli PRIVATE cneuron)
target_include_directories(cneuron_cli PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/external)

# Test executable
if(ENABLE_TESTING)
    add_executable(tests ${TEST_FILES})
    target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/test)
    target_link_libraries(tests PRIVATE cneuron GTest::gtest_main)
    add_test(NAME tests COMMAND tests)
    target_include_directories(tests PRIVATE ${INCLUDE_FOLDER})
endif()


message(STATUS "Copying data and input.py to the build directory...")
file(COPY ${CMAKE_SOURCE_DIR}/data ${CMAKE_SOURCE_DIR}/input.py DESTINATION ${CMAKE_BINARY_DIR})
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output)

add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E echo "Using BLAS vendor: ${BLA_VENDOR}"
    COMMAND ${CMAKE_COMMAND} -E echo "Building in: ${CMAKE_BUILD_TYPE} mode"
    COMMAND ${CMAKE_BINARY_DIR}/cneuron_cli
    DEPENDS cneuron_cli $<$<BOOL:${ENABLE_TESTING}>:tests>
)

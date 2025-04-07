cmake_minimum_required(VERSION 3.10)
project(cneuron C CXX)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

find_package(GTest REQUIRED)

file(GLOB_RECURSE SRC_FILES src/*.c EXCLUDE "src/main.c")
file(GLOB_RECURSE TEST_FILES test/*.cpp)

set(ASAN_FLAGS -fsanitize=address -fno-omit-frame-pointer -g)
set(WARNING_FLAGS -Wall -Wextra -Wpedantic)

add_library(libcneuron ${SRC_FILES})
target_include_directories(libcneuron PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(libcneuron PRIVATE ${WARNING_FLAGS} ${ASAN_FLAGS})
target_link_options(libcneuron PRIVATE ${ASAN_FLAGS})
target_link_libraries(libcneuron PRIVATE m)

add_executable(${PROJECT_NAME} src/main.c)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(${PROJECT_NAME} PRIVATE ${WARNING_FLAGS} ${ASAN_FLAGS})
target_link_options(${PROJECT_NAME} PRIVATE ${ASAN_FLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE m libcneuron)

add_executable(tests ${TEST_FILES})
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(tests PRIVATE ${WARNING_FLAGS} ${ASAN_FLAGS})
target_link_options(tests PRIVATE ${ASAN_FLAGS})
target_link_libraries(tests PRIVATE m libcneuron GTest::gtest_main)

add_test(NAME tests COMMAND tests)

message(STATUS "Checking and copying data and input.py to the build directory...")
file(COPY ${CMAKE_SOURCE_DIR}/data ${CMAKE_SOURCE_DIR}/input.py DESTINATION ${CMAKE_BINARY_DIR})
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output)

add_custom_target(run
    COMMAND ${CMAKE_BINARY_DIR}/${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME} tests
)

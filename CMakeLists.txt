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
endif()
message(STATUS "Using BLAS vendor: ${BLA_VENDOR}")
find_package(BLAS REQUIRED)

if(ENABLE_TESTING)
    find_package(GTest REQUIRED)
endif()


file(GLOB_RECURSE SRC_FILES src/*.c EXCLUDE "src/main.c")
file(GLOB_RECURSE TEST_FILES test/*.cpp)

set(ASAN_FLAGS -fsanitize=address -fno-omit-frame-pointer)
set(WARNING_FLAGS -Wall -Wextra -Wpedantic)
set(RELEASE_FLAGS -O3 -DNDEBUG -march=native)
set(PROFILE_FLAG -O3 -p -g)

option(USE_THREADING "Enable multithreading support for batch training" ON)
function(apply_target target)
    if(USE_THREADING)
        target_compile_definitions(${target} PRIVATE USE_THREADING)
    endif()

    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/external)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_options(${target} PRIVATE ${WARNING_FLAGS})
    target_link_libraries(libcneuron PRIVATE ${BLAS_LIBRARIES} m)

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${target} PRIVATE ${ASAN_FLAGS} -g)
        target_link_options(${target} PRIVATE ${ASAN_FLAGS})
    elseif(CMAKE_BUILD_TYPE STREQUAL "Profile")
        target_compile_options(${target} PRIVATE ${PROFILE_FLAG})
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${target} PRIVATE ${RELEASE_FLAGS})
    endif()
endfunction()

# libcneuron library
add_library(libcneuron ${SRC_FILES})
apply_target(libcneuron)

# Main executable
add_executable(${PROJECT_NAME} src/main.c)
apply_target(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PRIVATE libcneuron)

# Test executable
if(ENABLE_TESTING)
    add_executable(tests ${TEST_FILES})
    apply_target(tests)
    target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/test)
    target_link_libraries(tests PRIVATE libcneuron GTest::gtest_main)
    add_test(NAME tests COMMAND tests)
endif()


message(STATUS "Copying data and input.py to the build directory...")
file(COPY ${CMAKE_SOURCE_DIR}/data ${CMAKE_SOURCE_DIR}/input.py DESTINATION ${CMAKE_BINARY_DIR})
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output)

add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E echo "Using BLAS vendor: ${BLA_VENDOR}"
    COMMAND ${CMAKE_COMMAND} -E echo "Building in: ${CMAKE_BUILD_TYPE} mode"
    COMMAND ${CMAKE_BINARY_DIR}/${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME} tests
    $<$<BOOL:${ENABLE_TESTING}>:tests>
)

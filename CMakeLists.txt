cmake_minimum_required(VERSION 3.10)
project(neuralnetwork C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE SRC_FILES src/*.c)

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME} PRIVATE m)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)

message(STATUS "Checking and copying data and input.py to the build directory...")
file(COPY ${CMAKE_SOURCE_DIR}/data ${CMAKE_SOURCE_DIR}/input.py DESTINATION ${CMAKE_BINARY_DIR})
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output)

add_custom_target(run
    COMMAND ${CMAKE_BINARY_DIR}/${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
)

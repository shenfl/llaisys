cmake_minimum_required(VERSION 3.16)
project(llaisys LANGUAGES C CXX)

# -------------------------
# 全局配置
# -------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # 等价 -fPIC

add_compile_options(
        -Wall
        -Wextra
        -Werror
        -Wno-unknown-pragmas
        -Wno-unused-parameter
)

# include 目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# -------------------------
# CPU 子模块
# -------------------------
add_subdirectory(cmake/cpu)

# -------------------------
# llaisys-utils
# -------------------------
file(GLOB LLAISYS_UTILS_SRCS
        src/utils/*.cpp
)
add_library(llaisys-utils STATIC
        ${LLAISYS_UTILS_SRCS}
)

# -------------------------
# llaisys-device
# -------------------------
file(GLOB LLAISYS_DEVICE_SRCS
        src/device/*.cpp
)

add_library(llaisys-device STATIC
        ${LLAISYS_DEVICE_SRCS}
)

target_link_libraries(llaisys-device
        PUBLIC
        llaisys-utils
        llaisys-device-cpu
)

# -------------------------
# llaisys-core
# -------------------------
file(GLOB LLAISYS_CORE_SRCS
        src/core/*/*.cpp
)

add_library(llaisys-core STATIC
        ${LLAISYS_CORE_SRCS}
)

target_link_libraries(llaisys-core
        PUBLIC
        llaisys-utils
        llaisys-device
)

# -------------------------
# llaisys-tensor
# -------------------------
file(GLOB LLAISYS_TENSOR_SRCS
        src/tensor/*.cpp
)

add_library(llaisys-tensor STATIC
        ${LLAISYS_TENSOR_SRCS}
)

target_link_libraries(llaisys-tensor
        PUBLIC
        llaisys-core
)

# -------------------------
# llaisys-ops
# -------------------------
file(GLOB LLAISYS_OPS_SRCS
        src/ops/*/*.cpp
)

add_library(llaisys-ops STATIC
        ${LLAISYS_OPS_SRCS}
)

target_link_libraries(llaisys-ops
        PUBLIC
        llaisys-ops-cpu
)

# -------------------------
# llaisys (shared)
# -------------------------
file(GLOB LLAISYS_SHARED_SRCS
        src/llaisys/*.cc
)

add_library(llaisys SHARED
        ${LLAISYS_SHARED_SRCS}
)

target_link_libraries(llaisys
        PUBLIC
        llaisys-utils
        llaisys-device
        llaisys-core
        llaisys-tensor
        llaisys-ops
)

# -------------------------
# 安装 & Python 拷贝
# -------------------------
set_target_properties(llaisys PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_custom_command(
        TARGET llaisys POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/python/llaisys/libllaisys
)

if (WIN32)
    add_custom_command(
            TARGET llaisys POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:llaisys>
            ${CMAKE_CURRENT_SOURCE_DIR}/python/llaisys/libllaisys
    )
else()
    add_custom_command(
            TARGET llaisys POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:llaisys>
            ${CMAKE_CURRENT_SOURCE_DIR}/python/llaisys/libllaisys
    )
endif()

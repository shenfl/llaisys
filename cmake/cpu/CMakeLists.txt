# -------------------------
# llaisys-device-cpu
# -------------------------
file(GLOB LLAISYS_DEVICE_CPU_SRCS
        ${CMAKE_SOURCE_DIR}/src/device/cpu/*.cpp
)

add_library(llaisys-device-cpu STATIC
        ${LLAISYS_DEVICE_CPU_SRCS}
)

# -------------------------
# llaisys-ops-cpu
# -------------------------
file(GLOB LLAISYS_OPS_CPU_SRCS
        ${CMAKE_SOURCE_DIR}/src/ops/*/cpu/*.cpp
)

add_library(llaisys-ops-cpu STATIC
        ${LLAISYS_OPS_CPU_SRCS}
)

target_link_libraries(llaisys-ops-cpu
        PUBLIC
        llaisys-tensor
)
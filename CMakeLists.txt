cmake_minimum_required(VERSION 3.20)

# Set the CUDA architecture before enabling the CUDA language. Otherwise CMake may
# initialize CMAKE_CUDA_ARCHITECTURES from an old compiler default such as sm_52.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures to build for, e.g. 70;80;89;90")
endif()

project(tensorcore_learning LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "[-a-zA-Z]+$" "" arch_number "${arch}")
    if(arch_number MATCHES "^[0-9]+$" AND arch_number LESS 70)
        message(FATAL_ERROR "WMMA requires CUDA architecture sm_70 or newer, got ${arch}. Pass -DCMAKE_CUDA_ARCHITECTURES=70/80/89/90 for your GPU.")
    endif()
endforeach()

option(TENSORCORE_ENABLE_PTXAS_INFO "Print ptxas register/shared-memory resource usage" ON)

function(tensorcore_cuda_options target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${TENSORCORE_ENABLE_PTXAS_INFO}>>:--ptxas-options=-v>
    )
endfunction()

function(add_gemm_kernel_stage stage_name kernel_src launch_symbol)
    add_executable(${stage_name}_test
        apps/test_gemm.cu
        ${kernel_src}
    )
    target_include_directories(${stage_name}_test PRIVATE include)
    target_compile_definitions(${stage_name}_test PRIVATE
        TC_KERNEL_NAME="${stage_name}"
        TC_KERNEL_LAUNCH=${launch_symbol}
    )
    tensorcore_cuda_options(${stage_name}_test)
    set_target_properties(${stage_name}_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    add_executable(${stage_name}_bench
        apps/bench_gemm.cu
        ${kernel_src}
    )
    target_include_directories(${stage_name}_bench PRIVATE include)
    target_compile_definitions(${stage_name}_bench PRIVATE
        TC_KERNEL_NAME="${stage_name}"
        TC_KERNEL_LAUNCH=${launch_symbol}
    )
    target_link_libraries(${stage_name}_bench PRIVATE CUDA::cublas)
    tensorcore_cuda_options(${stage_name}_bench)
    set_target_properties(${stage_name}_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endfunction()

add_gemm_kernel_stage(00_wmma_1warp kernels/00_wmma_1warp.cu launch_wmma_1warp)
add_gemm_kernel_stage(01_wmma_4warp_independent kernels/01_wmma_4warp_independent.cu launch_wmma_4warp_independent)
add_gemm_kernel_stage(01_wmma_block_tiled kernels/01_wmma_block_tiled.cu launch_wmma_block_tiled)

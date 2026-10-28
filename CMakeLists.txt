cmake_minimum_required(VERSION 3.26)
project(libstrix LANGUAGES CXX HIP)

set(STRIX_GPU_ARCH "gfx1151" CACHE STRING "Target GPU architecture")
set(CMAKE_HIP_ARCHITECTURES ${STRIX_GPU_ARCH})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_HIP_STANDARD 20)

find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)

file(GLOB_RECURSE STRIX_KERNELS CONFIGURE_DEPENDS csrc/*.hip)
add_library(_C SHARED csrc/torch_ops.cpp ${STRIX_KERNELS})
set_source_files_properties(csrc/torch_ops.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(_C PRIVATE csrc)
target_compile_definitions(_C PRIVATE __HIP_PLATFORM_AMD__ USE_ROCM)
target_compile_options(_C PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffp-contract=fast -munsafe-fp-atomics>)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES})
set_target_properties(_C PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/python/strix)

cmake_minimum_required(VERSION 3.16)
project(add_csv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_OPENMP "Enable OpenMP" ON)

add_executable(add_csv
  Est.cpp
  main.cpp
)

# OpenMP on macOS
set(HAVE_OPENMP FALSE)
if(ENABLE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(add_csv PRIVATE OpenMP::OpenMP_CXX)
    set(HAVE_OPENMP TRUE)
  endif()

  if(APPLE AND NOT HAVE_OPENMP AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    find_path(OMP_INCLUDE_DIR omp.h
      HINTS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
    find_library(OMP_LIBRARY omp
      HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
    if(OMP_INCLUDE_DIR AND OMP_LIBRARY)
      message(STATUS "Using Homebrew libomp fallback: ${OMP_LIBRARY}")
      target_include_directories(add_csv PRIVATE ${OMP_INCLUDE_DIR})
      target_link_libraries(add_csv PRIVATE ${OMP_LIBRARY})
      target_compile_options(add_csv PRIVATE -Xpreprocessor -fopenmp)
      get_filename_component(OMP_LIB_DIR ${OMP_LIBRARY} DIRECTORY)
      if(OMP_LIB_DIR)
        target_link_options(add_csv PRIVATE "-Wl,-rpath,${OMP_LIB_DIR}")
      endif()
      set(HAVE_OPENMP TRUE)
    else()
      message(WARNING "OpenMP not found. Try: brew install libomp  或使用 brew install llvm")
    endif()
  endif()
endif()

# Warnings & Opts
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
  target_compile_options(add_csv PRIVATE -Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(add_csv PRIVATE -Wall -Wextra -Wpedantic -march=native)
endif()

target_compile_options(add_csv PRIVATE
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Release>:-O3>
)
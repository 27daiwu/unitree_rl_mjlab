cmake_minimum_required(VERSION 3.12)
project(g1_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 强制优先使用 /usr/local，避免链接到系统 libddsc.so.0debian
set(CMAKE_BUILD_RPATH "/usr/local/lib")
set(CMAKE_INSTALL_RPATH "/usr/local/lib")
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)
find_package(ZLIB REQUIRED)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-aarch64-1.22.0")
endif()

# 显式指定 Unitree SDK2 / CycloneDDS 的 /usr/local 版本
set(UNITREE_SDK2_LIB unitree_sdk2)
set(DDSC_LIB         "/usr/local/lib/libddsc.so.0")
set(DDSCXX_LIB       "/usr/local/lib/libddscxx.so.0")

include_directories(
  /usr/include/eigen3
  /usr/local/include
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
  include
  ${ONNXRUNTIME_DIR}/include
  ${PROJECT_SOURCE_DIR}/../../include/
  ${PROJECT_SOURCE_DIR}/../../thirdparty/cnpy
)

file(GLOB CNPY_SRC ${PROJECT_SOURCE_DIR}/../../thirdparty/cnpy/*.cpp)

add_library(cnpy_lib STATIC ${CNPY_SRC})

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})

target_link_libraries(${PROJECT_NAME}_lib
  ${UNITREE_SDK2_LIB}
  ${DDSC_LIB}
  ${DDSCXX_LIB}
  rt
  pthread
  Boost::program_options
  yaml-cpp
  fmt
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.22.0
  cnpy_lib
  ZLIB::ZLIB
)

add_executable(g1_ctrl main.cpp)

target_link_libraries(g1_ctrl
  ${PROJECT_NAME}_lib
  ${UNITREE_SDK2_LIB}
  ${DDSC_LIB}
  ${DDSCXX_LIB}
  rt
  pthread
  Boost::program_options
  yaml-cpp
  fmt
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.22.0
  cnpy_lib
  ZLIB::ZLIB
)
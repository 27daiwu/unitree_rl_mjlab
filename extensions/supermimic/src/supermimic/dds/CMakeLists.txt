cmake_minimum_required(VERSION 3.16)
project(supermimic_dds_transport LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(supermimic_dds SHARED dds_transport.cpp)
target_include_directories(supermimic_dds PRIVATE
  /usr/local/include
  /usr/local/include/ddsc
  /usr/local/include/ddscxx
)
target_link_libraries(supermimic_dds PRIVATE
  /usr/local/lib/libunitree_sdk2.a
  /usr/local/lib/libddsc.so
  /usr/local/lib/libddscxx.so
  rt
  pthread
)
set_target_properties(supermimic_dds PROPERTIES
  BUILD_RPATH "/usr/local/lib"
  INSTALL_RPATH "/usr/local/lib"
  OUTPUT_NAME "supermimic_dds"
)

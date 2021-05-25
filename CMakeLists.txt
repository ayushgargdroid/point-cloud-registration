cmake_minimum_required(VERSION 2.8 )
project(icp)

set (CMAKE_CXX_STANDARD 14)

find_package(PCL REQUIRED QUIET COMPONENTS io common)
find_package(Eigen3 REQUIRED)

include_directories(include)
add_executable(icp_svd src/icp_svd.cpp src/icp_helper.cpp include/icp_svd.h)

if(PCL_FOUND)

  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
  target_link_libraries(icp_svd ${PCL_LIBRARIES} ${Eigen3_LIBRARIES})

endif()
cmake_minimum_required(VERSION 3.16)
project(icp)

set (CMAKE_CXX_STANDARD 17)

find_package(PCL REQUIRED QUIET COMPONENTS io common kdtree search features)
find_package(Eigen3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(include)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(icp_svd src/icp_svd.cpp src/icp_helper.cpp)
target_precompile_headers(icp_svd PUBLIC include/project_includes.h)
target_link_libraries(icp_svd ${PCL_LIBRARIES} ${Eigen3_LIBRARIES})

add_executable(icp_point_to_point src/icp_point_to_point.cpp src/icp_helper.cpp)
target_precompile_headers(icp_point_to_point PUBLIC include/project_includes.h)
target_link_libraries(icp_point_to_point ${PCL_LIBRARIES} ${Eigen3_LIBRARIES})
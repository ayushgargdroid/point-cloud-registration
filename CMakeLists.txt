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

add_executable(icp_variants src/icp_variants.cpp src/icp_helper.cpp)
target_precompile_headers(icp_variants PUBLIC include/project_includes.h)
target_link_libraries(icp_variants ${PCL_LIBRARIES} ${Eigen3_LIBRARIES})
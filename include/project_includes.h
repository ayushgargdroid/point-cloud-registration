#pragma once

#include <iostream>
#include <iterator>
#include <random>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/SVD>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/io/pcd_io.h>

#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>

#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/feature.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/flann.h>
#include <pcl/search/kdtree.h>

#include <math.h>
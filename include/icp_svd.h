#pragma once

#include <iostream>
#include <iterator>
#include <random>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/SVD>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <math.h>

#define DEG_TO_RAD(x) (x / 180.0 * M_PI)

Eigen::MatrixXf convertPcltoEigen(pcl::PointCloud<pcl::PointXYZ>::ConstPtr inputCloud);
pcl::PointCloud<pcl::PointXYZ>::Ptr convertEigenToPcl(Eigen::MatrixXf inputPoints);
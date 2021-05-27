#pragma once

#include <project_includes.h>

#define DEG_TO_RAD(x) (x / 180.0 * M_PI)

struct CorrespondenceData {
    Eigen::MatrixXf srcPts, dstPts;
};

Eigen::MatrixXf convertPcltoEigen(pcl::PointCloud<pcl::PointXYZ>::ConstPtr inputCloud);
pcl::PointCloud<pcl::PointXYZ>::Ptr convertEigenToPcl(Eigen::MatrixXf inputPoints);
pcl::PointCloud<pcl::PointNormal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::ConstPtr inputCloud, float searchRadius);
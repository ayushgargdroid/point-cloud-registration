#pragma once

#define DEG_TO_RAD(x) (x / 180.0 * M_PI)

Eigen::MatrixXf convertPcltoEigen(pcl::PointCloud<pcl::PointXYZ>::ConstPtr inputCloud);
pcl::PointCloud<pcl::PointXYZ>::Ptr convertEigenToPcl(Eigen::MatrixXf inputPoints);
pcl::PointCloud<pcl::PointNormal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::ConstPtr inputCloud, float searchRadius);
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> getNearestCorrespondences(pcl::PointCloud<pcl::PointXYZ>::ConstPtr srcCloud, pcl::PointCloud<pcl::PointXYZ>::ConstPtr dstCloud, float maxDist);
#pragma once

#define DEG_TO_RAD(x) (x / 180.0 * M_PI)

template <typename T> int sgn(T val);
Eigen::Affine3f vectorToHomography(Eigen::VectorXf x);

Eigen::MatrixXf convertPcltoEigen(pcl::PointCloud<pcl::PointXYZ>::ConstPtr inputCloud);
Eigen::MatrixXf convertPcltoEigen(pcl::PointCloud<pcl::PointXYZ>::ConstPtr inputCloud, std::vector<int> indices);
pcl::PointCloud<pcl::PointXYZ>::Ptr convertEigenToPcl(Eigen::MatrixXf inputPoints);

pcl::PointCloud<pcl::PointNormal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::ConstPtr inputCloud, float searchRadius);

std::tuple<std::vector<int>, std::vector<int>> getNearestCorrespondences(pcl::PointCloud<pcl::PointXYZ>::ConstPtr srcCloud, pcl::PointCloud<pcl::PointXYZ>::ConstPtr dstCloud, float maxDist);

float getFitnessScore(Eigen::MatrixXf srcPoints, Eigen::MatrixXf dstPoints);
bool verifyRightHandRule(Eigen::Matrix3f eigenVec);
Eigen::Matrix3f fixEigenVecToUnitAxes(Eigen::Matrix3f eigenVec);
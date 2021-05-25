#include "../include/icp_svd.h"

Eigen::MatrixXf convertPcltoEigen(pcl::PointCloud<pcl::PointXYZ>::ConstPtr inputCloud) {
    Eigen::MatrixXf srcPoints(inputCloud->size(), 4);
    int i = 0;
    for(auto point : inputCloud->points) {
        srcPoints(i, 0) = point.x;
        srcPoints(i, 1) = point.y;
        srcPoints(i, 2) = point.z;
        srcPoints(i, 3) = 1.0;
        i++;
    }
    return srcPoints;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr convertEigenToPcl(Eigen::MatrixXf inputPoints) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->width = inputPoints.rows();
    cloud->height = 1;
    cloud->points.reserve(inputPoints.rows());

    for(int i = 0; i < inputPoints.rows(); i++) {
        cloud->points.emplace_back(inputPoints(i, 0), inputPoints(i, 1), inputPoints(i, 2));
    }

    return cloud;
}
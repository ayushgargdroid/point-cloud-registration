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

pcl::PointCloud<pcl::PointNormal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::ConstPtr inputCloud, float searchRadius) {
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr treeSearch(new pcl::KdTreeFLANN<pcl::PointXYZ>());
    treeSearch->setInputCloud(inputCloud);

    pcl::PointCloud<pcl::PointNormal>::Ptr normalCloud(new pcl::PointCloud<pcl::PointNormal>());
    normalCloud->header = inputCloud->header;
    normalCloud->width = inputCloud->width;
    normalCloud->height = inputCloud->height;
    normalCloud->points.reserve(inputCloud->size());
    
    for(int i = 0; i < inputCloud->size(); i++) {
        std::vector<int> indices;
        std::vector<float> dists;
        Eigen::Matrix3f covMat;
        Eigen::Vector4f mean;
        Eigen::Vector3f::Scalar eigenValues;
        Eigen::Vector3f eigenVector;
        pcl::PointXYZ pt = inputCloud->points[i];
        pcl::PointNormal normalPt;
        normalPt.x = pt.x;
        normalPt.y = pt.y;
        normalPt.z = pt.z;
        treeSearch->radiusSearch(i, searchRadius, indices, dists);
        pcl::computeMeanAndCovarianceMatrix(*inputCloud, indices, covMat, mean);
        pcl::eigen33(covMat, eigenValues, eigenVector);
        normalPt.normal_x = eigenVector[0];
        normalPt.normal_y = eigenVector[1];
        normalPt.normal_z = eigenVector[2];
        normalCloud->points.emplace_back(normalPt);
    }
    return normalCloud;
}
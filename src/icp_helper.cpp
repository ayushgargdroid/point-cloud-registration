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

Eigen::MatrixXf convertPcltoEigen(pcl::PointCloud<pcl::PointXYZ>::ConstPtr inputCloud, std::vector<int> indices) {
    Eigen::MatrixXf srcPoints(inputCloud->size(), 4);
    int i = 0;
    for(int indx : indices) {
        srcPoints(i, 0) = inputCloud->points[indx].x;
        srcPoints(i, 1) = inputCloud->points[indx].y;
        srcPoints(i, 2) = inputCloud->points[indx].z;
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

std::tuple<std::vector<int>, std::vector<int>> getNearestCorrespondences(pcl::PointCloud<pcl::PointXYZ>::ConstPtr srcCloud, pcl::PointCloud<pcl::PointXYZ>::ConstPtr dstCloud, float maxDist) {

    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr treeSearch(new pcl::KdTreeFLANN<pcl::PointXYZ>());
    treeSearch->setInputCloud(dstCloud);

    std::vector<int> srcIndx, dstIndx;
    std::vector<float> dist;
    std::vector<int> indx;

    for(int pti = 0; pti < srcCloud->size(); pti++) {
        indx.clear();
        dist.clear();

        pcl::PointXYZ srcPt = srcCloud->points[pti];
        treeSearch->nearestKSearch(srcPt, 1, indx, dist);

        if(indx.size() > 0 && dist[0] < maxDist) {
            srcIndx.push_back(pti);
            dstIndx.push_back(indx[0]);
        }

    }

    return {srcIndx, dstIndx};

}

float getFitnessScore(Eigen::MatrixXf srcPoints, Eigen::MatrixXf dstPoints) {
    return (dstPoints - srcPoints).norm();
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

bool verifyRightHandRule(Eigen::Matrix3f eigenVec) {
    Eigen::Vector3f correctCol0 = eigenVec.col(1).cross(eigenVec.col(2)).transpose();
    Eigen::Vector3f correctCol1 = eigenVec.col(2).cross(eigenVec.col(0)).transpose();
    Eigen::Vector3f col0 = eigenVec.col(0).transpose();
    Eigen::Vector3f col1 = eigenVec.col(1).transpose();

    bool isCorrect = true;
    for(int i = 0; i < 3; i++) {
        if(!((sgn<float>(correctCol0(i)) == sgn<float>(col0(i))) && (sgn<float>(correctCol1(i)) == sgn<float>(col1(i)))))
            return false;
    }
    return true;
}
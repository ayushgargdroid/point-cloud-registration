#include "../include/icp_svd.h"

int main() {
    const int numOfPoints = 100;
    const int angle = 180;
    const Eigen::Vector3f translation(10.0, 5.0, 0.0);
    const float a = 8;
    const float b = 2;
    const bool addNoise = true;
    const bool debug = false;

    Eigen::Affine3f gtTrans;
    gtTrans.setIdentity();
    gtTrans.translate(translation);
    gtTrans.rotate(Eigen::AngleAxisf(DEG_TO_RAD(angle), Eigen::Vector3f::UnitZ()));
    std::cout << "GT Transformation Matrix" << std::endl;
    std::cout << gtTrans.matrix() << std::endl;

    std::vector<float> noise(numOfPoints, 0.0);
    if(addNoise) { 
        // Define random generator with Gaussian distribution
        const double mean = 0.0;
        const double stddev = 0.5;
        std::default_random_engine generator;
        std::normal_distribution<float> dist(mean, stddev);
        
        // Add Gaussian noise
        for (auto& x : noise) {
            x = dist(generator);
        }
    }

    // Generate Points
    Eigen::MatrixXf srcPoints(numOfPoints, 4), dstPoints(numOfPoints, 4);
    for(int i = 0; i < numOfPoints; i++) {
        srcPoints(i, 0) = a * cos(M_PI*2.0*(i+noise[i])/(float)numOfPoints);
        srcPoints(i, 1) = b * sin(M_PI*2.0*(i+noise[i])/(float)numOfPoints);
        srcPoints(i, 3) = 1.0;

        dstPoints(i, 0) = a * cos(M_PI*2.0*i/(float)numOfPoints);
        dstPoints(i, 1) = b * sin(M_PI*2.0*i/(float)numOfPoints);
        dstPoints(i, 3) = 1.0;
    }
    dstPoints = (gtTrans.matrix() * dstPoints.transpose()).transpose();
    if(debug) {
        std::cout << "src points" << std::endl;
        std::cout << srcPoints << std::endl;
        std::cout << "dst points" << std::endl;
        std::cout << dstPoints << std::endl;
    }

    // Compute COM for point sets
    Eigen::Vector4f srcCom(srcPoints.col(0).mean(), srcPoints.col(1).mean(), srcPoints.col(2).mean(), srcPoints.col(3).mean());
    Eigen::Vector4f dstCom(dstPoints.col(0).mean(), dstPoints.col(1).mean(), dstPoints.col(2).mean(), dstPoints.col(3).mean());
    if(debug) {
        std::cout << "src points COM" << std::endl;
        std::cout << srcCom << std::endl;
        std::cout << "dst points COM" << std::endl;
        std::cout << dstCom << std::endl;
    }

    // Compute cross correlation matrix
    dstPoints.rowwise() -= dstCom.transpose();
    srcPoints.rowwise() -= srcCom.transpose();
    if(debug) {
        std::cout << "src points - com" <<std::endl;
        std::cout << srcPoints << std::endl;
        std::cout << "dst points - com" <<std::endl;
        std::cout << dstPoints << std::endl;
    }
    Eigen::MatrixXf H = dstPoints.leftCols(3).transpose() * srcPoints.leftCols(3);
    if(debug) {
        std::cout << "Cross correlation matrix" <<std::endl;
        std::cout << H << std::endl;
    }

    // Compute SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (debug) {
        std::cout << "Singular values" << std::endl;
        std::cout << svd.singularValues() << std::endl;
        std::cout << "U" << std::endl;
        std::cout << svd.matrixU().size() << std::endl;
        std::cout << "V" << std::endl;
        std::cout << svd.matrixV().size() << std::endl;
    }
    Eigen::Matrix3f computedRot = svd.matrixV() * svd.matrixU().transpose();
    Eigen::Matrix4f computedTrans;
    computedTrans.setZero();
    computedTrans.block<3,3>(0,0) = computedRot;
    Eigen::Vector4f computedT = dstCom - computedTrans * srcCom;
    computedTrans.block<4,1>(0,3) = computedT;
    std::cout << "Computed Transform" << std::endl;
    std::cout << computedTrans << std::endl;

    return 0;
}
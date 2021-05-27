#include "../include/icp_svd.h"

int main(int argc, char* argv[]) {
    const int angle = atoi(argv[1]);
    const Eigen::Vector3f translation(10.0, 5.0, 0.0);
    const bool debug = false;
    const std::string pcdFile = "../bun_zipper_res3.pcd";
    const int maxIters = 200;
    const float maxError = 0.01;
    const float maxCorrespondenceDist = 0.01;

    pcl::PointCloud<pcl::PointXYZ>::Ptr srcCloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr origSrcCloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr dstCloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr alignedCloud(new pcl::PointCloud<pcl::PointXYZ>());
    int numOfPoints;

    pcl::io::loadPCDFile(pcdFile, *srcCloud);
    pcl::copyPointCloud(*srcCloud, *origSrcCloud);
    numOfPoints = srcCloud->size();

    Eigen::Affine3f gtTrans;
    gtTrans.setIdentity();
    gtTrans.translate(translation);
    gtTrans.rotate(Eigen::AngleAxisf(DEG_TO_RAD(angle), Eigen::Vector3f::UnitZ()));
    std::cout << "GT Transformation Matrix" << std::endl;
    std::cout << gtTrans.matrix() << std::endl;

    // Generate transformed PC
    pcl::transformPointCloud(*srcCloud, *dstCloud, gtTrans.matrix());

    // Compute COM
    pcl::PointXYZ srcComPt, dstComPt;
    Eigen::Vector4f mean;
    Eigen::Matrix3f covMat;
    pcl::computeCentroid(*srcCloud, srcComPt);
    pcl::computeCentroid(*dstCloud, dstComPt);
    if(debug) {
        std::cout << "src points COM" << std::endl;
        std::cout << srcComPt << std::endl;
        std::cout << "dst points COM" << std::endl;
        std::cout << dstComPt << std::endl;
    }

    // Align COM for initial guess
    Eigen::Matrix4f initTrans, finalTrans;
    initTrans.setIdentity();
    finalTrans.setIdentity();
    initTrans(0,3) = dstComPt.x - srcComPt.x;
    initTrans(1,3) = dstComPt.y - srcComPt.y;
    initTrans(2,3) = dstComPt.z - srcComPt.z;
    pcl::transformPointCloud(*srcCloud, *srcCloud, initTrans);
    finalTrans = initTrans * finalTrans;
    pcl::io::savePCDFile("../results/bunny_dst.pcd", *dstCloud);

    if(debug) {
        std::cout << "Initial Guess" << std::endl;
        std::cout << initTrans << std::endl;
    }

    for(int iter = 0; iter < maxIters; iter ++) {
        // Find correspondences
        auto[srcIndx, dstIndx] = getNearestCorrespondences(srcCloud, dstCloud, maxCorrespondenceDist);
        if(srcIndx.size() < 10) break;

        // Convert to Eigen
        Eigen::MatrixXf srcPoints = convertPcltoEigen(srcCloud, srcIndx);
        Eigen::MatrixXf dstPoints = convertPcltoEigen(dstCloud, dstIndx);

        // Compute COM for point sets
        Eigen::Vector4f srcCom(srcPoints.col(0).mean(), srcPoints.col(1).mean(), srcPoints.col(2).mean(), srcPoints.col(3).mean());
        Eigen::Vector4f dstCom(dstPoints.col(0).mean(), dstPoints.col(1).mean(), dstPoints.col(2).mean(), dstPoints.col(3).mean());
        // Compute cross correlation matrix
        srcPoints.rowwise() -= srcCom.transpose();
        dstPoints.rowwise() -= dstCom.transpose();
        Eigen::MatrixXf H = srcPoints.leftCols(3).transpose() * dstPoints.leftCols(3);
        // Compute SVD        
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f computedRot = svd.matrixV() * svd.matrixU().transpose();
        Eigen::Matrix4f computedTrans;
        computedTrans.setZero();
        computedTrans.block<3,3>(0,0) = computedRot;
        Eigen::Vector4f computedT = dstCom - computedTrans * srcCom;
        computedTrans.block<4,1>(0,3) = computedT;
        finalTrans = computedTrans * finalTrans;

        // Compute Fitness score
        srcPoints = (computedTrans.matrix() * srcPoints.transpose()).transpose();
        float fitnessScore = getFitnessScore(srcPoints, dstPoints);
        if(debug) {
            std::cout << "Iter " << iter << " fitness score " << fitnessScore << std::endl;
            std::cout << "Trans\n" << computedTrans << std::endl;
        }

        // Transform src point cloud to the new computedTrans
        pcl::transformPointCloud(*srcCloud, *srcCloud, computedTrans);

        if(fitnessScore < maxError) {
            std::cout << "Success in " << iter+1 << " iterations" << std::endl;
            std::cout << finalTrans << std::endl;
            break;
        }
        else if(iter == maxIters-1) {
            std::cout << "Failiure in " << iter+1 << " iterations" << std::endl;
            std::cout << finalTrans << std::endl;
        }
    }    
    pcl::io::savePCDFile("../results/transformed_bunny.pcd", *srcCloud);

    return 0;
}
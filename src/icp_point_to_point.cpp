#include "../include/icp_svd.h"

int main(int argc, char* argv[]) {
    const int angle = atoi(argv[1]);
    const bool estimateInitOri = atoi(argv[2]);
    const Eigen::Vector3f translation(0.0, 0.0, 0.0);
    const bool debug = true;
    const std::string pcdFile = "../bun_zipper_res3.pcd";
    const int maxIters = 100;
    const float maxError = 0.01;
    const float maxCorrespondenceDist = 0.001;

    pcl::PointCloud<pcl::PointXYZ>::Ptr srcCloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr origSrcCloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr dstCloud(new pcl::PointCloud<pcl::PointXYZ>());
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
    pcl::io::savePCDFile("../results/bunny_dst.pcd", *dstCloud);

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

    if(estimateInitOri) {
        std::cout << "Using PCA for initial Orientation estimate" << std::endl;
        // Compute Eigen vectors for the point clouds for initial orientation using PCA
        // Refer https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fwww.cse.wustl.edu%2F~taoju%2Fcse554%2Flectures%2Flect08_Alignment.pptx&wdOrigin=BROWSELINK
        Eigen::Matrix3f srcEigenVec, dstEigenVec, dstEigenVecFix1, dstEigenVecFix2;
        Eigen::Vector3f srcEigenVal, dstEigenVal;
        Eigen::Matrix3f srcCov, dstCov;

        Eigen::MatrixXf srcPoints = convertPcltoEigen(srcCloud);
        Eigen::MatrixXf dstPoints = convertPcltoEigen(dstCloud);
        Eigen::Vector4f srcCom(srcPoints.col(0).mean(), srcPoints.col(1).mean(), srcPoints.col(2).mean(), srcPoints.col(3).mean());
        Eigen::Vector4f dstCom(dstPoints.col(0).mean(), dstPoints.col(1).mean(), dstPoints.col(2).mean(), dstPoints.col(3).mean());
        srcPoints.rowwise() -= srcCom.transpose();
        dstPoints.rowwise() -= dstCom.transpose();
        Eigen::MatrixXf srcH = srcPoints.leftCols(3).transpose() * srcPoints.leftCols(3);
        Eigen::MatrixXf dstH = dstPoints.leftCols(3).transpose() * dstPoints.leftCols(3);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> srcSolver(srcH);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> dstSolver(dstH);

        srcEigenVal = srcSolver.eigenvalues();
        srcEigenVec = fixEigenVecToUnitAxes(srcSolver.eigenvectors());
        dstEigenVal = dstSolver.eigenvalues();
        dstEigenVec = dstSolver.eigenvectors();

        if(dstEigenVec(2,2) < 0) dstEigenVec *= -1;
        std::cout << "DST vec normal - values " << dstEigenVal.transpose() << " det " << dstEigenVec.determinant() << std::endl;
        std::cout << dstEigenVec << std::endl;

        // Compute Orientation estimate by aligning eigen values
        // PCA eigenvectors are not orthonormal so there is ambiguity, hence taking 2 guesses and finding which one has highest correspondences with dst cloud
        // This check should strictly follow right hand rule cross product
        if(verifyRightHandRule(dstEigenVec)) {
            std::cout << "Did not need" << std::endl;
            initTrans.setIdentity();
            initTrans.block<3,3>(0,0) = dstEigenVec * srcEigenVec.transpose();
            pcl::transformPointCloud(*srcCloud, *srcCloud, initTrans);
        }
        else {
            pcl::PointCloud<pcl::PointXYZ>::Ptr alignedCloud1(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::PointCloud<pcl::PointXYZ>::Ptr alignedCloud2(new pcl::PointCloud<pcl::PointXYZ>());
            Eigen::Matrix4f oriTrans1, oriTrans2;
            dstEigenVecFix1 = dstEigenVec;
            dstEigenVecFix2 = dstEigenVec;
            dstEigenVecFix1.col(1) *= -1;
            dstEigenVecFix2.col(0) *= -1;

            if(debug) {
                std::cout << "fix1 " << dstEigenVecFix1.determinant() << " " << dstEigenVecFix1.trace() << " " << (dstEigenVecFix1 * srcEigenVec.transpose()).trace() << std::endl;
                std::cout << dstEigenVecFix1 << std::endl;
                std::cout << "fix2 " << dstEigenVecFix2.determinant() << " " << dstEigenVecFix2.trace() << " " << (dstEigenVecFix2 * srcEigenVec.transpose()).trace() << std::endl;
                std::cout << dstEigenVecFix2 << std::endl;
            }

            oriTrans1.setIdentity();
            oriTrans1.block<3,3>(0,0) = dstEigenVecFix1 * srcEigenVec.transpose();
            pcl::transformPointCloud(*srcCloud, *alignedCloud1, oriTrans1);
            auto[srcIndx1, dstIndx1] = getNearestCorrespondences(alignedCloud1, dstCloud, maxCorrespondenceDist);
            // pcl::io::savePCDFile("../results/init_bunny1.pcd", *alignedCloud1);

            oriTrans2.setIdentity();
            oriTrans2.block<3,3>(0,0) = dstEigenVecFix2 * srcEigenVec.transpose();
            pcl::transformPointCloud(*srcCloud, *alignedCloud2, oriTrans2);
            auto[srcIndx2, dstIndx2] = getNearestCorrespondences(alignedCloud2, dstCloud, maxCorrespondenceDist);
            // pcl::io::savePCDFile("../results/init_bunny2.pcd", *alignedCloud2);

            initTrans.setIdentity();
            Eigen::Affine3f tempAffine1, tempAffine2;
            tempAffine1.matrix() << oriTrans1;
            tempAffine2.matrix() << oriTrans2;
            float x1, x2, y1, y2, z1, z2;
            Eigen::Vector3f t1 = tempAffine1.rotation().eulerAngles(2, 1 ,0) / M_PI * 180.0;
            Eigen::Vector3f t2 = tempAffine2.rotation().eulerAngles(2, 1 ,0) / M_PI * 180.0;
            if(srcIndx1.size() > srcIndx2.size()) {
                dstEigenVec = dstEigenVecFix1;
                initTrans = oriTrans1;
                srcCloud = alignedCloud1;
            }
            else {
                dstEigenVec = dstEigenVecFix2;
                initTrans = oriTrans2;
                srcCloud = alignedCloud2;
            }
        }

        finalTrans = initTrans * finalTrans;

        if(debug) {
            std::cout << "Src Eigen Values " << srcEigenVal.transpose() << " det " << srcEigenVec.determinant() << std::endl;
            std::cout << srcEigenVec << std::endl;
            std::cout << "Dst Eigen Values " << dstEigenVal.transpose() << " det " << dstEigenVec.determinant() << std::endl;
            std::cout << dstEigenVec << std::endl; 
        }
    }

    if(debug) {
        std::cout << "Initial Guess" << std::endl;
        std::cout << finalTrans << std::endl;
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
            // std::cout << "Trans\n" << computedTrans << std::endl;
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
#include "../include/icp_svd.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct PointToPointOptimizationFunctor : LmOptimizationFunctor<float> {

    Eigen::MatrixXf m_srcPoints, m_dstPoints;
    int m_cloudSize;
    
    PointToPointOptimizationFunctor(int cloudSize, Eigen::MatrixXf srcPoints, Eigen::MatrixXf dstPoints) : 
        LmOptimizationFunctor<float>(6, cloudSize), m_cloudSize(cloudSize), m_srcPoints(srcPoints), m_dstPoints(dstPoints) {}

    int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const {
        Eigen::Affine3f transform = vectorToHomography(x);

        Eigen::MatrixXf warpedSrcCloud = (transform.matrix() * m_srcPoints.transpose()).transpose();
        Eigen::MatrixXf errorCloud = m_dstPoints - warpedSrcCloud;
        fvec = errorCloud.rowwise().norm();

        std::cout << "err " << fvec.sum() << std::endl;

        return 0;
    }

};

class CeresPointToPointOptimizationFunctor {
public:
    CeresPointToPointOptimizationFunctor(Eigen::Vector4f x, Eigen::Vector4f y){
        for(int i = 0; i < 3; i++) {
            m_srcPoint[i] = x[i];
            m_dstPoint[i] = y[i];
        }
    }

    template<typename T>
    bool operator()(const T* quat, const T* transl, T* residual) const {
        T rot[9];
        ceres::QuaternionToRotation(quat, &rot[0]);

        T transformedPt[3];
        for(int i = 0; i < 3; i++) {
            transformedPt[i] = rot[(i*3)] * m_srcPoint[0] + rot[(i*3)+1] * m_srcPoint[1] + rot[(i*3)+2] * m_srcPoint[2] + transl[i];
        }

        residual[0] = m_dstPoint[0] - transformedPt[0];
        residual[1] = m_dstPoint[1] - transformedPt[1];
        residual[2] = m_dstPoint[2] - transformedPt[2];
        return true;
    }

private:
    double m_srcPoint[3];
    double m_dstPoint[3];
};

int main(int argc, char* argv[]) {
    const int angle = atoi(argv[1]);
    const bool estimateInitOri = atoi(argv[2]);
    const Eigen::Vector3f translation(0.0, 0.0, 0.0);
    const bool debug = true;
    const std::string pcdFile = "../bun_zipper_res3.pcd";
    const int maxIters = 20;
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
    Eigen::MatrixXf srcPoints = convertPcltoEigen(srcCloud);
    Eigen::MatrixXf dstPoints = convertPcltoEigen(dstCloud);
    Eigen::Vector4f srcCom(srcPoints.col(0).mean(), srcPoints.col(1).mean(), srcPoints.col(2).mean(), srcPoints.col(3).mean());
    Eigen::Vector4f dstCom(dstPoints.col(0).mean(), dstPoints.col(1).mean(), dstPoints.col(2).mean(), dstPoints.col(3).mean());
    if(debug) {
        std::cout << "src points COM" << std::endl;
        std::cout << srcCom << std::endl;
        std::cout << "dst points COM" << std::endl;
        std::cout << dstCom << std::endl;
    }

    // Align COM for initial guess
    Eigen::Matrix4f initTrans, finalTrans;
    initTrans.setIdentity();
    finalTrans.setIdentity();
    initTrans.block<4,1>(0,3) = dstCom - srcCom;
    pcl::transformPointCloud(*srcCloud, *srcCloud, initTrans);
    finalTrans = initTrans * finalTrans;

    if(debug) {
        srcPoints = convertPcltoEigen(srcCloud);
        dstPoints = convertPcltoEigen(dstCloud);
        std::cout << "Init guess \n" << finalTrans << std::endl;
        std::cout << "Initial fitness score " << getFitnessScore(srcPoints, dstPoints) << std::endl;
    }

    for(int iter = 0; iter < maxIters; iter ++) {
        // Find correspondences
        auto[srcIndx, dstIndx] = getNearestCorrespondences(srcCloud, dstCloud, maxCorrespondenceDist);
        if(srcIndx.size() < 10) {
            std::cout << "Less than 10 correspondences found" << std::endl;
            return 1;
        }

        // Convert to Eigen
        Eigen::MatrixXf srcPoints = convertPcltoEigen(srcCloud, srcIndx);
        Eigen::MatrixXf dstPoints = convertPcltoEigen(dstCloud, dstIndx);

        Eigen::VectorXf x(6);
        x.setZero();
        // PointToPointOptimizationFunctor functor(static_cast<int>(srcPoints.rows()), srcPoints, dstPoints);
        // Eigen::NumericalDiff<PointToPointOptimizationFunctor> numDiff(functor);
        // Eigen::LevenbergMarquardt<Eigen::NumericalDiff<PointToPointOptimizationFunctor>, float> lm(numDiff);
        // lm.parameters.maxfev = 2000;
        // lm.parameters.factor = 20;
        // lm.parameters.xtol = 1e-10;
        // int ret = lm.minimize(x);
        // std::cout << "iter count: " << lm.iter << std::endl;
        // std::cout << "return status: " << ret << std::endl;

        ///////////////////////////////
        ceres::Problem problem;
        double optTransl[3] = {0.0, 0.0, 0.0};
        double optQuat[4]  = {1.0, 0.0, 0.0, 0.0};
        for(int pti = 0; pti < srcPoints.rows(); pti++) {
            // CeresPointToPointOptimizationFunctor<Eigen::Vector3f>* costFunctor = new CeresPointToPointOptimizationFunctor<Eigen::Vector3f>(srcPoints.row(pti), dstPoints.row(pti));
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CeresPointToPointOptimizationFunctor, 3, 4, 3>(new CeresPointToPointOptimizationFunctor(srcPoints.row(pti), dstPoints.row(pti))), NULL, &optQuat[0], &optTransl[0]);
        }
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n\n";
        Eigen::Quaternionf quatRot(optQuat[0], optQuat[1], optQuat[2], optQuat[3]);
        Eigen::Affine3f computedTrans;
        computedTrans.setIdentity();
        computedTrans.rotate(quatRot);
        computedTrans.matrix().col(3) << optTransl[0], optTransl[1], optTransl[2];
        std::cout << "Ceres\n" << computedTrans.matrix() << std::endl;
        std::cout << "Ceres \n" << optTransl[0] << " " << optTransl[1] << " " << optTransl[2] << std::endl;
        // std::cout << optEuler[0] << " " << optEuler[1] << " " << optEuler[2] << std::endl;
        // x << optEuler[0], optEuler[1], optEuler[2], optTransl[0], optTransl[1], optTransl[2];
        ///////////////////////////////

        // Eigen::Affine3f computedTrans = vectorToHomography(x);
        finalTrans = computedTrans * finalTrans;
        // Compute Fitness score
        srcPoints = (computedTrans.matrix() * srcPoints.transpose()).transpose();
        float fitnessScore = getFitnessScore(srcPoints, dstPoints);
        if(debug) {
            std::cout << "Iter " << iter << " fitness score " << fitnessScore << std::endl;
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
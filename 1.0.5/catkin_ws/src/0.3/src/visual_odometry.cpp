#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"

namespace myslam
{

VisualOdometry::VisualOdometry() :
    state_ ( INITIALIZING ), lref_ ( nullptr ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ), matcher_flann_( new cv::flann::LshIndexParams(5,10,2) )
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    map_point_erase_ratio_ = Config::get<double> ("map_point_erase_ratio");
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame ( Frame::Ptr frame ) 
{
    F_num++;//04171448 可视化
    switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = OK;
        curr_ = ref_ = lref_ = frame;
        map_->insertKeyFrame ( frame );
        // extract features from first frame and add them into map
        extractKeyPoints();
        computeDescriptors();
        setRef3DPoints();
        break;
    }
    case OK:
    {
        curr_ = frame;
        extractKeyPoints();
        computeDescriptors();
        //从第十帧开始用优化匹配方法
        if(F_num<4)
        {
            featureMatchingByFlann();
        }
        else
        {
            featureMatchingOptimization();
        }
        poseEstimationPnP();
        //coutFunction();
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r*T_r_w 
            lref_ = ref_;
            ref_ = curr_;
            setRef3DPoints();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                addKeyFrame();//question:cann't add key frame?????????
                //coutFunction();
                //coutFunction();
            }
            else//test
            {
                cout<<"\033[01;31mNO Key-Frame!\033[0m"<<endl;
            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            return false;
        }
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;
        break;
    }
    }

    return true;
}

void VisualOdometry::extractKeyPoints()
{
    boost::timer timer;
    orb_->detect ( curr_->color_, keypoints_curr_ );
    cout<<"extract keypoints cost time: "<<timer.elapsed()<<endl;
}

void VisualOdometry::computeDescriptors()
{
    boost::timer timer;
    orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
    cout<<"descriptor computation cost time: "<<timer.elapsed()<<endl;
}

void VisualOdometry::featureMatchingByFlann()
{
    cout<<"*******-----原始的FLANN匹配算法-----********"<<endl;
    boost::timer timer;
    vector<cv::DMatch> matches;
    matcher_flann_.match( descriptors_ref_, descriptors_curr_, matches );
    // select the best matches
    float min_dis = std::min_element (
                        matches.begin(), matches.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {
        return m1.distance < m2.distance;
    } )->distance;

    feature_matches_.clear();
    for ( cv::DMatch& m : matches )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            feature_matches_.push_back(m);
        }
    }
    for(cv::DMatch m : feature_matches_)
    {
        match_3dpcam_.push_back(pts_3d_ref_[m.queryIdx]);
        match_2dkp_index_.push_back(m.trainIdx);
    }
    //绘制两两图像的匹配结果 04162100 可视化
    // if(F_num>1)//从第二帧图像开始绘制(1-2,2-3,3-4等等)
    // {
    //     cout<<"--------可视化---------"<<endl;
    //     cv::Mat img_feature_show;
    //     //cv::imshow("上一帧",ref_->color_);
    //     //cv::imshow("当前帧",curr_->color_);//能输出前后两帧图像
    //     cv::drawMatches(ref_->color_, keypointors_ref_show, curr_->color_, keypoints_curr_, feature_matches_, img_feature_show);
    //     cv::imshow("相邻两帧匹配结果：",img_feature_show);
    //     cv::waitKey(1000);
    // }
    cout<<"good matches: "<<feature_matches_.size()<<endl;
    cout<<"match cost time: "<<timer.elapsed()<<endl;
}

void VisualOdometry::featureMatchingOptimization()
{
    cout<<"---在0.3版本上——改动1.0.5----"<<endl;
    boost::timer timer;//计时器
    //已知上一帧地图点相对相机的坐标，再根据匀速运动模型，即已知上一帧到当前帧的变换位姿，那么可以估计地图点相对于当前帧相机坐标系下的坐标
    Eigen::Matrix3d Rcr_ = T_c_r_estimated_.rotationMatrix();//上一帧到当前帧旋转矩阵
    Eigen::Vector3d tcr_ = T_c_r_estimated_.translation();//上一帧到当前帧平移向量 

    cv::Mat Rcr(3,3,CV_32F);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            Rcr.at<float>(i,j)=Rcr_(i,j);//将Eigen的矩阵转化成cv矩阵 //SE3不支持直接转cv::Mat

    cv::Mat tcr(3,1,CV_32F);
    for(int i=0;i<3;i++)
        tcr.at<float>(i,0)=tcr_(i,0);//Rcw tcw已经有了

    //保存已经匹配的MapPoint //
    match_3dpcam_.clear();
    //保存已经匹配的2d关键点的索引
    match_2dkp_index_.clear();//for循环开始前清空

    vector<cv::DMatch> good_matches;//04172015 可视化

    cout<<"上一帧地图点数量："<<ref_->N_mp<<endl;//六七百的样子
    for(int i=0; i<ref_->N_mp; i++)//遍历上一帧所有地图点
    {
        cv::Point3f Pr = pts_3d_ref_[i];//获得该点相对于上一帧的相机坐标
       
        cv::Mat x3Dr(3,1,CV_32F);
        x3Dr.at<float>(0,0)=Pr.x;
        x3Dr.at<float>(1,0)=Pr.y;
        x3Dr.at<float>(2,0)=Pr.z;//将Point3f的行向量转化成列向量，现在已经得到上一帧那一个点的世界坐标了

        cv::Mat x3Dc = Rcr*x3Dr+tcr;//计算该点在当前帧相机坐标系下的坐标
        //三个的数据类型要一样32F or 64F，否则会有段错误   ////通过调试！！！
        
        const float xc = x3Dc.at<float>(0);
        const float yc = x3Dc.at<float>(1);
        const float invzc = 1.0/x3Dc.at<float>(2);//inverse zc

        if(invzc<0)//点在相机后方了（一种判断点是否在帧视野内的方法）
            continue;

        float u = ref_->camera_->fx_ * xc * invzc + ref_->camera_->cx_;
        float v = ref_->camera_->fy_ * yc * invzc + ref_->camera_->cy_;//计算投影点的像素坐标
           
        if(u<0 || u>ref_->color_.cols)
            continue;
        if(v<0 || v>ref_->color_.rows)//用像素坐标判断投影点是否在图像内的
            continue;

        //已经求出投影点的像素坐标，接下来要确定搜索区域（包括半径取多少合适， 确定区域内的点）

        float r = 30;//定义一个搜索半径，这个由多次调试实验得出

        vector<size_t> vIndices;//存放当前帧圆区域内所有候选点索引值  圆区域改成方区域，可行！！
        
        for(int j=0; j<keypoints_curr_.size(); j++)//遍历当前帧所有关键点的坐标，把在正方形区域内的关键点的索引值存放在vIndices容器中
        {
            if(keypoints_curr_.at(j).pt.x > u-r && keypoints_curr_.at(j).pt.x < u+r)
            {
                if(keypoints_curr_.at(j).pt.y > v-r && keypoints_curr_.at(j).pt.y < v+r)
                {
                    vIndices.push_back(j);//vIndices的数据类型可以换成int吗
                }
            }
        }

        if(vIndices.empty())
            continue;

        const cv::Mat dMp = descriptors_ref_.row(i);//当前投影点的描述子

        //尝试第一种方法：一对多暴力匹配
        cv::BFMatcher matcher_BF_ (cv::NORM_HAMMING);
        vector<cv::DMatch> matches;
        cv::Mat descriptors_range_all_;//搜索区域内所有点的描述子，一行一个描述子
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t i2 = *vit;//重点 *vit 指的是vIndives索引值对应的容器里的值，即在搜索区域内关键点的索引值
            descriptors_range_all_.push_back(descriptors_curr_.row(i2));//一定要注意数据类型是否一致！！！！
        }
        //matcher_flann_.match(dMp, descriptors_range_all_, matches);
        matcher_BF_.match(dMp, descriptors_range_all_, matches);
        //第一种方法结束
        //cout<<"区域内匹配对数："<<matches.size()<<endl;

        float min_dis = 20;//随便取的一个数
        float v_T = 30.0;//阈值
        if(matches[0].distance < max<float> ( min_dis * match_ratio_, v_T ))//没法获取最小distance,人为取一个
        {
            match_3dpcam_.push_back(Pr);//匹配合适的上一帧特征点的相机坐标
            match_2dkp_index_.push_back(vIndices[matches[0].trainIdx]);// int类数据容器,当前帧匹配成功关键点的索引值，
            //有了索引值就能找到相应点的像素坐标，也会位姿估计需要用到的数据

//失败的可能原因：descriptors_range_all_与它真正的keypoint没有对应起来，即match_2dkp_index_容器里放的并不是想要的索引值！！！！！
//经过简单验证：正是此原因 //04181520
//改正：match_2dkp_index_.push_back(vIndices[matches[0].trainIdx]);

            //good_matches.push_back(matches[0]); 可视化
        }
    }
    
    //  //绘制两两图像的匹配结果 04162100 可视化     //此段代码无用！！！！！
    // if(F_num>1)//从第二帧图像开始绘制(1-2,2-3,3-4等等)
    // {
    //     cout<<"------------可视化---------"<<endl;
    //     cv::Mat img_feature_show;
    //     //cv::imshow("上一帧",ref_->color_);
    //     //cv::imshow("当前帧",curr_->color_);//能输出前后两帧图像
    //     cv::drawMatches(ref_->color_, keypointors_ref_show, curr_->color_, keypoints_curr_, good_matches, img_feature_show);
    //     cv::imshow("相邻两帧匹配结果：",img_feature_show);
    //     cv::waitKey(1000);
    // }
    cout<<"good matches: "<<match_3dpcam_.size() <<endl;
    cout<<"match cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::setRef3DPoints()
{
    // select the features with depth measurements 
    pts_3d_ref_.clear();
    keypointors_ref_show.clear();
    descriptors_ref_ = Mat();
    for ( size_t i=0; i<keypoints_curr_.size(); i++ )
    {
        double d = ref_->findDepth(keypoints_curr_[i]);               
        if ( d > 0)
        {
            Vector3d p_cam = ref_->camera_->pixel2camera(
                Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), d
            );
            pts_3d_ref_.push_back( cv::Point3f( p_cam(0,0), p_cam(1,0), p_cam(2,0) ));
            keypointors_ref_show.push_back(keypoints_curr_[i]);
            descriptors_ref_.push_back(descriptors_curr_.row(i));
        }
    }
    ref_->N_mp = pts_3d_ref_.size();
}

void VisualOdometry::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;
    
    // for ( cv::DMatch m:feature_matches_ )
    // {
    //     pts3d.push_back( pts_3d_ref_[m.queryIdx] );//上一帧地图点的相机坐标，含深度值 Point3f
    //     pts2d.push_back( keypoints_curr_[m.trainIdx].pt );//当前帧匹配上的特征点的像素坐标 Point2f
    // }

    for ( cv::Point3f pt:match_3dpcam_ )
    {
        pts3d.push_back(pt);//三维地图点，  Point3f的数据类型
    }
    for ( int index:match_2dkp_index_ )
    {
        pts2d.push_back ( keypoints_curr_[index].pt );//当前帧二维像素点，  Point2f
    }
    
    
    Mat K = ( cv::Mat_<double>(3,3)<<
        ref_->camera_->fx_, 0, ref_->camera_->cx_,
        0, ref_->camera_->fy_, ref_->camera_->cy_,
        0,0,1
    );
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
    num_inliers_ = inliers.rows;
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    
    
    // 此处旋转向量经罗德里格斯转换
    Mat R;
    cv::Rodrigues(rvec, R);
    Eigen::Matrix3d RE;
    RE << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
    // SO3构造函数参数为旋转矩阵
    T_c_r_estimated_ = SE3<double>(SO3<double>(RE),Vector3d(tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0)));

    
    //g2o optimization  
    // using bundle adjustment to optimize the pose 
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    //Block* solver_ptr = new Block( linearSolver );
    Block* solver_ptr = new Block( std::unique_ptr<Block::LinearSolverType>(linearSolver) );
    //g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(solver_ptr) );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );
    
    //添加顶点，一帧只有一个位姿，也就是只有一个顶点
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
        T_c_r_estimated_.rotationMatrix(), 
        T_c_r_estimated_.translation()
    ) );
    optimizer.addVertex ( pose );

    // edges
    // edges边有许多，每个特征点都对应一个重投影误差，也就有一个边。
    for ( int i=0; i<inliers.rows; i++ )
    {
        int index = inliers.at<int>(i,0);
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId(i);
        edge->setVertex(0, pose);
        edge->camera_ = curr_->camera_.get();
        edge->point_ = Vector3d( pts3d[index].x, pts3d[index].y, pts3d[index].z );
        edge->setMeasurement( Vector2d(pts2d[index].x, pts2d[index].y) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        optimizer.addEdge( edge );
    }
    
    //开始优化
    optimizer.initializeOptimization();
    //设置迭代次数
    optimizer.optimize(10);
    
    //这步就是将优化后的结果，赋值给T_c_r_estimated_
    T_c_r_estimated_ = SE3<double>(
        pose->estimate().rotation(),
        pose->estimate().translation()
    );
}

bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"\033[01;31mreject because inlier is too small: \033[0m"<<num_inliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    Sophus::Vector6d d = T_c_r_estimated_.log();
    if ( d.norm() > 12.0 )
    {
        cout<<"\033[01;31mreject because motion is too large: \033[0m"<<d.norm()<<endl;
        return false;
    }
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    Sophus::Vector6d d = T_c_r_estimated_.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    cout<<"\033[01;31mrot.norm :"<<rot.norm()<<"\033[0m"<<endl;
    cout<<"\033[01;31mtrans.norm :"<<trans.norm()<<"\033[0m"<<endl;//rot.norm() & trans.norm are too small,so have no key-frames been added
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometry::addKeyFrame()
{
    //cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;
    //cout<<"\033[01;31madding a key-frame\033[0m"<<endl;
    map_->insertKeyFrame ( curr_ );
}

void VisualOdometry::coutFunction()
{
    cout<<"\033[01;31m++++++++++I'm here!++++++++++++++++++\033[0m"<<endl;
}

}

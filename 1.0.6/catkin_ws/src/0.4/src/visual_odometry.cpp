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
    state_ ( INITIALIZING ), lref_ ( nullptr )/*04120916*/, ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ), matcher_flann_ ( new cv::flann::LshIndexParams ( 5,10,2 ) )
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    map_point_erase_ratio_ = Config::get<double> ( "map_point_erase_ratio" );
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame ( Frame::Ptr frame )
{
    F_num++;//帧的数量,用于判断使用哪个匹配函数
    switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = OK;
        curr_ = ref_ = lref_ = frame;
        // extract features from first frame and add them into map
        extractKeyPoints();
        computeDescriptors();
        addKeyFrame();      // the first frame is a key-frame
        break;
    }
    case OK:
    {
        curr_ = frame;
        //说一下这一句，新的帧来了，先将其位姿赋值为参考帧的位姿，
        //因为考虑到匹配失败的情况下，这一帧就定义为丢失了，所以位姿就用参考帧的了。
        //如果一切正常，求得了当前帧的位姿，就进行赋值覆盖掉就好了。
        curr_->T_c_w_ = ref_->T_c_w_;
        extractKeyPoints();
        computeDescriptors();
        //从第十帧开始用优化匹配方法
        if(F_num<10)
        {
            featureMatchingByFlann();
        }
        else
        {
            featureMatchingOptimization();
        }
        poseEstimationPnP();//采用两种匹配方法，用Flann解决初始化问题

        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_w_estimated_;
            lref_ = ref_;//04120904
            ref_ = curr_;//改错，加上了这一行，让参考帧变成上一次循环的当前帧

            optimizeMap();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                addKeyFrame();
            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            cout<<"丢失次数："<<num_lost_<<endl;
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
    cout<<"extract keypoints cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::computeDescriptors()
{
    boost::timer timer;
    orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
    cout<<"descriptor computation cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::featureMatchingByFlann()
{
    boost::timer timer;//计时器
    cout<<"**************----原始的FLANN匹配方法----***************"<<endl;
    vector<cv::DMatch> matches;//存放两个描述子比较的结果

    ///////////////////////////////////////////////001 遍历地图点，找到在视野中的候选点，将点和它的描述子放入容器中备用
    // select the candidates in map  候选点
    Mat desp_map;//描述子是一个行向量，建立一个保存描述子的矩阵，保存 匹配需要的地图点的 描述子

    vector<MapPoint::Ptr> candidate;//candidate的生命周期只限于这个函数，暂时存放在当前帧视野以内的mappoint

    cout<<"地图点数量:"<<map_->map_points_.size()<<endl;
    for ( auto& allpoints: map_->map_points_ )//遍历所有地图点，将符合条件的mappoint放入candidate，将描述子信息放入desp_map
    {
        MapPoint::Ptr& p = allpoints.second;
        // check if p in curr frame image 
        if ( curr_->isInFrame(p->pos_) )// 检测这个点（世界坐标系中）是否在当前帧的视野之内
        {
            // add to candidate 
            p->visible_times_++;//某个地图点在帧中的观测次数

            candidate.push_back( p );//把能够在当前帧观测到的地图点放入候选点容器中

            desp_map.push_back( p->descriptor_ );//地图点的描述子放入矩阵中，//Mat也可以利用push_back来插入一行，和vector类似
            //一行即代表一个点的描述子，有几个候选点就有几行描述子 n*128(256)维
        }
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    

    //采用新的匹配方法FlannBasedMatcher(最近邻近似匹配)，而不是暴力匹配BFMatcher；
    //这步匹配中，第一个参数由原来的参考帧，变成了上面定义的desp_map地图，进行匹配。
    //也就是当前帧直接和地图进行匹配
    matcher_flann_.match ( desp_map, descriptors_curr_, matches );//结果放入了matches

    // select the best matches
    //找出容器matches中最小的元素的指针或迭代器
    //distance变量返回两个描述子之间的距离
    //这里用到了lambda表达式，也就是这里面的第三个参数
    //可以隐藏返回值的类型，会根据return自动确定，这里又加上了
    float min_dis = std::min_element (
                        matches.begin(), matches.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {
        return m1.distance < m2.distance;//m1 m2 类型与 matches 一样
    } )->distance;//////////////////////计算出的最小距离是为了作为后面匹配成功与否的条件

    //保存已经匹配的MapPoint（MapPoint的指针类型）
    match_3dpts_.clear();
    //保存已经匹配的2d关键点的索引
    match_2dkp_index_.clear();

    vector<cv::DMatch> good_matches;//04162109  可视化
    for ( cv::DMatch& m : matches )
    {
        //如果描述子之间的距离小于一个值（30和min_dis*match_ratio_中较大的），则表示匹配成功
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            //会匹配成功很多，但只有一部分符合条件，将符合条件的这些特征点放入容器
	        //queryIdx表示参考帧的匹配成功的索引(查询点)；trainIdx表示当前帧的匹配成功的索引(被查询点)
            match_3dpts_.push_back( candidate[m.queryIdx] );
            match_2dkp_index_.push_back( m.trainIdx );
            good_matches.push_back(m);//04162109  可视化
        }
    }
    // //绘制两两图像的匹配结果 04162100 可视化
    // if(F_num>1)//从第二帧图像开始绘制(1-2,2-3,3-4等等)
    // {
    //     cout<<"------------可视化---------"<<endl;
    //     cv::Mat img_feature_show;
    //     // cv::imshow("测试",ref_->color_);
    //     // cv::imshow("测试+1",curr_->color_);//能输出前后两帧图像
    //     cv::drawMatches(ref_->color_, ref_->keypoints_ref_, curr_->color_, keypoints_curr_, good_matches, img_feature_show);
    //     //good_matches有问题！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    //     cv::imshow("相邻两帧匹配结果：",img_feature_show);
    //     cv::waitKey(0);
    // }

    cout<<"good matches: "<<match_3dpts_.size() <<endl;
    cout<<"match cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::featureMatchingOptimization()
{
    cout<<"**********---------0.4基础上改动1.0.6---------********"<<endl;
    boost::timer timer;//计时器
    //已知前两帧在世界坐标系下的位姿，可求出上上帧到上一帧的变换，再根据匀速运动模型，估计当前帧位姿
    SE3<double> T_r_lr = ref_->T_c_w_ * lref_->T_c_w_.inverse();//lref到ref的变换位姿
    SE3<double> T_c_w_temp = T_r_lr * ref_->T_c_w_;//当前帧估计的位姿(temporary)
    Eigen::Matrix3d Rcw_ = T_c_w_temp.rotationMatrix();//当前帧旋转矩阵
    Eigen::Vector3d tcw_ = T_c_w_temp.translation();//当前帧平移向量，都是相对于世界坐标系  
    
    cv::Mat Rcw(3,3,CV_32F);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            Rcw.at<float>(i,j)=Rcw_(i,j);//将Eigen的矩阵转化成cv矩阵 //SE3不支持直接转cv::Mat

    cv::Mat tcw(3,1,CV_32F);
    for(int i=0;i<3;i++)
        tcw.at<float>(i,0)=tcw_(i,0);//Rcw tcw已经有了

    //保存已经匹配的MapPoint //
    match_3dpts_.clear();
    //保存已经匹配的2d关键点的索引
    match_2dkp_index_.clear();//for循环开始前清空

    
    //------------------------------------------------------
    Mat desp_map;//描述子是一个行向量，建立一个保存描述子的矩阵，保存 匹配需要的地图点的 描述子
    vector<MapPoint::Ptr> candidate;//candidate的生命周期只限于这个函数，暂时存放在当前帧视野以内的mappoint
    cout<<"地图点数量:"<<map_->map_points_.size()<<endl;
    for ( auto& allpoints: map_->map_points_ )//遍历所有地图点，将符合条件的mappoint放入candidate，将描述子信息放入desp_map
    {
        MapPoint::Ptr& p = allpoints.second;
        // check if p in curr frame image 
        if ( curr_->isInFrame(p->pos_) )// 检测这个点（世界坐标系中）是否在当前帧的视野之内
        {
            // add to candidate 
            p->visible_times_++;//某个地图点在帧中的观测次数

            candidate.push_back( p );//把能够在当前帧观测到的地图点放入候选点容器中

            desp_map.push_back( p->descriptor_ );//地图点的描述子放入矩阵中，//Mat也可以利用push_back来插入一行，和vector类似
            //一行即代表一个点的描述子，有几个候选点就有几行描述子 n*128(256)维
        }
    }
    cout<<"上一帧地图点数量："<<candidate.size()<<endl;//
    //------------------------------------------------------------


    for(int i=0; i<candidate.size(); i++)//遍历上一帧所有地图点
    {
        Vector3d Pw = candidate[i]->pos_;//获得该点在世界坐标下的位姿

        cv::Mat x3Dw(3,1,CV_32F);
        for(int i=0;i<3;i++)
            x3Dw.at<float>(i,0)=Pw(i,0);//将Vector3d的列向量转化成cv::Mat列向量，现在已经得到上一帧那一个点的世界坐标了

        cv::Mat x3Dc = Rcw*x3Dw+tcw;//计算该点在当前帧相机坐标系下的坐标
        //三个的数据类型要一样32F or 64F，否则会有段错误   ////通过调试！！！
        
        const float xc = x3Dc.at<float>(0);
        const float yc = x3Dc.at<float>(1);
        const float invzc = 1.0/x3Dc.at<float>(2);//inverse zc

        float u = ref_->camera_->fx_ * xc * invzc + ref_->camera_->cx_;
        float v = ref_->camera_->fy_ * yc * invzc + ref_->camera_->cy_;//计算投影点的像素坐标

        //已经求出投影点的像素坐标，接下来要确定搜索区域（包括半径取多少合适， 确定区域内的点）

        float r = 20;//定义一个搜索半径，这个由多次调试实验得出

        vector<size_t> vIndices;//存放当前帧圆区域内所有候选点索引值  圆区域改成方区域，可行！！

        for(int j=0; j<keypoints_curr_.size(); j++)//遍历当前帧所有关键点的坐标，把在正方形区域内的关键点的索引值存放在vIndices容器中
        {
            if(keypoints_curr_.at(j).pt.x > u-r && keypoints_curr_.at(j).pt.x < u+r)
            {
                if(keypoints_curr_.at(j).pt.y > v-r && keypoints_curr_.at(j).pt.y < v+r)
                {
                    vIndices.push_back(j);
                }
            }
        }

        if(vIndices.empty())
            continue;

        const cv::Mat dMp = desp_map.row(i);//当前投影点的描述子

        //尝试第一种方法：一对多暴力匹配
        cv::BFMatcher matcher_BF_ (cv::NORM_HAMMING);
        vector<cv::DMatch> matches;
        cv::Mat descriptors_range_all_;//搜索区域内所有点的描述子，一行一个描述子
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t i2 = *vit;//重点 *vit 指的是vIndives索引值对应的容器里的值，即在搜索区域内关键点的索引值
            descriptors_range_all_.push_back(descriptors_curr_.row(i2));//一定要注意数据类型是否一致！！！！
        }

        matcher_BF_.match(dMp, descriptors_range_all_, matches);
        //第一种方法结束
        //区域内匹配对数：1对

        float min_dis = 20;//随便取的一个数
        float v_T = 30.0;
        if(matches[0].distance < max<float> ( min_dis * match_ratio_, v_T ))//没法获取最小distance,人为取一个
        {
            //用点的三维世界坐标构建地图点，放入match_3dpts_容器中！！！！！！！！！！！！   
            match_3dpts_.push_back(candidate[i]);//在当前帧视野内的地图点，相对于世界坐标系 MapPoint类数据容器,坐标是Vector3d类型
            match_2dkp_index_.push_back(vIndices[matches[0].trainIdx]);// int类数据容器,当前帧匹配成功关键点的索引值，
            //有了索引值就能找到相应点的像素坐标，也会位姿估计需要用到的数据

//失败的可能原因：descriptors_range_all_与它真正的keypoint没有对应起来，即match_2dkp_index_容器里放的并不是想要的索引值！！！！！
//经过简单验证：正是此原因 //04181520
//改正：match_2dkp_index_.push_back(vIndices[matches[0].trainIdx]);
        }
    }

    //绘制两两图像的匹配结果 04162100 可视化
    // vector<cv::DMatch> good_matches;//04162109  可视化
    // if(F_num>1)//从第二帧图像开始绘制(1-2,2-3,3-4等等)
    // {
    //     cout<<"------------可视化---------"<<endl;
    //     cv::Mat img_feature_show;
    //     // cv::imshow("测试",ref_->color_);
    //     // cv::imshow("测试+1",curr_->color_);//能输出前后两帧图像
    //     cv::drawMatches(ref_->color_, ref_->keypoints_ref_, curr_->color_, keypoints_curr_, good_matches, img_feature_show);
    //     cv::imshow("相邻两帧匹配结果：",img_feature_show);
    //     cv::waitKey(0);
    // }

    cout<<"good matches: "<<match_3dpts_.size() <<endl;
    cout<<"match cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;
    for ( int index:match_2dkp_index_ )
    {
        pts2d.push_back ( keypoints_curr_[index].pt );//当前帧二维像素点，  Point2f
    }
    for ( MapPoint::Ptr pt:match_3dpts_ )
    {
        pts3d.push_back( pt->getPositionCV() );//三维地图点，  Point3f的数据类型
    }

    Mat K = ( cv::Mat_<double> ( 3,3 ) <<
              ref_->camera_->fx_, 0, ref_->camera_->cx_,
              0, ref_->camera_->fy_, ref_->camera_->cy_,
              0,0,1
            );
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac ( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
    num_inliers_ = inliers.rows;
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    
     // 此处旋转向量经罗德里格斯转换
        Mat R;
        cv::Rodrigues(rvec, R);
        Eigen::Matrix3d RE;
        RE << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
        // SO3构造函数参数为旋转矩阵
        T_c_w_estimated_ = SE3<double>(SO3<double>(RE), Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)));
        

    // using bundle adjustment to optimize the pose
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> Block;
        Block::LinearSolverType *linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
        // Block* solver_ptr = new Block( linearSolver );
        Block *solver_ptr = new Block(std::unique_ptr<Block::LinearSolverType>(linearSolver));
        // g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(solver_ptr));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
        pose->setId(0);
        pose->setEstimate(g2o::SE3Quat(
            T_c_w_estimated_.rotationMatrix(), T_c_w_estimated_.translation()));
        optimizer.addVertex(pose);

    // edges
    for ( int i=0; i<inliers.rows; i++ )
    {
        int index = inliers.at<int> ( i,0 );
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId ( i );
        edge->setVertex ( 0, pose );
        edge->camera_ = curr_->camera_.get();
        edge->point_ = Vector3d ( pts3d[index].x, pts3d[index].y, pts3d[index].z );
        edge->setMeasurement ( Vector2d ( pts2d[index].x, pts2d[index].y ) );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        optimizer.addEdge ( edge );
        // set the inlier map points 
        match_3dpts_[index]->matched_times_++;
    }

    optimizer.initializeOptimization();
    optimizer.optimize ( 10 );

    T_c_w_estimated_ = SE3<double> (
        pose->estimate().rotation(),
        pose->estimate().translation()
    );
    
    cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
}

bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    SE3<double> T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    if ( d.norm() > 5.0 )
    {
        cout<<"reject because motion is too large: "<<d.norm() <<endl;
        return false;
    }
    return true;
}

void VisualOdometry::optimizeMap()
{
    // remove the hardly seen and no visible points 
    for ( auto iter = map_->map_points_.begin(); iter != map_->map_points_.end(); )
    {
        if ( !curr_->isInFrame(iter->second->pos_) )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        float match_ratio = float(iter->second->matched_times_)/iter->second->visible_times_;
        if ( match_ratio < map_point_erase_ratio_ )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        
        double angle = getViewAngle( curr_, iter->second );
        if ( angle > M_PI/6. )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        if ( iter->second->good_ == false )
        {
            // TODO try triangulate this map point 
        }
        iter++;
    }
    
    if ( match_2dkp_index_.size()<100 )
        addMapPoints();
    if ( map_->map_points_.size() > 1000 )  
    {
        // TODO map is too large, remove some one 
        map_point_erase_ratio_ += 0.05;
    }
    else 
        map_point_erase_ratio_ = 0.1;
    cout<<"map points: "<<map_->map_points_.size()<<endl;
}

bool VisualOdometry::checkKeyFrame()
{
    SE3<double> T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometry::addKeyFrame()
{
    if ( map_->keyframes_.empty() )
    {
        // first key-frame, add all 3d points into map
        for ( size_t i=0; i<keypoints_curr_.size(); i++ )
        {
            double d = curr_->findDepth ( keypoints_curr_[i] );
            if ( d < 0 ) 
                continue;
            Vector3d p_world = ref_->camera_->pixel2world (
                Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ), curr_->T_c_w_, d
            );
            Vector3d n = p_world - ref_->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
            );
            map_->insertMapPoint( map_point );
        }
    }
    
    map_->insertKeyFrame ( curr_ );
    ref_ = curr_;
}


void VisualOdometry::addMapPoints()
{
    // add the new map points into map
    vector<bool> matched(keypoints_curr_.size(), false); 
    for ( int index:match_2dkp_index_ )
        matched[index] = true;
    for ( int i=0; i<keypoints_curr_.size(); i++ )
    {
        if ( matched[i] == true )   
            continue;
        double d = ref_->findDepth ( keypoints_curr_[i] );
        if ( d<0 )  
            continue;
        Vector3d p_world = ref_->camera_->pixel2world (
            Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ), 
            curr_->T_c_w_, d
        );
        Vector3d n = p_world - ref_->getCamCenter();
        n.normalize();
        MapPoint::Ptr map_point = MapPoint::createMapPoint(
            p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
        );
        map_->insertMapPoint( map_point );
    }
}

double VisualOdometry::getViewAngle ( Frame::Ptr frame, MapPoint::Ptr point )
{
    Vector3d n = point->pos_ - frame->getCamCenter();
    n.normalize();
    return acos( n.transpose()*point->norm_ );
}

}

// -------------- test the visual odometry -------------
#include <iomanip>
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp> 

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

int main ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: run_vo parameter_file"<<endl;
        return 1;
    }

    myslam::Config::setParameterFile ( argv[1] );
    myslam::VisualOdometry::Ptr vo ( new myslam::VisualOdometry );

    string dataset_dir = myslam::Config::get<string> ( "dataset_dir" );
    cout<<"dataset: "<<dataset_dir<<endl;
    ifstream fin ( dataset_dir+"/associate.txt" );
    if ( !fin )
    {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }

    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while ( !fin.eof() )
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( fin.good() == false )
            break;
    }

    myslam::Camera::Ptr camera ( new myslam::Camera );
    
    // visualization
    cv::viz::Viz3d vis("Visual Odometry");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    cv::Point3d cam_pos( 0, -1.0, -1.0 ), cam_focal_point(0,0,0), cam_y_dir(0,1,0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose( cam_pos, cam_focal_point, cam_y_dir );
    vis.setViewerPose( cam_pose );
    
    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    vis.showWidget( "World", world_coor );
    vis.showWidget( "Camera", camera_coor );

    cout<<"read total "<<rgb_files.size() <<" entries"<<endl;
    
    /////////////////////////////////////////////////////////////////////////////////
    
    ofstream fout("results/estimate_pose_03.txt");
    /////////////////////////////////////////////////////////////////////////////////
        
    
    
    for ( int i=1; i<=rgb_files.size(); i++ )
    {
    
        cout << "*********************************" << i << "*************************************" << endl;
        Mat color = cv::imread ( rgb_files[i] );
        Mat depth = cv::imread ( depth_files[i], -1 );
        if ( color.data==nullptr || depth.data==nullptr )
            break;
        myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

        boost::timer timer;
        vo->addFrame ( pFrame );
        cout<<"VO costs time: "<<timer.elapsed()<<endl;
        
        if ( vo->state_ == myslam::VisualOdometry::LOST )
            break;
        SE3<double> Twc = pFrame->T_c_w_.inverse();
        
        // show the map and the camera pose 
        cv::Affine3d M(
            cv::Affine3d::Mat3( 
                Twc.rotationMatrix()(0,0), Twc.rotationMatrix()(0,1), Twc.rotationMatrix()(0,2),
                Twc.rotationMatrix()(1,0), Twc.rotationMatrix()(1,1), Twc.rotationMatrix()(1,2),
                Twc.rotationMatrix()(2,0), Twc.rotationMatrix()(2,1), Twc.rotationMatrix()(2,2)
            ), 
            cv::Affine3d::Vec3(
                Twc.translation()(0,0), Twc.translation()(1,0), Twc.translation()(2,0)
            )
        );
        
        //0.4中加入了局部地图
        
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//把每一个关键帧的位姿写入文件

	cout << "估计位姿写入文件中..." << endl;
	//fout.open("../estimate_pose.txt");
	//fout << std::setprecision(16) << iter->second->id_ << " " << iter->second->time_stamp_ << " ";
	Eigen::Matrix3d R = Twc.rotationMatrix();//旋转矩阵
	Vector3d t = Twc.translation();//平移向量
	//将旋转矩阵转换为四元数
	Eigen::Quaterniond q = Eigen::Quaterniond ( R );	
	fout << setprecision(16) << pFrame->time_stamp_ << " ";
	fout << t(0,0) << " " << t(1,0) << " " << t(2,0) << " " << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << endl;
	cout << t(0,0) << " " << t(1,0) << " " << t(2,0) << " " << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << endl;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////   
        
        cout << endl;
        cout << endl;
        
        cv::imshow("image", color );
        cv::waitKey(1);
        vis.setWidgetPose( "Camera", M);
        vis.spinOnce(1, false);
    }

    return 0;
}

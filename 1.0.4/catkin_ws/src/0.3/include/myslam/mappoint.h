#ifndef MAPPOINT_H
#define MAPPOINT_H

namespace myslam
{
    
class Frame;
class MapPoint
{
public:
    typedef shared_ptr<MapPoint> Ptr;
    unsigned long      id_;        // ID
    bool        good_;      // wheter a good point 
    Vector3d    pos_;       // Position in world
    Vector3d    norm_;      // Normal of viewing direction 
    Mat         descriptor_; // Descriptor for matching 
    
    list<Frame*>    observed_frames_;   // frames that can observe this point //0.3+
    
    int         observed_times_;    // being observed by feature matching algo.
    int         matched_times_;     // being an inliner in pose estimation
    
    MapPoint();
    MapPoint( long id, Vector3d position, Vector3d norm );
    
    static MapPoint::Ptr createMapPoint();
};
}

#endif // MAPPOINT_H

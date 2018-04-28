/* Body Tracker Node using Nuitrack library

   Publish data as 2 messages:
   
   1. 2D position of person relative to head camera; allows for fast, smooth tracking
   Joint.proj:  position in normalized projective coordinates
   (x, y from 0.0 to 1.0, z is real)
   Astra Mini FOV: 60 horz, 49.5 vert (degrees)
   Publish using geometry_msgs/Pose2D.msg (body_tracking_pose_pub_)
   
   2. 3D position of joint in relation to robot (using TF)
   Joint.real: position in real world coordinates
   Publish using custom skeleton message (see Astra)
   Useful for turning robot to face / follow person (body_tracking_skeleton_pub_)
*/

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Int32.h"
#include <sstream>
#include "ros/console.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>  // setprecision

#include "geometry_msgs/PoseStamped.h"
#include <visualization_msgs/Marker.h>
#include "body_tracker_msgs/BodyTracker.h"  // Publish custom message


//For Nuitrack SDK
#include "nuitrack/Nuitrack.h"

namespace nuitrack_body_tracker
{
  using namespace tdv::nuitrack;

  class nuitrack_body_tracker_node 
  {
  public:

    nuitrack_body_tracker_node(std::string name) :
      _name(name)
    {
      ROS_INFO("%s: Starting...", _name.c_str());
      bool initialized = false;

      ros::NodeHandle nodeHandle("~");
      nodeHandle.param<std::string>("camera_depth_frame",camera_depth_frame_,"camera_depth_frame");

      // Publishers

      // Publish tracked person as a basic Pose message, for basic use
      // NOTE: We only provide to POSITION not full pose
      body_tracking_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>
        ("body_tracker/pose", 1); 

      // Publish tracked person upper body skeleton for advanced uses
      body_tracking_skeleton_pub_ = nh_.advertise<body_tracker_msgs::BodyTracker>
        ("body_tracker/skeleton", 1);

      // Publish markers to show where robot thinks person is in RViz
      marker_pub_ = nh_.advertise<visualization_msgs::Marker>
        ("body_tracker/marker", 1);

    }

    ~nuitrack_body_tracker_node()
    {
      ROS_INFO("nuitrack_body_tracker_node shutting down");
    }

    ///////////////////////////////////////////////////////////////////////////
    // Nuitrack callbacks
    // Copy depth frame data, received from Nuitrack, to texture to visualize
    void onNewDepthFrame(DepthFrame::Ptr frame)
    {
	    // std::cout << "Nuitrack: onNewDepthFrame callback" << std::endl;
    }

    /* Not used (yet?)
    void onNewRGBFrame(RGBFrame::Ptr frame)
    {
	    std::cout << "Nuitrack: onNewRGBFrame callback" << std::endl;
    }
    */

    void onUserUpdate(tdv::nuitrack::UserFrame::Ptr frame)
    {
	    // std::cout << "Nuitrack: onUserUpdate callback" << std::endl;
    }

    void onSkeletonUpdate(SkeletonData::Ptr userSkeletons)
    {
	    // std::cout << "Nuitrack: onSkeletonUpdate callback" << std::endl;
	    auto skeletons = userSkeletons->getSkeletons();
	    for (auto skeleton: skeletons)
	    {
	      std::cout << "Nuitrack: Skeleton.id = " << skeleton.id << std::endl;

        // Use JOINT_NECK to determine if we have a good lock on the person
        float tracking_confidence = skeleton.joints[JOINT_NECK].confidence;
        if (tracking_confidence < 0.15)
        {
  	      std::cout << "Nuitrack: Low Confidence (" << tracking_confidence << "), skipping"  << std::endl;
          continue;  // assume none of the joints are valid 
        }


        // Fill in message data from Nuitracker SDK data
        // camera z,x,y coordinates are mapped to ROS x,y,z coordinates 
        // All values are relative to camera position in meters (ie, in camera's TF frame)
        // ROS x = camera z - distance to person
        // ROS y = camera x - side to side
        // ROS z = camera y - vertical height, *relative to camera position*


        ///////////////////////////////////////////////////////////////
        // Basic Pose for person location tracking
        // This is for compatability with other trackers, which use PoseStamped messages
        geometry_msgs::PoseStamped body_pose;
        body_pose.header.frame_id = camera_depth_frame_;
        body_pose.header.stamp = ros::Time::now();

        body_pose.pose.position.x = skeleton.joints[JOINT_NECK].real.z / 1000.0;
        body_pose.pose.position.y = skeleton.joints[JOINT_NECK].real.x / -1000.0;
        body_pose.pose.position.z = skeleton.joints[JOINT_NECK].real.y / 1000.0;

        std::cout << std::setprecision(4) << std::setw(7) 
          << "Nuitrack: " << "JOINT_NECK"  
          << " x: " << (float)body_pose.pose.position.x 
          << " y: " << body_pose.pose.position.y
          << " z: " << body_pose.pose.position.z
          << "  Confidence: " << skeleton.joints[JOINT_NECK].confidence
          << std::endl;


        ///////////////////////////////////////////////////////////////
        // Skeleton Data for publishing more detail
        body_tracker_msgs::BodyTracker_ <body_tracker_msgs::BodyTracker> skeleton_data;

        // skeleton_data.frame_id = camera_depth_frame_;
        skeleton_data.body_id = skeleton.id;
        skeleton_data.tracking_status = 0; // TODO

        //skeleton_data.centerOfMass.x = 0.0;
        //skeleton_data.centerOfMass.y = 0.0;
        //skeleton_data.centerOfMass.z = 0.0;
        
        skeleton_data.joint_position_head.x = skeleton.joints[JOINT_HEAD].real.z / 1000.0;
        skeleton_data.joint_position_head.y = skeleton.joints[JOINT_HEAD].real.x / 1000.0;
        skeleton_data.joint_position_head.z = skeleton.joints[JOINT_HEAD].real.y / 1000.0;

        skeleton_data.joint_position_neck.x = skeleton.joints[JOINT_NECK].real.z / 1000.0;
        skeleton_data.joint_position_neck.y = skeleton.joints[JOINT_NECK].real.x / 1000.0;
        skeleton_data.joint_position_neck.z = skeleton.joints[JOINT_NECK].real.y / 1000.0;

        skeleton_data.joint_position_spine_top.x = skeleton.joints[JOINT_TORSO].real.z / 1000.0;
        skeleton_data.joint_position_spine_top.y = skeleton.joints[JOINT_TORSO].real.x / 1000.0;
        skeleton_data.joint_position_spine_top.z = skeleton.joints[JOINT_TORSO].real.y / 1000.0;

        skeleton_data.joint_position_spine_mid.x = skeleton.joints[JOINT_WAIST].real.z / 1000.0;
        skeleton_data.joint_position_spine_mid.y = skeleton.joints[JOINT_WAIST].real.x / 1000.0;
        skeleton_data.joint_position_spine_mid.z = skeleton.joints[JOINT_WAIST].real.y / 1000.0;

        skeleton_data.joint_position_spine_bottom.x = 0.0;
        skeleton_data.joint_position_spine_bottom.y = 0.0;
        skeleton_data.joint_position_spine_bottom.z = 0.0;

        skeleton_data.joint_position_left_shoulder.x = skeleton.joints[JOINT_LEFT_SHOULDER].real.z / 1000.0;
        skeleton_data.joint_position_left_shoulder.y = skeleton.joints[JOINT_LEFT_SHOULDER].real.x / 1000.0;
        skeleton_data.joint_position_left_shoulder.z = skeleton.joints[JOINT_LEFT_SHOULDER].real.y / 1000.0;

        skeleton_data.joint_position_left_elbow.x = skeleton.joints[JOINT_LEFT_ELBOW].real.z / 1000.0;
        skeleton_data.joint_position_left_elbow.y = skeleton.joints[JOINT_LEFT_ELBOW].real.x / 1000.0;
        skeleton_data.joint_position_left_elbow.z = skeleton.joints[JOINT_LEFT_ELBOW].real.y / 1000.0;

        skeleton_data.joint_position_left_hand.x = skeleton.joints[JOINT_LEFT_HAND].real.z / 1000.0;
        skeleton_data.joint_position_left_hand.y = skeleton.joints[JOINT_LEFT_HAND].real.x / 1000.0;
        skeleton_data.joint_position_left_hand.z = skeleton.joints[JOINT_LEFT_HAND].real.y / 1000.0;

        skeleton_data.joint_position_right_shoulder.x = skeleton.joints[JOINT_RIGHT_SHOULDER].real.z / 1000.0;
        skeleton_data.joint_position_right_shoulder.y = skeleton.joints[JOINT_RIGHT_SHOULDER].real.x / 1000.0;
        skeleton_data.joint_position_right_shoulder.z = skeleton.joints[JOINT_RIGHT_SHOULDER].real.y / 1000.0;

        skeleton_data.joint_position_right_elbow.x = skeleton.joints[JOINT_RIGHT_ELBOW].real.z / 1000.0;
        skeleton_data.joint_position_right_elbow.y = skeleton.joints[JOINT_RIGHT_ELBOW].real.x / 1000.0;
        skeleton_data.joint_position_right_elbow.z = skeleton.joints[JOINT_RIGHT_ELBOW].real.y / 1000.0;

        skeleton_data.joint_position_right_hand.x = skeleton.joints[JOINT_RIGHT_HAND].real.z / 1000.0;
        skeleton_data.joint_position_right_hand.y = skeleton.joints[JOINT_RIGHT_HAND].real.x / 1000.0;
        skeleton_data.joint_position_right_hand.z = skeleton.joints[JOINT_RIGHT_HAND].real.y / 1000.0;

        // Hand:  open (0), grasping (1), waving (2)
        /* TODO - see which of these actually work
	      GESTURE_WAVING          = 0,
	      GESTURE_SWIPE_LEFT      = 1,
	      GESTURE_SWIPE_RIGHT     = 2,
	      GESTURE_SWIPE_UP        = 3,
	      GESTURE_SWIPE_DOWN      = 4,
	      GESTURE_PUSH            = 5,
        in MSG: int32 gesture # 0 = none, 1 = right fist, 2 = left fist, 3 = waving

        */

	      for (int i = 0; i < userGestures_.size(); ++i)
	      {
          if(userGestures_[i].userId == skeleton.id) // match which person being tracked
          {
            skeleton_data.gesture = userGestures_[i].type; // TODO - map Nuitrack to my MSG enum
          }
		      printf("Gesture Recognized %d for User %d\n", userGestures_[i].type, userGestures_[i].userId);
	      }


        ////////////////////////////////////////////////////
        // Publish
        body_tracking_pose_pub_.publish(body_pose); // this is position only!
        body_tracking_skeleton_pub_.publish(skeleton_data); // full skeleton data
        PublishMarker(  // show marker at body_pose message location
          1, // ID
          body_pose.pose.position.x, // Distance to person = ROS X
          body_pose.pose.position.y, // side to side = ROS Y
          body_pose.pose.position.z, // Height = ROS Z
          1.0, 0.0, 1.0 ); // r,g,b
 	    }

    }


    void onHandUpdate(HandTrackerData::Ptr handData)
    {
	    // std::cout << "Nuitrack: onHandUpdate callback" << std::endl;
    }


    void onNewGesture(GestureData::Ptr gestureData)
    {
	    std::cout << "Nuitrack: onNewGesture callback" << std::endl;

	    userGestures_ = gestureData->getGestures(); // Save for use in next skeleton frame
	    for (int i = 0; i < userGestures_.size(); ++i)
	    {
		    printf("Recognized %d from %d\n", userGestures_[i].type, userGestures_[i].userId);
	    }

    }


    void PublishMarker(int id, float x, float y, float z, float color_r, float color_g, float color_b)
    {
      // Display marker for RVIZ to show where robot thinks person is
      // For Markers info, see http://wiki.ros.org/rviz/Tutorials/Markers%3A%20Basic%20Shapes

      // ROS_INFO("DBG: PublishMarker called");
      //if( id != 1)
      // printf ("DBG PublishMarker called for %f, %f, %f\n", x,y,z);

      visualization_msgs::Marker marker;
      marker.header.frame_id = camera_depth_frame_;
      marker.header.stamp = ros::Time::now();

      // Any marker sent with the same namespace and id will overwrite the old one
      marker.ns = _name;
      marker.id = id; // This must be id unique for each marker

      uint32_t shape = visualization_msgs::Marker::SPHERE;
      marker.type = shape;

      // Set the marker action.  Options are ADD, DELETE, and DELETEALL
      marker.action = visualization_msgs::Marker::ADD;
      marker.color.r = color_r;
      marker.color.g = color_g; 
      marker.color.b = color_b;
      marker.color.a = 1.0;
      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;

      marker.scale.x = 0.1; // size of marker in meters
      marker.scale.y = 0.1;
      marker.scale.z = 0.1;  

      marker.pose.position.x = x;
      marker.pose.position.y = y;
      marker.pose.position.z = z;

      // ROS_INFO("DBG: Publishing Marker");
      marker_pub_.publish(marker);

    }


    // Publish 2D position of person, relative to camera
    // useful for direct control of servo pan/tilt for tracking
    // Example: publishJoint2D("JOINT_NECK", joints[JOINT_NECK]);

    /* Not currently used
    void publishJoint2D(const char *name, const tdv::nuitrack::Joint& joint)
    {
      const float ASTRA_MINI_FOV_X =  1.047200; // (60 degrees horizontal)
      const float ASTRA_MINI_FOV_Y = -0.863938; // (49.5 degrees vertical)
      if (joint.confidence < 0.15)
      {
        return;  // ignore low confidence joints
      }

      // Convert projection to radians
      // proj is 0.0 (left) --> 1.0 (right)
      float radians_x = (joint.proj.x - 0.5) * ASTRA_MINI_FOV_X;
      float radians_y = (joint.proj.y - 0.5) * ASTRA_MINI_FOV_Y;
      std::cout << std::setprecision(4) << std::setw(7) 
        << "Nuitrack: " << name  
        << " x: " << joint.proj.x << " (" << radians_x << ")  y: " 
        << joint.proj.y << " (" << radians_y << ")" 
        // << "  Confidence: " << joint.confidence 
        << std::endl;

      // Future? Add in servo position to get absolute position relative to the robot body
      // This allows the subscriber to use these values directly for servo control
    }
    */


    ///////////////////////////////////////////////////////////////////////////
    void Init(const std::string& config)
    {
	    // Initialize Nuitrack first, then create Nuitrack modules
      ROS_INFO("%s: Initializing...", _name.c_str());
	    // std::cout << "Nuitrack: Initializing..." << std::endl;

	    std::cout << 
      "\n============ IGNORE ERRORS THAT SAY 'Couldnt open device...' ===========\n" 
      << std::endl;

	    try
	    {
		    tdv::nuitrack::Nuitrack::init(config);
	    }
	    catch (const tdv::nuitrack::Exception& e)
	    {
		    std::cerr << 
        "Can not initialize Nuitrack (ExceptionType: " << e.type() << ")" << std::endl;
		    exit(EXIT_FAILURE);
	    }

	    // Create all required Nuitrack modules

	    std::cout << "Nuitrack: DepthSensor::create()" << std::endl;
	    depthSensor_ = tdv::nuitrack::DepthSensor::create();
	    // Bind to event new frame
	    depthSensor_->connectOnNewFrame(std::bind(
      &nuitrack_body_tracker_node::onNewDepthFrame, this, std::placeholders::_1));
	
      outputMode_ = depthSensor_->getOutputMode();
	    width_ = outputMode_.xres;
	    height_ = outputMode_.yres;
      std::cout << "========= Nuitrack: GOT DEPTH SENSOR =========" << std::endl;
	    std::cout << "Nuitrack: Depth:  width = " << width_ << "  height = " << height_ << std::endl;

      /*  Color frame not currently used
	    std::cout << "Nuitrack: ColorSensor::create()" << std::endl;
	    colorSensor_ = tdv::nuitrack::ColorSensor::create();
	    // Bind to event new frame
	    colorSensor_->connectOnNewFrame(std::bind(
        &nuitrack_body_tracker_node::onNewRGBFrame, this, std::placeholders::_1));
	    */

	    std::cout << "Nuitrack: UserTracker::create()" << std::endl;
	    userTracker_ = tdv::nuitrack::UserTracker::create();
	    // Bind to event update user tracker
	    userTracker_->connectOnUpdate(std::bind(
        &nuitrack_body_tracker_node::onUserUpdate, this, std::placeholders::_1));
	
	    std::cout << "Nuitrack: SkeletonTracker::create()" << std::endl;
	    skeletonTracker_ = tdv::nuitrack::SkeletonTracker::create();
	    // Bind to event update skeleton tracker
	    skeletonTracker_->connectOnUpdate(std::bind(
        &nuitrack_body_tracker_node::onSkeletonUpdate, this, std::placeholders::_1));
	
	    std::cout << "Nuitrack: HandTracker::create()" << std::endl;
	    handTracker_ = tdv::nuitrack::HandTracker::create();
	    // Bind to event update Hand tracker
	    handTracker_->connectOnUpdate(std::bind(
        &nuitrack_body_tracker_node::onHandUpdate, this, std::placeholders::_1));
	
	    std::cout << "Nuitrack: GestureRecognizer::create()" << std::endl;
	    gestureRecognizer_ = tdv::nuitrack::GestureRecognizer::create();
	    gestureRecognizer_->connectOnNewGestures(std::bind(
        &nuitrack_body_tracker_node::onNewGesture, this, std::placeholders::_1));

      ROS_INFO("%s: Init complete.  Waiting for frames...", _name.c_str());

    }

    void Run()
    {
	    // Initialize Nuitrack first, then create Nuitrack modules
      ROS_INFO("%s: Running...", _name.c_str());
	    // std::cout << "Nuitrack: Running..." << std::endl;
      // Start Nuitrack
      try
      {
          tdv::nuitrack::Nuitrack::run();
      }
      catch (const tdv::nuitrack::Exception& e)
      {
          std::cerr << "Can not start Nuitrack (ExceptionType: " 
            << e.type() << ")" << std::endl;
          return;
      }

      ROS_INFO("%s: Waiting for person to be detected...", _name.c_str());

      // Run Loop
      ros::Rate r(30); // hz
      while (ros::ok())
      {
	        // std::cout << "Nuitrack: Looping..." << std::endl;

          try
          {
              // Wait for new person tracking data
              tdv::nuitrack::Nuitrack::waitUpdate(skeletonTracker_);
          }
          catch (tdv::nuitrack::LicenseNotAcquiredException& e)
          {
              std::cerr << "LicenseNotAcquired exception (ExceptionType: " 
                << e.type() << ")" << std::endl;
              break;
          }
          catch (const tdv::nuitrack::Exception& e)
          {
              std::cerr << "Nuitrack update failed (ExceptionType: " 
                << e.type() << ")" << std::endl;
          }

	        // std::cout << "Nuitrack: Sleeping..." << std::endl;
          ros::spinOnce();
          r.sleep();
      }

      // Release Nuitrack
      try
      {
          tdv::nuitrack::Nuitrack::release();
      }
      catch (const tdv::nuitrack::Exception& e)
      {
          std::cerr << "Nuitrack release failed (ExceptionType: " 
            << e.type() << ")" << std::endl;
      }

    } // Run()


  private:
    /////////////// DATA MEMBERS /////////////////////

    std::string _name;
    ros::NodeHandle nh_;
    std::string camera_depth_frame_;
    ros::Publisher body_tracking_pose_pub_;
    ros::Publisher body_tracking_skeleton_pub_;
    ros::Publisher marker_pub_;

	  int width_, height_;
	  tdv::nuitrack::OutputMode outputMode_;
	  std::vector<tdv::nuitrack::Gesture> userGestures_;
	  tdv::nuitrack::DepthSensor::Ptr depthSensor_;
	  tdv::nuitrack::ColorSensor::Ptr colorSensor_;
	  tdv::nuitrack::UserTracker::Ptr userTracker_;
	  tdv::nuitrack::SkeletonTracker::Ptr skeletonTracker_;
	  tdv::nuitrack::HandTracker::Ptr handTracker_;
	  tdv::nuitrack::GestureRecognizer::Ptr gestureRecognizer_;

  };
};  // namespace nuitrack_body_tracker


  // The main entry point for this node.
  int main( int argc, char *argv[] )
  {
    using namespace nuitrack_body_tracker;
    ros::init( argc, argv, "nuitrack_body_tracker" );
    nuitrack_body_tracker_node node(ros::this_node::getName());
	  node.Init("");
		node.Run();

    return 0;
  }





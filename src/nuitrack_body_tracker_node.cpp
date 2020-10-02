#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Int32.h"
#include <sstream>
#include "ros/console.h"

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iostream>
#include <iomanip> // setprecision

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/Marker.h>
#include <body_tracker_msgs/BodyTracker.h>      // Publish custom message
#include <body_tracker_msgs/BodyTrackerArray.h> // Custom message, multiple people
#include <body_tracker_msgs/Skeleton.h>         // Publish custom message

//For Nuitrack SDK
#include "nuitrack/Nuitrack.h"
#define KEY_JOINT_TO_TRACK JOINT_LEFT_COLLAR // JOINT_TORSO // JOINT_NECK

// For Face JSON parsing
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

// For Point Cloud publishing
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <sensor_msgs/JointState.h>
// #include <pcl/point_types.h>

const bool ENABLE_PUBLISHING_FRAMES = true;

namespace nuitrack_body_tracker
{
  using namespace tdv::nuitrack;
  //namespace pt = boost::property_tree;

  class nuitrack_body_tracker_node
  {
  public:
    nuitrack_body_tracker_node(std::string name) : _name(name)
    {
      ROS_INFO("%s: Starting...", _name.c_str());

      ros::NodeHandle nodeHandle("~");
      nodeHandle.param<std::string>("camera_depth_frame", camera_depth_frame_, "camera_depth_frame");
      nodeHandle.param<std::string>("camera_color_frame", camera_color_frame_, "camera_color_frame");
      nodeHandle.getParam("bone_confidence_threshold", confidence_value);

      // Publishers and Subscribers

      // Publish tracked person in 2D and 3D
      // 2D: x,y in camera frame.   3D: x,y,z in world coordinates
      body_tracking_position_pub_ = nh_.advertise<body_tracker_msgs::BodyTracker>("body_tracker/position", 1);

      body_tracking_array_pub_ =
          nh_.advertise<body_tracker_msgs::BodyTrackerArray>("body_tracker_array/position", 1);

      // Publish tracked person upper body skeleton for advanced uses
      body_tracking_skeleton_pub_ = nh_.advertise<body_tracker_msgs::Skeleton>("body_tracker/skeleton", 1);

      // Publish markers to show where robot thinks person is in RViz
      marker_pub_ = nh_.advertise<visualization_msgs::Marker>("body_tracker/marker", 1);

      // Publish the depth frame for other nodes
      depth_image_pub_ = nh_.advertise<sensor_msgs::Image>("camera/depth/image", 1);
      color_image_pub_ = nh_.advertise<sensor_msgs::Image>("camera/color/image", 1);
      depth_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("camera/depth_cloud", 1);

      //Added mechaless command
      mechaless_pygame_state_sub_ = nh_.subscribe("/mechaless_pygame_command", 1, &nuitrack_body_tracker_node::PygameCallBack, this);
    }

    ~nuitrack_body_tracker_node()
    {
        // Release Nuitrack
              try
              {
                tdv::nuitrack::Nuitrack::release();
                std::cout << "========= Nuitrack: RELEASE COMPLETED =========" << std::endl;

              }
              catch (const tdv::nuitrack::Exception &e)
              {
                std::cerr << "Nuitrack release failed (ExceptionType: "
                          << e.type() << ")" << std::endl;
              }
      ROS_INFO("nuitrack_body_tracker_node shutting down");
    }

    ///////////////////////////////////////////////////////////////////////////
    // Nuitrack callbacks
    // WARNING!  THIS CODE ASSUMES COLOR AND DEPTH ARE SAME RESOLUTION!
    // TO FIX THIS, SEE NUITRACK GL SAMPLE

    void onNewColorFrame(RGBFrame::Ptr frame)
    {
      //ROS_INFO("DBG: Nuitrack::onNewColorFrame(), Frame = %d", ++color_frame_number_);

      if (!ENABLE_PUBLISHING_FRAMES)
      {
        return;
      }

      int _width = frame->getCols();
      int _height = frame->getRows();
      //std::cout << "DBG COLOR:  Width = " << _width << " Height = " << _height << std::endl;

      // Point Cloud message for colorized depth cloud
      int numpoints = _width * _height;
      //cloud_msg_ = new(sensor_msgs::PointCloud2);

      sensor_msgs::PointCloud2Iterator<uint8_t> out_r(cloud_msg_, "r");
      sensor_msgs::PointCloud2Iterator<uint8_t> out_g(cloud_msg_, "g");
      sensor_msgs::PointCloud2Iterator<uint8_t> out_b(cloud_msg_, "b");

      sensor_msgs::Image color_msg;

      const tdv::nuitrack::Color3 *colorPtr = frame->getData();

      color_msg.header.stamp = ros::Time::now();
      color_msg.header.frame_id = camera_color_frame_;
      color_msg.height = _height;
      color_msg.width = _width;
      color_msg.encoding = "rgb8"; //sensor_msgs::image_encodings::TYPE_16UC1;
      color_msg.is_bigendian = false;

      color_msg.step = 3 * _width; // sensor_msgs::ImagePtr row step size

      for (size_t row = 0; row < _height; ++row)
      {
        for (size_t col = 0; col < _width; ++col)
        {
          color_msg.data.push_back((colorPtr + col)->red);
          color_msg.data.push_back((colorPtr + col)->green);
          color_msg.data.push_back((colorPtr + col)->blue);

          *out_r = (colorPtr + col)->red; // pointcloud
          *out_g = (colorPtr + col)->green;
          *out_b = (colorPtr + col)->blue;
          ++out_r;
          ++out_g;
          ++out_b;
        }
        colorPtr += _width; // Next row
      }

      // Publish color frame
      color_image_pub_.publish(color_msg);
    }

    void onNewDepthFrame(DepthFrame::Ptr frame)
    {
    }


    void onSkeletonUpdate(SkeletonData::Ptr userSkeletons)
    {

      // Message for array of body detections
      body_tracker_msgs::BodyTrackerArray body_tracker_array_msg;
      body_tracker_array_msg.header.frame_id = camera_depth_frame_;
      ros::Time frame_time_stamp = ros::Time::now();
      body_tracker_array_msg.header.stamp = frame_time_stamp;

      // process skeletons for each user found
      auto skeletons = userSkeletons->getSkeletons();
      float conf = 0.6;
      int identify_point_w = -1;
      int identify_point_d = -1;
      int identify_point = 0;
      float minimum = -1.0f;
      double min_dist = 999999.0;
      for (auto pre_skeleton : skeletons)
      {
        float min_num = -1000.0f;
        double dist;
        // std::cout << "Nuitrack: Skeleton.id = " << skeleton.id << std::endl;

        // Use KEY_JOINT_TO_TRACK to determine if we have a good lock on the person
        float tracking_confidence = pre_skeleton.joints[KEY_JOINT_TO_TRACK].confidence;
        //トラッキングの精度が悪い場合は排除
        if (tracking_confidence < confidence_value)
        {
          std::cout << "Nuitrack: ID " << pre_skeleton.id << " Low Confidence ("
                    << tracking_confidence << "), skipping" << std::endl;
          continue; // assume none of the joints are valid
        }
        ///////////////////////////////////////////////////////////////
        // Waist position filter program detect only one person.
        //腰位置の移動位置を比較して直前と最も近い人をずっとロックし続ける
        else
        {
          if (waist_set == 1)
          {
            if (pre_skeleton.joints[JOINT_WAIST].confidence > conf)
            {
              //腰の精度が一定値移動なら前フレームの腰の位置と比較する
              dist = std::pow((pre_waist_position[0] - pre_skeleton.joints[JOINT_WAIST].proj.x), 2.0) + std::pow((pre_waist_position[1] - pre_skeleton.joints[JOINT_WAIST].proj.y), 2.0);
              if (min_dist > dist)
              {
                //もしもmin_distよりも比較位置が短いならば、その腰座標を持つ人を前フレームで認識した人として扱う。
                min_dist = dist;
                identify_point_w = pre_skeleton.id;
                tmp_waist_position = {pre_skeleton.joints[JOINT_WAIST].proj.x, pre_skeleton.joints[JOINT_WAIST].proj.y};
              }
              //printf("ID：%dの位置は{%f,%f}、移動距離は%fで、前フレームの位置は{%f,%f}、現最小距離は%f、IDは%dで、waist_setは%dです\n", pre_skeleton.id, pre_skeleton.joints[JOINT_WAIST].proj.x, pre_skeleton.joints[JOINT_WAIST].proj.y, dist, pre_waist_position[0], pre_waist_position[1], min_dist, identify_point_w, waist_set);
            }
            waist_set = 0;
          }
          ///////////////////////////////////////////////////////////////
          // Depth filter program detect only one person.
          //上記腰の移動位置で比較後、腰の位置があまりにも離れた位置に出た場合、
          //各体の座標の中で最もカメラからの奥行き座標が近い物を取得し
          //その座標がカメラから最も近い人にトラッキング権利を譲渡する
          //（なお、最初の認識では必ずこれが呼ばれる）
          if (min_dist > 0.5 && waist_set == 0)
          {
            if (pre_skeleton.joints[JOINT_HEAD].confidence > conf)
            {
              if (min_num == -1000.0f)
              {
                min_num = pre_skeleton.joints[JOINT_HEAD].proj.z;
              }
              else
              {
                if (min_num > pre_skeleton.joints[JOINT_HEAD].proj.z)
                {
                  min_num = pre_skeleton.joints[JOINT_HEAD].proj.z;
                }
              }
            }
            if (pre_skeleton.joints[JOINT_NECK].confidence > conf)
            {
              if (min_num == -1000.0f)
              {
                min_num = pre_skeleton.joints[JOINT_NECK].proj.z;
              }
              else
              {
                if (min_num > pre_skeleton.joints[JOINT_NECK].proj.z)
                {
                  min_num = pre_skeleton.joints[JOINT_NECK].proj.z;
                }
              }
            }
            if (pre_skeleton.joints[JOINT_LEFT_COLLAR].confidence > conf)
            {
              if (min_num == -1000.0f)
              {
                min_num = pre_skeleton.joints[JOINT_LEFT_COLLAR].proj.z;
              }
              else
              {
                if (min_num > pre_skeleton.joints[JOINT_LEFT_COLLAR].proj.z)
                {
                  min_num = pre_skeleton.joints[JOINT_LEFT_COLLAR].proj.z;
                }
              }
            }
            if (pre_skeleton.joints[JOINT_TORSO].confidence > conf)
            {
              if (min_num == -1000.0f)
              {
                min_num = pre_skeleton.joints[JOINT_TORSO].proj.z;
              }
              else
              {
                if (min_num > pre_skeleton.joints[JOINT_TORSO].proj.z)
                {
                  min_num = pre_skeleton.joints[JOINT_TORSO].proj.z;
                }
              }
            }
            if (pre_skeleton.joints[JOINT_TORSO].confidence > conf)
            {
              if (min_num == -1000.0f)
              {
                min_num = pre_skeleton.joints[JOINT_TORSO].proj.z;
              }
              else
              {
                if (min_num > pre_skeleton.joints[JOINT_TORSO].proj.z)
                {
                  min_num = pre_skeleton.joints[JOINT_TORSO].proj.z;
                }
              }
            }
            if (pre_skeleton.joints[JOINT_WAIST].confidence > conf)
            {
              if (min_num == -1000.0f)
              {
                min_num = pre_skeleton.joints[JOINT_WAIST].proj.z;
              }
              else
              {
                if (min_num > pre_skeleton.joints[JOINT_WAIST].proj.z)
                {
                  min_num = pre_skeleton.joints[JOINT_WAIST].proj.z;
                }
              }
            }
            if (pre_skeleton.joints[JOINT_WAIST].confidence > conf)
            {
              if (min_num == -1000.0f)
              {
                min_num = pre_skeleton.joints[JOINT_WAIST].proj.z;
              }
              else
              {
                if (min_num > pre_skeleton.joints[JOINT_WAIST].proj.z)
                {
                  min_num = pre_skeleton.joints[JOINT_WAIST].proj.z;
                }
              }
            }
            if (pre_skeleton.joints[JOINT_LEFT_SHOULDER].confidence > conf)
            {
              if (min_num == -1000.0f)
              {
                min_num = pre_skeleton.joints[JOINT_LEFT_SHOULDER].proj.z;
              }
              else
              {
                if (min_num > pre_skeleton.joints[JOINT_LEFT_SHOULDER].proj.z)
                {
                  min_num = pre_skeleton.joints[JOINT_LEFT_SHOULDER].proj.z;
                }
              }
            }
            if (pre_skeleton.joints[JOINT_LEFT_ELBOW].confidence > conf)
            {
              if (min_num == -1000.0f)
              {
                min_num = pre_skeleton.joints[JOINT_LEFT_ELBOW].proj.z;
              }
              else
              {
                if (min_num > pre_skeleton.joints[JOINT_LEFT_ELBOW].proj.z)
                {
                  min_num = pre_skeleton.joints[JOINT_LEFT_ELBOW].proj.z;
                }
              }
            }
            if (pre_skeleton.joints[JOINT_LEFT_WRIST].confidence > conf)
            {
              if (min_num == -1000.0f)
              {
                min_num = pre_skeleton.joints[JOINT_LEFT_WRIST].proj.z;
              }
              else
              {
                if (min_num > pre_skeleton.joints[JOINT_LEFT_WRIST].proj.z)
                {
                  min_num = pre_skeleton.joints[JOINT_LEFT_WRIST].proj.z;
                }
              }
            }
            if (pre_skeleton.joints[JOINT_LEFT_HAND].confidence > conf)
            {
              if (min_num == -1000.0f)
              {
                min_num = pre_skeleton.joints[JOINT_LEFT_HAND].proj.z;
              }
              else
              {
                if (min_num > pre_skeleton.joints[JOINT_LEFT_HAND].proj.z)
                {
                  min_num = pre_skeleton.joints[JOINT_LEFT_HAND].proj.z;
                }
              }
            }
            if (pre_skeleton.joints[JOINT_RIGHT_SHOULDER].confidence > conf)
            {
              if (min_num == -1000.0f)
              {
                min_num = pre_skeleton.joints[JOINT_RIGHT_SHOULDER].proj.z;
              }
              else
              {
                if (min_num > pre_skeleton.joints[JOINT_RIGHT_SHOULDER].proj.z)
                {
                  min_num = pre_skeleton.joints[JOINT_RIGHT_SHOULDER].proj.z;
                }
              }
            }
            if (pre_skeleton.joints[JOINT_RIGHT_ELBOW].confidence > conf)
            {
              if (min_num == -1000.0f)
              {
                min_num = pre_skeleton.joints[JOINT_RIGHT_ELBOW].proj.z;
              }
              else
              {
                if (min_num > pre_skeleton.joints[JOINT_RIGHT_ELBOW].proj.z)
                {
                  min_num = pre_skeleton.joints[JOINT_RIGHT_ELBOW].proj.z;
                }
              }
            }
            if (pre_skeleton.joints[JOINT_RIGHT_WRIST].confidence > conf)
            {
              if (min_num == -1000.0f)
              {
                min_num = pre_skeleton.joints[JOINT_RIGHT_WRIST].proj.z;
              }
              else
              {
                if (min_num > pre_skeleton.joints[JOINT_RIGHT_WRIST].proj.z)
                {
                  min_num = pre_skeleton.joints[JOINT_RIGHT_WRIST].proj.z;
                }
              }
            }
            if (pre_skeleton.joints[JOINT_RIGHT_HAND].confidence > conf)
            {
              if (min_num == -1000.0f)
              {
                min_num = pre_skeleton.joints[JOINT_RIGHT_HAND].proj.z;
              }
              else
              {
                if (min_num > pre_skeleton.joints[JOINT_RIGHT_HAND].proj.z)
                {
                  min_num = pre_skeleton.joints[JOINT_RIGHT_HAND].proj.z;
                }
              }
            }
            if (minimum == -1.0f && min_num > 0)
            {
              minimum = min_num;
              identify_point_d = pre_skeleton.id;
              tmp_waist_position = {pre_skeleton.joints[JOINT_WAIST].proj.x, pre_skeleton.joints[JOINT_WAIST].proj.y};
            }
            else
            {
              if (minimum > min_num && min_num > 0)
              {
                minimum = min_num;
                identify_point_d = pre_skeleton.id;
                tmp_waist_position = {pre_skeleton.joints[JOINT_WAIST].proj.x, pre_skeleton.joints[JOINT_WAIST].proj.y};
              }
            }
            identify_point = identify_point_d;
            pre_waist_position = tmp_waist_position;
            waist_set = 1;
            //printf("ID：%dの最小値は%fで、現最小値は%f、IDは%dです\n",pre_skeleton.id,min_num,minimum,identify_point_d);
          }
          //腰の吹っ飛びがなければ腰の移動位置が最も少なかったidを優先する
          else
          {
            pre_waist_position = tmp_waist_position;
            //printf("結局、初期腰はID：%dで、座標位置(%f, %f)です\n",identify_point_w,pre_waist_position[0],pre_waist_position[1]);
            identify_point = identify_point_w;
            waist_set = 1;
          }
          /*
          //カメラに最も近い座標を算出し、waist_set==0の場合
          if (minimum != -1.0f && waist_set == 0)
          {
            pre_waist_position = {pre_skeleton.joints[JOINT_WAIST].proj.x, pre_skeleton.joints[JOINT_WAIST].proj.y};
            waist_set = 1;
            //printf("初期腰はID：%dで、座標位置(%f, %f)、距離は%fです\n",identify_point_d,pre_waist_position[0],pre_waist_position[1],minimum);
          }
          */
        }
      }
      //If minimum distance is not smaller than distance threshold, use depth filter
      for (auto skeleton : skeletons)
      {
        if (skeleton.id != identify_point)
        {
          continue;
        }
        ///////////////////////////////////////////////////////////////
        // Position data in 2D and 3D for tracking people
        body_tracker_msgs::BodyTracker person_data;

        person_data.body_id = skeleton.id;
        person_data.tracking_status = 0; // TODO
        person_data.gesture = -1;        // No gesture
        person_data.face_found = false;
        person_data.face_left = 0;
        person_data.face_top = 0;
        person_data.face_width = 0;
        person_data.face_height = 0;
        person_data.age = 0;
        person_data.gender = 0;
        person_data.name = "";

        if (skeleton.id != last_id_)
        {
          ROS_INFO("%s: detected person ID %d", _name.c_str(), skeleton.id);
          last_id_ = skeleton.id;
        }

        ///////////////////////////////////////////////////////////////
        // 2D position for camera servo tracking
        const float ASTRA_MINI_FOV_X = -1.047200; // (60 degrees horizontal)
        const float ASTRA_MINI_FOV_Y = -0.863938; // (49.5 degrees vertical)

        // Convert projection to radians
        // proj is 0.0 (left) --> 1.0 (right)
        geometry_msgs::Pose2D track2d;
        track2d.x = (skeleton.joints[KEY_JOINT_TO_TRACK].proj.x - 0.5) * ASTRA_MINI_FOV_X;
        track2d.y = (skeleton.joints[KEY_JOINT_TO_TRACK].proj.y - 0.5) * ASTRA_MINI_FOV_Y;
        track2d.theta = (float)skeleton.id;

        person_data.position2d.x =
            (skeleton.joints[KEY_JOINT_TO_TRACK].proj.x - 0.5) * ASTRA_MINI_FOV_X;
        person_data.position2d.y =
            (skeleton.joints[KEY_JOINT_TO_TRACK].proj.y - 0.5) * ASTRA_MINI_FOV_Y;
        person_data.position2d.z = skeleton.joints[KEY_JOINT_TO_TRACK].proj.z / 1000.0;

        ///////////////////////////////////////////////////////////////
        // Face Data
        // if the same ID as skeleton id, publish face data too

        std::string face_info = tdv::nuitrack::Nuitrack::getInstancesJson();
        ///////////////////////////////////////////////////////////////
        // Skeleton Data for publishing more detail
        body_tracker_msgs::Skeleton_<body_tracker_msgs::Skeleton> skeleton_data;

        // skeleton_data.frame_id = camera_depth_frame_;
        skeleton_data.body_id = skeleton.id;
        skeleton_data.tracking_status = 0; // TODO

        // *** POSITION 3D ***
        person_data.position3d.x = skeleton.joints[KEY_JOINT_TO_TRACK].real.z / 1000.0;
        person_data.position3d.y = skeleton.joints[KEY_JOINT_TO_TRACK].real.x / 1000.0;
        person_data.position3d.z = skeleton.joints[KEY_JOINT_TO_TRACK].real.y / 1000.0;

        float conf = 0.6;

        if (skeleton.joints[JOINT_HEAD].confidence > conf)
        {
          skeleton_data.joint_position_head_real.x = skeleton.joints[JOINT_HEAD].real.z / 1000.0;
          skeleton_data.joint_position_head_real.y = skeleton.joints[JOINT_HEAD].real.x / 1000.0;
          skeleton_data.joint_position_head_real.z = skeleton.joints[JOINT_HEAD].real.y / 1000.0;
        }
        else
        {
          skeleton_data.joint_position_head_real.x = -9999;
          skeleton_data.joint_position_head_real.y = -9999;
          skeleton_data.joint_position_head_real.z = -9999;
        }

        if (skeleton.joints[JOINT_NECK].confidence > conf)
        {
          skeleton_data.joint_position_neck_real.x = skeleton.joints[JOINT_NECK].real.z / 1000.0;
          skeleton_data.joint_position_neck_real.y = skeleton.joints[JOINT_NECK].real.x / 1000.0;
          skeleton_data.joint_position_neck_real.z = skeleton.joints[JOINT_NECK].real.y / 1000.0;
        }
        else
        {
          skeleton_data.joint_position_neck_real.x = -9999;
          skeleton_data.joint_position_neck_real.y = -9999;
          skeleton_data.joint_position_neck_real.z = -9999;
        }

        if (skeleton.joints[JOINT_LEFT_COLLAR].confidence > conf)
        {
          skeleton_data.joint_position_left_collar_real.x = skeleton.joints[JOINT_LEFT_COLLAR].real.z / 1000.0;
          skeleton_data.joint_position_left_collar_real.y = skeleton.joints[JOINT_LEFT_COLLAR].real.x / 1000.0;
          skeleton_data.joint_position_left_collar_real.z = skeleton.joints[JOINT_LEFT_COLLAR].real.y / 1000.0;
        }
        else
        {
          skeleton_data.joint_position_left_collar_real.x = -9999;
          skeleton_data.joint_position_left_collar_real.y = -9999;
          skeleton_data.joint_position_left_collar_real.z = -9999;
        }

        if (skeleton.joints[JOINT_TORSO].confidence > conf)
        {
          skeleton_data.joint_position_torso_real.x = skeleton.joints[JOINT_TORSO].real.z / 1000.0;
          skeleton_data.joint_position_torso_real.y = skeleton.joints[JOINT_TORSO].real.x / 1000.0;
          skeleton_data.joint_position_torso_real.z = skeleton.joints[JOINT_TORSO].real.y / 1000.0;
        }
        else
        {
          skeleton_data.joint_position_torso_real.x = -9999;
          skeleton_data.joint_position_torso_real.y = -9999;
          skeleton_data.joint_position_torso_real.z = -9999;
        }

        if (skeleton.joints[JOINT_WAIST].confidence > conf)
        {
          skeleton_data.joint_position_waist_real.x = skeleton.joints[JOINT_WAIST].real.z / 1000.0;
          skeleton_data.joint_position_waist_real.y = skeleton.joints[JOINT_WAIST].real.x / 1000.0;
          skeleton_data.joint_position_waist_real.z = skeleton.joints[JOINT_WAIST].real.y / 1000.0;
        }
        else
        {
          skeleton_data.joint_position_waist_real.x = -9999;
          skeleton_data.joint_position_waist_real.y = -9999;
          skeleton_data.joint_position_waist_real.z = -9999;
        }

        if (skeleton.joints[JOINT_LEFT_SHOULDER].confidence > conf)
        {
          skeleton_data.joint_position_left_shoulder_real.x = skeleton.joints[JOINT_LEFT_SHOULDER].real.z / 1000.0;
          skeleton_data.joint_position_left_shoulder_real.y = skeleton.joints[JOINT_LEFT_SHOULDER].real.x / 1000.0;
          skeleton_data.joint_position_left_shoulder_real.z = skeleton.joints[JOINT_LEFT_SHOULDER].real.y / 1000.0;
        }
        else
        {
          skeleton_data.joint_position_left_shoulder_real.x = -9999;
          skeleton_data.joint_position_left_shoulder_real.y = -9999;
          skeleton_data.joint_position_left_shoulder_real.z = -9999;
        }

        if (skeleton.joints[JOINT_LEFT_ELBOW].confidence > conf)
        {
          skeleton_data.joint_position_left_elbow_real.x = skeleton.joints[JOINT_LEFT_ELBOW].real.z / 1000.0;
          skeleton_data.joint_position_left_elbow_real.y = skeleton.joints[JOINT_LEFT_ELBOW].real.x / 1000.0;
          skeleton_data.joint_position_left_elbow_real.z = skeleton.joints[JOINT_LEFT_ELBOW].real.y / 1000.0;
        }
        else
        {
          skeleton_data.joint_position_left_elbow_real.x = -9999;
          skeleton_data.joint_position_left_elbow_real.y = -9999;
          skeleton_data.joint_position_left_elbow_real.z = -9999;
        }

        if (skeleton.joints[JOINT_LEFT_WRIST].confidence > conf)
        {
          skeleton_data.joint_position_left_wrist_real.x = skeleton.joints[JOINT_LEFT_WRIST].real.z / 1000.0;
          skeleton_data.joint_position_left_wrist_real.y = skeleton.joints[JOINT_LEFT_WRIST].real.x / 1000.0;
          skeleton_data.joint_position_left_wrist_real.z = skeleton.joints[JOINT_LEFT_WRIST].real.y / 1000.0;
        }
        else
        {
          skeleton_data.joint_position_left_wrist_real.x = -9999;
          skeleton_data.joint_position_left_wrist_real.y = -9999;
          skeleton_data.joint_position_left_wrist_real.z = -9999;
        }

        if (skeleton.joints[JOINT_LEFT_HAND].confidence > conf)
        {
          skeleton_data.joint_position_left_hand_real.x = skeleton.joints[JOINT_LEFT_HAND].real.z / 1000.0;
          skeleton_data.joint_position_left_hand_real.y = skeleton.joints[JOINT_LEFT_HAND].real.x / 1000.0;
          skeleton_data.joint_position_left_hand_real.z = skeleton.joints[JOINT_LEFT_HAND].real.y / 1000.0;
        }
        else
        {
          skeleton_data.joint_position_left_hand_real.x = -9999;
          skeleton_data.joint_position_left_hand_real.y = -9999;
          skeleton_data.joint_position_left_hand_real.z = -9999;
        }

        if (skeleton.joints[JOINT_RIGHT_SHOULDER].confidence > conf)
        {
          skeleton_data.joint_position_right_shoulder_real.x = skeleton.joints[JOINT_RIGHT_SHOULDER].real.z / 1000.0;
          skeleton_data.joint_position_right_shoulder_real.y = skeleton.joints[JOINT_RIGHT_SHOULDER].real.x / 1000.0;
          skeleton_data.joint_position_right_shoulder_real.z = skeleton.joints[JOINT_RIGHT_SHOULDER].real.y / 1000.0;
        }
        else
        {
          skeleton_data.joint_position_right_shoulder_real.x = -9999;
          skeleton_data.joint_position_right_shoulder_real.y = -9999;
          skeleton_data.joint_position_right_shoulder_real.z = -9999;
        }

        if (skeleton.joints[JOINT_RIGHT_ELBOW].confidence > conf)
        {
          skeleton_data.joint_position_right_elbow_real.x = skeleton.joints[JOINT_RIGHT_ELBOW].real.z / 1000.0;
          skeleton_data.joint_position_right_elbow_real.y = skeleton.joints[JOINT_RIGHT_ELBOW].real.x / 1000.0;
          skeleton_data.joint_position_right_elbow_real.z = skeleton.joints[JOINT_RIGHT_ELBOW].real.y / 1000.0;
        }
        else
        {
          skeleton_data.joint_position_right_elbow_real.x = -9999;
          skeleton_data.joint_position_right_elbow_real.y = -9999;
          skeleton_data.joint_position_right_elbow_real.z = -9999;
        }

        if (skeleton.joints[JOINT_RIGHT_WRIST].confidence > conf)
        {
          skeleton_data.joint_position_right_wrist_real.x = skeleton.joints[JOINT_RIGHT_WRIST].real.z / 1000.0;
          skeleton_data.joint_position_right_wrist_real.y = skeleton.joints[JOINT_RIGHT_WRIST].real.x / 1000.0;
          skeleton_data.joint_position_right_wrist_real.z = skeleton.joints[JOINT_RIGHT_WRIST].real.y / 1000.0;
        }
        else
        {
          skeleton_data.joint_position_right_wrist_real.x = -9999;
          skeleton_data.joint_position_right_wrist_real.y = -9999;
          skeleton_data.joint_position_right_wrist_real.z = -9999;
        }

        if (skeleton.joints[JOINT_RIGHT_HAND].confidence > conf)
        {
          skeleton_data.joint_position_right_hand_real.x = skeleton.joints[JOINT_RIGHT_HAND].real.z / 1000.0;
          skeleton_data.joint_position_right_hand_real.y = skeleton.joints[JOINT_RIGHT_HAND].real.x / 1000.0;
          skeleton_data.joint_position_right_hand_real.z = skeleton.joints[JOINT_RIGHT_HAND].real.y / 1000.0;
        }
        else
        {
          skeleton_data.joint_position_right_hand_real.x = -9999;
          skeleton_data.joint_position_right_hand_real.y = -9999;
          skeleton_data.joint_position_right_hand_real.z = -9999;
        }

        if (skeleton.joints[JOINT_HEAD].confidence > conf)
        {
          skeleton_data.joint_position_head_proj.x = skeleton.joints[JOINT_HEAD].proj.x;
          skeleton_data.joint_position_head_proj.y = skeleton.joints[JOINT_HEAD].proj.y;
          skeleton_data.joint_position_head_proj.z = skeleton.joints[JOINT_HEAD].proj.z;
        }
        else
        {
          skeleton_data.joint_position_head_proj.x = -9999;
          skeleton_data.joint_position_head_proj.y = -9999;
          skeleton_data.joint_position_head_proj.z = -9999;
        }

        if (skeleton.joints[JOINT_NECK].confidence > conf)
        {
          skeleton_data.joint_position_neck_proj.x = skeleton.joints[JOINT_NECK].proj.x;
          skeleton_data.joint_position_neck_proj.y = skeleton.joints[JOINT_NECK].proj.y;
          skeleton_data.joint_position_neck_proj.z = skeleton.joints[JOINT_NECK].proj.z;
        }
        else
        {
          skeleton_data.joint_position_neck_proj.x = -9999;
          skeleton_data.joint_position_neck_proj.y = -9999;
          skeleton_data.joint_position_neck_proj.z = -9999;
        }

        if (skeleton.joints[JOINT_LEFT_COLLAR].confidence > conf)
        {
          skeleton_data.joint_position_left_collar_proj.x = skeleton.joints[JOINT_LEFT_COLLAR].proj.x;
          skeleton_data.joint_position_left_collar_proj.y = skeleton.joints[JOINT_LEFT_COLLAR].proj.y;
          skeleton_data.joint_position_left_collar_proj.z = skeleton.joints[JOINT_LEFT_COLLAR].proj.z;
        }
        else
        {
          skeleton_data.joint_position_left_collar_proj.x = -9999;
          skeleton_data.joint_position_left_collar_proj.y = -9999;
          skeleton_data.joint_position_left_collar_proj.z = -9999;
        }

        if (skeleton.joints[JOINT_TORSO].confidence > conf)
        {
          skeleton_data.joint_position_torso_proj.x = skeleton.joints[JOINT_TORSO].proj.x;
          skeleton_data.joint_position_torso_proj.y = skeleton.joints[JOINT_TORSO].proj.y;
          skeleton_data.joint_position_torso_proj.z = skeleton.joints[JOINT_TORSO].proj.z;
        }
        else
        {
          skeleton_data.joint_position_torso_proj.x = -9999;
          skeleton_data.joint_position_torso_proj.y = -9999;
          skeleton_data.joint_position_torso_proj.z = -9999;
        }

        if (skeleton.joints[JOINT_WAIST].confidence > conf)
        {
          skeleton_data.joint_position_waist_proj.x = skeleton.joints[JOINT_WAIST].proj.x;
          skeleton_data.joint_position_waist_proj.y = skeleton.joints[JOINT_WAIST].proj.y;
          skeleton_data.joint_position_waist_proj.z = skeleton.joints[JOINT_WAIST].proj.z;
        }
        else
        {
          skeleton_data.joint_position_waist_proj.x = -9999;
          skeleton_data.joint_position_waist_proj.y = -9999;
          skeleton_data.joint_position_waist_proj.z = -9999;
        }

        if (skeleton.joints[JOINT_LEFT_SHOULDER].confidence > conf)
        {
          skeleton_data.joint_position_left_shoulder_proj.x = skeleton.joints[JOINT_LEFT_SHOULDER].proj.x;
          skeleton_data.joint_position_left_shoulder_proj.y = skeleton.joints[JOINT_LEFT_SHOULDER].proj.y;
          skeleton_data.joint_position_left_shoulder_proj.z = skeleton.joints[JOINT_LEFT_SHOULDER].proj.z;
        }
        else
        {
          skeleton_data.joint_position_left_shoulder_proj.x = -9999;
          skeleton_data.joint_position_left_shoulder_proj.y = -9999;
          skeleton_data.joint_position_left_shoulder_proj.z = -9999;
        }

        if (skeleton.joints[JOINT_LEFT_ELBOW].confidence > conf)
        {
          skeleton_data.joint_position_left_elbow_proj.x = skeleton.joints[JOINT_LEFT_ELBOW].proj.x;
          skeleton_data.joint_position_left_elbow_proj.y = skeleton.joints[JOINT_LEFT_ELBOW].proj.y;
          skeleton_data.joint_position_left_elbow_proj.z = skeleton.joints[JOINT_LEFT_ELBOW].proj.z;
        }
        else
        {
          skeleton_data.joint_position_left_elbow_proj.x = -9999;
          skeleton_data.joint_position_left_elbow_proj.y = -9999;
          skeleton_data.joint_position_left_elbow_proj.z = -9999;
        }

        if (skeleton.joints[JOINT_LEFT_WRIST].confidence > conf)
        {
          skeleton_data.joint_position_left_wrist_proj.x = skeleton.joints[JOINT_LEFT_WRIST].proj.x;
          skeleton_data.joint_position_left_wrist_proj.y = skeleton.joints[JOINT_LEFT_WRIST].proj.y;
          skeleton_data.joint_position_left_wrist_proj.z = skeleton.joints[JOINT_LEFT_WRIST].proj.z;
        }
        else
        {
          skeleton_data.joint_position_left_wrist_proj.x = -9999;
          skeleton_data.joint_position_left_wrist_proj.y = -9999;
          skeleton_data.joint_position_left_wrist_proj.z = -9999;
        }

        if (skeleton.joints[JOINT_LEFT_HAND].confidence > conf)
        {
          skeleton_data.joint_position_left_hand_proj.x = skeleton.joints[JOINT_LEFT_HAND].proj.x;
          skeleton_data.joint_position_left_hand_proj.y = skeleton.joints[JOINT_LEFT_HAND].proj.y;
          skeleton_data.joint_position_left_hand_proj.z = skeleton.joints[JOINT_LEFT_HAND].proj.z;
        }
        else
        {
          skeleton_data.joint_position_left_hand_proj.x = -9999;
          skeleton_data.joint_position_left_hand_proj.y = -9999;
          skeleton_data.joint_position_left_hand_proj.z = -9999;
        }

        if (skeleton.joints[JOINT_RIGHT_SHOULDER].confidence > conf)
        {
          skeleton_data.joint_position_right_shoulder_proj.x = skeleton.joints[JOINT_RIGHT_SHOULDER].proj.x;
          skeleton_data.joint_position_right_shoulder_proj.y = skeleton.joints[JOINT_RIGHT_SHOULDER].proj.y;
          skeleton_data.joint_position_right_shoulder_proj.z = skeleton.joints[JOINT_RIGHT_SHOULDER].proj.z;
        }
        else
        {
          skeleton_data.joint_position_right_shoulder_proj.x = -9999;
          skeleton_data.joint_position_right_shoulder_proj.y = -9999;
          skeleton_data.joint_position_right_shoulder_proj.z = -9999;
        }

        if (skeleton.joints[JOINT_RIGHT_ELBOW].confidence > conf)
        {
          skeleton_data.joint_position_right_elbow_proj.x = skeleton.joints[JOINT_RIGHT_ELBOW].proj.x;
          skeleton_data.joint_position_right_elbow_proj.y = skeleton.joints[JOINT_RIGHT_ELBOW].proj.y;
          skeleton_data.joint_position_right_elbow_proj.z = skeleton.joints[JOINT_RIGHT_ELBOW].proj.z;
        }
        else
        {
          skeleton_data.joint_position_right_elbow_proj.x = -9999;
          skeleton_data.joint_position_right_elbow_proj.y = -9999;
          skeleton_data.joint_position_right_elbow_proj.z = -9999;
        }

        if (skeleton.joints[JOINT_RIGHT_WRIST].confidence > conf)
        {
          skeleton_data.joint_position_right_wrist_proj.x = skeleton.joints[JOINT_RIGHT_WRIST].proj.x;
          skeleton_data.joint_position_right_wrist_proj.y = skeleton.joints[JOINT_RIGHT_WRIST].proj.y;
          skeleton_data.joint_position_right_wrist_proj.z = skeleton.joints[JOINT_RIGHT_WRIST].proj.z;
        }
        else
        {
          skeleton_data.joint_position_right_wrist_proj.x = -9999;
          skeleton_data.joint_position_right_wrist_proj.y = -9999;
          skeleton_data.joint_position_right_wrist_proj.z = -9999;
        }

        if (skeleton.joints[JOINT_RIGHT_HAND].confidence > conf)
        {
          skeleton_data.joint_position_right_hand_proj.x = skeleton.joints[JOINT_RIGHT_HAND].proj.x;
          skeleton_data.joint_position_right_hand_proj.y = skeleton.joints[JOINT_RIGHT_HAND].proj.y;
          skeleton_data.joint_position_right_hand_proj.z = skeleton.joints[JOINT_RIGHT_HAND].proj.z;
        }
        else
        {
          skeleton_data.joint_position_right_hand_proj.x = -9999;
          skeleton_data.joint_position_right_hand_proj.y = -9999;
          skeleton_data.joint_position_right_hand_proj.z = -9999;
        }

        if (skeleton.joints[JOINT_HEAD].confidence > conf)
        {
          skeleton_data.joint_orientation1_head.x = skeleton.joints[JOINT_HEAD].orient.matrix[0];
          skeleton_data.joint_orientation1_head.y = skeleton.joints[JOINT_HEAD].orient.matrix[1];
          skeleton_data.joint_orientation1_head.z = skeleton.joints[JOINT_HEAD].orient.matrix[2];
          skeleton_data.joint_orientation2_head.x = skeleton.joints[JOINT_HEAD].orient.matrix[3];
          skeleton_data.joint_orientation2_head.y = skeleton.joints[JOINT_HEAD].orient.matrix[4];
          skeleton_data.joint_orientation2_head.z = skeleton.joints[JOINT_HEAD].orient.matrix[5];
          skeleton_data.joint_orientation3_head.x = skeleton.joints[JOINT_HEAD].orient.matrix[6];
          skeleton_data.joint_orientation3_head.y = skeleton.joints[JOINT_HEAD].orient.matrix[7];
          skeleton_data.joint_orientation3_head.z = skeleton.joints[JOINT_HEAD].orient.matrix[8];
        }
        else
        {
          skeleton_data.joint_orientation1_head.x = -9999;
          skeleton_data.joint_orientation1_head.y = -9999;
          skeleton_data.joint_orientation1_head.z = -9999;
          skeleton_data.joint_orientation2_head.x = -9999;
          skeleton_data.joint_orientation2_head.y = -9999;
          skeleton_data.joint_orientation2_head.z = -9999;
          skeleton_data.joint_orientation3_head.x = -9999;
          skeleton_data.joint_orientation3_head.y = -9999;
          skeleton_data.joint_orientation3_head.z = -9999;
        }

        if (skeleton.joints[JOINT_HEAD].confidence > conf)
        {
          skeleton_data.joint_orientation1_neck.x = skeleton.joints[JOINT_NECK].orient.matrix[0];
          skeleton_data.joint_orientation1_neck.y = skeleton.joints[JOINT_NECK].orient.matrix[1];
          skeleton_data.joint_orientation1_neck.z = skeleton.joints[JOINT_NECK].orient.matrix[2];
          skeleton_data.joint_orientation2_neck.x = skeleton.joints[JOINT_NECK].orient.matrix[3];
          skeleton_data.joint_orientation2_neck.y = skeleton.joints[JOINT_NECK].orient.matrix[4];
          skeleton_data.joint_orientation2_neck.z = skeleton.joints[JOINT_NECK].orient.matrix[5];
          skeleton_data.joint_orientation3_neck.x = skeleton.joints[JOINT_NECK].orient.matrix[6];
          skeleton_data.joint_orientation3_neck.y = skeleton.joints[JOINT_NECK].orient.matrix[7];
          skeleton_data.joint_orientation3_neck.z = skeleton.joints[JOINT_NECK].orient.matrix[8];
        }
        else
        {
          skeleton_data.joint_orientation1_neck.x = -9999;
          skeleton_data.joint_orientation1_neck.y = -9999;
          skeleton_data.joint_orientation1_neck.z = -9999;
          skeleton_data.joint_orientation2_neck.x = -9999;
          skeleton_data.joint_orientation2_neck.y = -9999;
          skeleton_data.joint_orientation2_neck.z = -9999;
          skeleton_data.joint_orientation3_neck.x = -9999;
          skeleton_data.joint_orientation3_neck.y = -9999;
          skeleton_data.joint_orientation3_neck.z = -9999;
        }

        if (skeleton.joints[JOINT_WAIST].confidence > conf)
        {
          skeleton_data.joint_orientation1_waist.x = skeleton.joints[JOINT_WAIST].orient.matrix[0];
          skeleton_data.joint_orientation1_waist.y = skeleton.joints[JOINT_WAIST].orient.matrix[1];
          skeleton_data.joint_orientation1_waist.z = skeleton.joints[JOINT_WAIST].orient.matrix[2];
          skeleton_data.joint_orientation2_waist.x = skeleton.joints[JOINT_WAIST].orient.matrix[3];
          skeleton_data.joint_orientation2_waist.y = skeleton.joints[JOINT_WAIST].orient.matrix[4];
          skeleton_data.joint_orientation2_waist.z = skeleton.joints[JOINT_WAIST].orient.matrix[5];
          skeleton_data.joint_orientation3_waist.x = skeleton.joints[JOINT_WAIST].orient.matrix[6];
          skeleton_data.joint_orientation3_waist.y = skeleton.joints[JOINT_WAIST].orient.matrix[7];
          skeleton_data.joint_orientation3_waist.z = skeleton.joints[JOINT_WAIST].orient.matrix[8];
        }
        else
        {
          skeleton_data.joint_orientation1_waist.x = -9999;
          skeleton_data.joint_orientation1_waist.y = -9999;
          skeleton_data.joint_orientation1_waist.z = -9999;
          skeleton_data.joint_orientation2_waist.x = -9999;
          skeleton_data.joint_orientation2_waist.y = -9999;
          skeleton_data.joint_orientation2_waist.z = -9999;
          skeleton_data.joint_orientation3_waist.x = -9999;
          skeleton_data.joint_orientation3_waist.y = -9999;
          skeleton_data.joint_orientation3_waist.z = -9999;
        }

        if (skeleton.joints[JOINT_LEFT_SHOULDER].confidence > conf)
        {
          skeleton_data.joint_orientation1_left_shoulder.x = skeleton.joints[JOINT_LEFT_SHOULDER].orient.matrix[0];
          skeleton_data.joint_orientation1_left_shoulder.y = skeleton.joints[JOINT_LEFT_SHOULDER].orient.matrix[1];
          skeleton_data.joint_orientation1_left_shoulder.z = skeleton.joints[JOINT_LEFT_SHOULDER].orient.matrix[2];
          skeleton_data.joint_orientation2_left_shoulder.x = skeleton.joints[JOINT_LEFT_SHOULDER].orient.matrix[3];
          skeleton_data.joint_orientation2_left_shoulder.y = skeleton.joints[JOINT_LEFT_SHOULDER].orient.matrix[4];
          skeleton_data.joint_orientation2_left_shoulder.z = skeleton.joints[JOINT_LEFT_SHOULDER].orient.matrix[5];
          skeleton_data.joint_orientation3_left_shoulder.x = skeleton.joints[JOINT_LEFT_SHOULDER].orient.matrix[6];
          skeleton_data.joint_orientation3_left_shoulder.y = skeleton.joints[JOINT_LEFT_SHOULDER].orient.matrix[7];
          skeleton_data.joint_orientation3_left_shoulder.z = skeleton.joints[JOINT_LEFT_SHOULDER].orient.matrix[8];
        }
        else
        {
          skeleton_data.joint_orientation1_left_shoulder.x = -9999;
          skeleton_data.joint_orientation1_left_shoulder.y = -9999;
          skeleton_data.joint_orientation1_left_shoulder.z = -9999;
          skeleton_data.joint_orientation2_left_shoulder.x = -9999;
          skeleton_data.joint_orientation2_left_shoulder.y = -9999;
          skeleton_data.joint_orientation2_left_shoulder.z = -9999;
          skeleton_data.joint_orientation3_left_shoulder.x = -9999;
          skeleton_data.joint_orientation3_left_shoulder.y = -9999;
          skeleton_data.joint_orientation3_left_shoulder.z = -9999;
        }

        if (skeleton.joints[JOINT_LEFT_ELBOW].confidence > conf)
        {
          skeleton_data.joint_orientation1_left_elbow.x = skeleton.joints[JOINT_LEFT_ELBOW].orient.matrix[0];
          skeleton_data.joint_orientation1_left_elbow.y = skeleton.joints[JOINT_LEFT_ELBOW].orient.matrix[1];
          skeleton_data.joint_orientation1_left_elbow.z = skeleton.joints[JOINT_LEFT_ELBOW].orient.matrix[2];
          skeleton_data.joint_orientation2_left_elbow.x = skeleton.joints[JOINT_LEFT_ELBOW].orient.matrix[3];
          skeleton_data.joint_orientation2_left_elbow.y = skeleton.joints[JOINT_LEFT_ELBOW].orient.matrix[4];
          skeleton_data.joint_orientation2_left_elbow.z = skeleton.joints[JOINT_LEFT_ELBOW].orient.matrix[5];
          skeleton_data.joint_orientation3_left_elbow.x = skeleton.joints[JOINT_LEFT_ELBOW].orient.matrix[6];
          skeleton_data.joint_orientation3_left_elbow.y = skeleton.joints[JOINT_LEFT_ELBOW].orient.matrix[7];
          skeleton_data.joint_orientation3_left_elbow.z = skeleton.joints[JOINT_LEFT_ELBOW].orient.matrix[8];
        }
        else
        {
          skeleton_data.joint_orientation1_left_elbow.x = -9999;
          skeleton_data.joint_orientation1_left_elbow.y = -9999;
          skeleton_data.joint_orientation1_left_elbow.z = -9999;
          skeleton_data.joint_orientation2_left_elbow.x = -9999;
          skeleton_data.joint_orientation2_left_elbow.y = -9999;
          skeleton_data.joint_orientation2_left_elbow.z = -9999;
          skeleton_data.joint_orientation3_left_elbow.x = -9999;
          skeleton_data.joint_orientation3_left_elbow.y = -9999;
          skeleton_data.joint_orientation3_left_elbow.z = -9999;
        }

        if (skeleton.joints[JOINT_RIGHT_SHOULDER].confidence > conf)
        {
          skeleton_data.joint_orientation1_right_shoulder.x = skeleton.joints[JOINT_RIGHT_SHOULDER].orient.matrix[0];
          skeleton_data.joint_orientation1_right_shoulder.y = skeleton.joints[JOINT_RIGHT_SHOULDER].orient.matrix[1];
          skeleton_data.joint_orientation1_right_shoulder.z = skeleton.joints[JOINT_RIGHT_SHOULDER].orient.matrix[2];
          skeleton_data.joint_orientation2_right_shoulder.x = skeleton.joints[JOINT_RIGHT_SHOULDER].orient.matrix[3];
          skeleton_data.joint_orientation2_right_shoulder.y = skeleton.joints[JOINT_RIGHT_SHOULDER].orient.matrix[4];
          skeleton_data.joint_orientation2_right_shoulder.z = skeleton.joints[JOINT_RIGHT_SHOULDER].orient.matrix[5];
          skeleton_data.joint_orientation3_right_shoulder.x = skeleton.joints[JOINT_RIGHT_SHOULDER].orient.matrix[6];
          skeleton_data.joint_orientation3_right_shoulder.y = skeleton.joints[JOINT_RIGHT_SHOULDER].orient.matrix[7];
          skeleton_data.joint_orientation3_right_shoulder.z = skeleton.joints[JOINT_RIGHT_SHOULDER].orient.matrix[8];
        }
        else
        {
          skeleton_data.joint_orientation1_right_shoulder.x = -9999;
          skeleton_data.joint_orientation1_right_shoulder.y = -9999;
          skeleton_data.joint_orientation1_right_shoulder.z = -9999;
          skeleton_data.joint_orientation2_right_shoulder.x = -9999;
          skeleton_data.joint_orientation2_right_shoulder.y = -9999;
          skeleton_data.joint_orientation2_right_shoulder.z = -9999;
          skeleton_data.joint_orientation3_right_shoulder.x = -9999;
          skeleton_data.joint_orientation3_right_shoulder.y = -9999;
          skeleton_data.joint_orientation3_right_shoulder.z = -9999;
        }

        if (skeleton.joints[JOINT_RIGHT_ELBOW].confidence > conf)
        {
          skeleton_data.joint_orientation1_right_elbow.x = skeleton.joints[JOINT_RIGHT_ELBOW].orient.matrix[0];
          skeleton_data.joint_orientation1_right_elbow.y = skeleton.joints[JOINT_RIGHT_ELBOW].orient.matrix[1];
          skeleton_data.joint_orientation1_right_elbow.z = skeleton.joints[JOINT_RIGHT_ELBOW].orient.matrix[2];
          skeleton_data.joint_orientation2_right_elbow.x = skeleton.joints[JOINT_RIGHT_ELBOW].orient.matrix[3];
          skeleton_data.joint_orientation2_right_elbow.y = skeleton.joints[JOINT_RIGHT_ELBOW].orient.matrix[4];
          skeleton_data.joint_orientation2_right_elbow.z = skeleton.joints[JOINT_RIGHT_ELBOW].orient.matrix[5];
          skeleton_data.joint_orientation3_right_elbow.x = skeleton.joints[JOINT_RIGHT_ELBOW].orient.matrix[6];
          skeleton_data.joint_orientation3_right_elbow.y = skeleton.joints[JOINT_RIGHT_ELBOW].orient.matrix[7];
          skeleton_data.joint_orientation3_right_elbow.z = skeleton.joints[JOINT_RIGHT_ELBOW].orient.matrix[8];
        }
        else
        {
          skeleton_data.joint_orientation1_right_elbow.x = -9999;
          skeleton_data.joint_orientation1_right_elbow.y = -9999;
          skeleton_data.joint_orientation1_right_elbow.z = -9999;
          skeleton_data.joint_orientation2_right_elbow.x = -9999;
          skeleton_data.joint_orientation2_right_elbow.y = -9999;
          skeleton_data.joint_orientation2_right_elbow.z = -9999;
          skeleton_data.joint_orientation3_right_elbow.x = -9999;
          skeleton_data.joint_orientation3_right_elbow.y = -9999;
          skeleton_data.joint_orientation3_right_elbow.z = -9999;
        }

        skeleton_data.gesture = -1; // No gesture
        person_data.gesture = -1;
        for (int i = 0; i < userGestures_.size(); ++i)
        {
          if ((userGestures_[i].userId == skeleton.id) && // match the person being reported
              (userGestures_[i].type > -1))               // Not already reported
          {
            skeleton_data.gesture = userGestures_[i].type; // TODO - map Nuitrack to my MSG enum
            person_data.gesture = userGestures_[i].type;   // TODO - map Nuitrack to my MSG enum
            printf("Reporting Gesture %d for User %d\n",
                   userGestures_[i].type, userGestures_[i].userId);
            userGestures_[i].type = (tdv::nuitrack::GestureType)(-1); // clear so we don't report old gestures
          }
        }

        ////////////////////////////////////////////////////
        // Publish custom position and skeleton messages for each person found

        body_tracking_position_pub_.publish(person_data);   // position data
        body_tracking_skeleton_pub_.publish(skeleton_data); // full skeleton data

        // Msg with array of position data for each person detected
        // body_tracker_array_msg.detected_list.push_back(person_data);

        // Publish skeleton markers

        PublishMarker( // show marker at KEY_JOINT_TO_TRACK location
            1,         // ID
            person_data.position3d.x,
            person_data.position3d.y,
            person_data.position3d.z,
            1.0, 0.0, 0.0); // r,g,b

        PublishMarker(
            3, // ID
            skeleton_data.joint_position_head_real.x,
            skeleton_data.joint_position_head_real.y,
            skeleton_data.joint_position_head_real.z,
            0.7, 0.0, 0.7); // r,g,b

        PublishMarker(
            4, // ID
            skeleton_data.joint_position_left_collar_real.x,
            skeleton_data.joint_position_left_collar_real.y,
            skeleton_data.joint_position_left_collar_real.z,
            0.0, 0.0, 1.0); // r,g,b

        PublishMarker(
            5, // ID
            skeleton_data.joint_position_torso_real.x,
            skeleton_data.joint_position_torso_real.y,
            skeleton_data.joint_position_torso_real.z,
            0.0, 1.0, 0.0); // r,g,b
      }

      ////////////////////////////////////////////////////
    }

    void PublishMarker(int id, float x, float y, float z,
                       float color_r, float color_g, float color_b)
    {

      visualization_msgs::Marker marker;
      marker.header.frame_id = camera_depth_frame_;
      marker.header.stamp = ros::Time::now();
      marker.lifetime = ros::Duration(3.0); // seconds
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

    void publishJoint2D(const char *name, const tdv::nuitrack::Joint &joint)
    {
      const float ASTRA_MINI_FOV_X = 1.047200;  // (60 degrees horizontal)
      const float ASTRA_MINI_FOV_Y = -0.863938; // (49.5 degrees vertical)
      if (joint.confidence < 0.15)
      {
        return; // ignore low confidence joints
      }

      float radians_x = (joint.proj.x - 0.5) * ASTRA_MINI_FOV_X;
      float radians_y = (joint.proj.y - 0.5) * ASTRA_MINI_FOV_Y;
      std::cout << std::setprecision(4) << std::setw(7)
                << "Nuitrack: " << name
                << " x: " << joint.proj.x << " (" << radians_x << ")  y: "
                << joint.proj.y << " (" << radians_y << ")"
                // << "  Confidence: " << joint.confidence
                << std::endl;
    }

    ///////////////////////////////////////////////////////////////////////////
    void Init(const std::string &config)
    {
      // Initialize Nuitrack first, then create Nuitrack modules
      ROS_INFO("%s: Initializing...", _name.c_str());
      // std::cout << "Nuitrack: Initializing..." << std::endl;

      std::cout << "\n============ IGNORE ERRORS THAT SAY 'Couldnt open device...' ===========\n"
                << std::endl;

      try
      {
        //tdv::nuitrack::Nuitrack::init("");
        tdv::nuitrack::Nuitrack::init(config);
      }
      catch (const tdv::nuitrack::Exception &e)
      {
        std::cerr << "Cannot initialize Nuitrack (ExceptionType: " << e.type() << ")" << std::endl;
        exit(EXIT_FAILURE);
      }

      // Set config values.  Overrides $NUITRACK_HOME/data/nuitrack.config

      // Align depth and color
      Nuitrack::setConfigValue("DepthProvider.Depth2ColorRegistration", "true");

      // Realsense Depth Module - force to 848x480 @ 60 FPS
      Nuitrack::setConfigValue("Realsense2Module.Depth.Preset", "5");
      Nuitrack::setConfigValue("Realsense2Module.Depth.RawWidth", "848");
      Nuitrack::setConfigValue("Realsense2Module.Depth.RawHeight", "480");
      Nuitrack::setConfigValue("Realsense2Module.Depth.ProcessWidth", "848");
      Nuitrack::setConfigValue("Realsense2Module.Depth.ProcessHeight", "480");
      Nuitrack::setConfigValue("Realsense2Module.Depth.FPS", "60");

      // Realsense RGB Module - force to 848x480 @ 60 FPS
      Nuitrack::setConfigValue("Realsense2Module.RGB.RawWidth", "848");
      Nuitrack::setConfigValue("Realsense2Module.RGB.RawHeight", "480");
      Nuitrack::setConfigValue("Realsense2Module.RGB.ProcessWidth", "848");
      Nuitrack::setConfigValue("Realsense2Module.RGB.ProcessHeight", "480");
      Nuitrack::setConfigValue("Realsense2Module.RGB.FPS", "60");

      // Enable face tracking()
      Nuitrack::setConfigValue("Faces.ToUse", "false");

      depth_frame_number_ = 0;
      color_frame_number_ = 0;

      // Create all required Nuitrack modules

      std::cout << "Nuitrack: DepthSensor::create()" << std::endl;
      depthSensor_ = tdv::nuitrack::DepthSensor::create();
      // Bind to event new frame
      depthSensor_->connectOnNewFrame(std::bind(
          &nuitrack_body_tracker_node::onNewDepthFrame, this, std::placeholders::_1));

      std::cout << "Nuitrack: ColorSensor::create()" << std::endl;
      colorSensor_ = tdv::nuitrack::ColorSensor::create();
      // Bind to event new frame
      colorSensor_->connectOnNewFrame(std::bind(
          &nuitrack_body_tracker_node::onNewColorFrame, this, std::placeholders::_1));

      outputMode_ = depthSensor_->getOutputMode();
      OutputMode colorOutputMode = colorSensor_->getOutputMode();
      if ((colorOutputMode.xres != outputMode_.xres) || (colorOutputMode.yres != outputMode_.yres))
      {
        ROS_WARN("%s: WARNING! DEPTH AND COLOR SIZE NOT THE SAME!", _name.c_str());
      }

      // Use depth as the frame size
      frame_width_ = outputMode_.xres;
      frame_height_ = outputMode_.yres;
      last_id_ = -1;
      std::cout << "========= Nuitrack: GOT DEPTH SENSOR =========" << std::endl;
      std::cout << "Nuitrack: Depth:  width = " << frame_width_ << "  height = " << frame_height_ << std::endl;

      // Point Cloud message (includes depth and color)
      int numpoints = frame_width_ * frame_height_;
      cloud_msg_.header.frame_id = camera_depth_frame_;
      //cloud_msg_.header.stamp = ros::Time::now();
      cloud_msg_.width = numpoints;
      cloud_msg_.height = 1;
      cloud_msg_.is_bigendian = false;
      cloud_msg_.is_dense = false; // there may be invalid points

      sensor_msgs::PointCloud2Modifier modifier(cloud_msg_);
      modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
      modifier.resize(numpoints);

      std::cout << "Nuitrack: SkeletonTracker::create()" << std::endl;
      skeletonTracker_ = tdv::nuitrack::SkeletonTracker::create();
      // Bind to event update skeleton tracker
      skeletonTracker_->connectOnUpdate(std::bind(
          &nuitrack_body_tracker_node::onSkeletonUpdate, this, std::placeholders::_1));

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
      catch (const tdv::nuitrack::Exception &e)
      {
        std::cerr << "Can not start Nuitrack (ExceptionType: "
                  << e.type() << ")" << std::endl;
        return;
      }

      ROS_INFO("%s: Waiting for person to be detected...", _name.c_str());

      // Run Loop
      ros::Rate r(30); // hz

      //while(true){r.sleep();}
      //while文が入るとメモリが増える？

      while (ros::ok())
      {
        // std::cout << "Nuitrack: Looping..." << std::endl;

        try
        {
          // Wait for new person tracking data
          tdv::nuitrack::Nuitrack::waitUpdate(skeletonTracker_);
        }
        catch (tdv::nuitrack::LicenseNotAcquiredException &e)
        {
          std::cerr << "LicenseNotAcquired exception (ExceptionType: "
                    << e.type() << ")" << std::endl;
          break;
        }
        catch (const tdv::nuitrack::Exception &e)
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
        std::cout << "========= Nuitrack: RELEASE COMPLETED =========" << std::endl;

      }
      catch (const tdv::nuitrack::Exception &e)
      {
        std::cerr << "Nuitrack release failed (ExceptionType: "
                  << e.type() << ")" << std::endl;
      }

    } // Run()

    //Add mechaless command subscriber
    void PygameCallBack(const sensor_msgs::JointState &_pygame)
    {
      int shutdown = _pygame.position[8];
      if (shutdown == 0)
      {
        printf("Shutdown");
        ros::shutdown();
      }
    }

  private:
    /////////////// DATA MEMBERS /////////////////////

    std::string _name;
    ros::NodeHandle nh_;
    std::string camera_depth_frame_;
    std::string camera_color_frame_;
    int frame_width_, frame_height_;
    int last_id_;
    sensor_msgs::PointCloud2 cloud_msg_; // color and depth point cloud
    int depth_frame_number_;
    int color_frame_number_;

    ros::Publisher body_tracking_position_pub_;
    ros::Publisher body_tracking_array_pub_;
    ros::Publisher body_tracking_skeleton_pub_;
    ros::Publisher marker_pub_;
    ros::Publisher depth_image_pub_;
    ros::Publisher color_image_pub_;
    ros::Publisher depth_cloud_pub_;

    ros::Subscriber mechaless_pygame_state_sub_;

    //ros::Publisher body_tracking_pose2d_pub_;
    //ros::Publisher body_tracking_pose3d_pub_;
    //ros::Publisher body_tracking_gesture_pub_;
    //ros::Subscriber servo_pan_sub_;

    tdv::nuitrack::OutputMode outputMode_;
    std::vector<tdv::nuitrack::Gesture> userGestures_;
    tdv::nuitrack::DepthSensor::Ptr depthSensor_;
    tdv::nuitrack::ColorSensor::Ptr colorSensor_;
    tdv::nuitrack::UserTracker::Ptr userTracker_;
    tdv::nuitrack::SkeletonTracker::Ptr skeletonTracker_;
    tdv::nuitrack::HandTracker::Ptr handTracker_;
    tdv::nuitrack::GestureRecognizer::Ptr gestureRecognizer_;
    //tdv::nuitrack::getInstancesJson::Ptr getInstancesJson;
    float confidence_value;
    int waist_set = 0;
    std::vector<double> pre_waist_position;
    std::vector<double> tmp_waist_position;

    /* Note from http://download.3divi.com/Nuitrack/doc/Instance_based_API.html
    Face modules are by default disabled. To enable face modules, open nuitrack.config file and set Faces.ToUse and DepthProvider.Depth2ColorRegistration to true.
    */
  };
}; // namespace nuitrack_body_tracker

// The main entry point for this node.
int main(int argc, char *argv[])
{
  using namespace nuitrack_body_tracker;
  ros::init(argc, argv, "nuitrack_body_tracker");
  nuitrack_body_tracker_node node(ros::this_node::getName());
  node.Init("");
  node.Run();

  return 0;
}

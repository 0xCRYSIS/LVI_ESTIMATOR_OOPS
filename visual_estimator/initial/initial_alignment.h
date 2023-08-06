#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <rclcpp/rclcpp.hpp>
#include <map>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame
{
    public:
        ImageFrame(){};
        ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>>& _points, 
                   const vector<float> &_lidar_initialization_info,
                   double _t):
        t{_t}, is_key_frame{false}, reset_id{-1}, gravity{9.805}
        {
            points = _points;
            
            // reset id in case lidar odometry relocate
            reset_id = (int)round(_lidar_initialization_info[0]);
            // Pose
            T.x() = _lidar_initialization_info[1];
            T.y() = _lidar_initialization_info[2];
            T.z() = _lidar_initialization_info[3];
            // Rotation
            Eigen::Quaterniond Q = Eigen::Quaterniond(_lidar_initialization_info[7],
                                                      _lidar_initialization_info[4],
                                                      _lidar_initialization_info[5],
                                                      _lidar_initialization_info[6]);
            R = Q.normalized().toRotationMatrix();
            // Velocity
            V.x() = _lidar_initialization_info[8];
            V.y() = _lidar_initialization_info[9];
            V.z() = _lidar_initialization_info[10];
            // Acceleration bias
            Ba.x() = _lidar_initialization_info[11];
            Ba.y() = _lidar_initialization_info[12];
            Ba.z() = _lidar_initialization_info[13];
            // Gyroscope bias
            Bg.x() = _lidar_initialization_info[14];
            Bg.y() = _lidar_initialization_info[15];
            Bg.z() = _lidar_initialization_info[16];
            // Gravity
            gravity = _lidar_initialization_info[17];
        };

        map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>> > > points;
        double t;
        
        IntegrationBase *pre_integration;
        bool is_key_frame;

        // Lidar odometry info
        int reset_id;
        Vector3d T;
        Matrix3d R;
        Vector3d V;
        Vector3d Ba;
        Vector3d Bg;
        double gravity;
};


bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);

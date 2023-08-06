#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <functional>
#include <memory>
#include <condition_variable>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/CameraPoseVisualization.h"

using namespace std::chrono_literals;


class EstimatorNode : public rclcpp::Node
{
    public:

        // from odom register
        tf2::Quaternion q_lidar_to_cam;
        Eigen::Quaterniond q_lidar_to_cam_eigen;

        // rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_latest_odometry;

        // from estimator node
        Estimator estimator;

        std::condition_variable con;
        double current_time = -1;
        queue<sensor_msgs::msg::Imu::ConstPtr> imu_buf;
        queue<sensor_msgs::msg::PointCloud::ConstPtr> feature_buf;

        // global variable saving the lidar odometry
        deque<nav_msgs::msg::Odometry> odomQueue;
        
        std::mutex m_buf;
        std::mutex m_state;
        std::mutex m_estimator;
        std::mutex m_odom;

        double latest_time;
        Eigen::Vector3d tmp_P;
        Eigen::Quaterniond tmp_Q;
        Eigen::Vector3d tmp_V;
        Eigen::Vector3d tmp_Ba;
        Eigen::Vector3d tmp_Bg;
        Eigen::Vector3d acc_0;
        Eigen::Vector3d gyr_0;
        bool init_feature = 0;
        bool init_imu = 1;
        double last_imu_t = 0;

        // form visualization.cpp
        nav_msgs::msg::Path path;
        CameraPoseVisualization cameraposevisual = CameraPoseVisualization(0,1,0,1);
        CameraPoseVisualization keyframebasevisual= CameraPoseVisualization(0.0, 0.0, 1.0, 1.0);
        double sum_of_path = 0;
        Eigen::Vector3d last_path = Eigen::Vector3d(0.0, 0.0, 0.0);
        std::unique_ptr<tf2_ros::TransformBroadcaster> br;
        std::shared_ptr<tf2_ros::TransformListener> listener{nullptr};
        std::unique_ptr<tf2_ros::Buffer> tf_buffer; 

        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_latest_odometry;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_latest_odometry_ros;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_point_cloud;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_margin_cloud;
        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_key_poses;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_camera_pose;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_keyframe_pose;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_keyframe_point;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_extrinsic;


        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr sub_image;
        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_restart;

        EstimatorNode() : Node("EstimatorNode")
        {

            readParameters();

            // from OdomRegister
            q_lidar_to_cam = tf2::Quaternion(0,1,0,0); // rotate orientation // mark: camera - lidar
            q_lidar_to_cam_eigen = Eigen::Quaterniond(0, 0, 0, 1); // rotate position by pi, (w, x, y, z) // mark: camera - lidar
            // pub_latest_odometry = this->create_publisher<nav_msgs::msg::Odometry>("odometry/test", 1000);

            // from visualization.cpp
            pub_latest_odometry     = this->create_publisher<nav_msgs::msg::Odometry>(PROJECT_NAME + "/vins/odometry/imu_propagate", 1000);
            pub_latest_odometry_ros = this->create_publisher<nav_msgs::msg::Odometry>(PROJECT_NAME + "/vins/odometry/imu_propagate_ros", 1000);
            pub_path                = this->create_publisher<nav_msgs::msg::Path>(PROJECT_NAME + "/vins/odometry/path", 1000);
            pub_odometry            = this->create_publisher<nav_msgs::msg::Odometry>(PROJECT_NAME + "/vins/odometry/odometry", 1000);
            pub_point_cloud         = this->create_publisher<sensor_msgs::msg::PointCloud>(PROJECT_NAME + "/vins/odometry/point_cloud", 1000);
            pub_margin_cloud        = this->create_publisher<sensor_msgs::msg::PointCloud>(PROJECT_NAME + "/vins/odometry/history_cloud", 1000);
            pub_key_poses           = this->create_publisher<visualization_msgs::msg::Marker>(PROJECT_NAME + "/vins/odometry/key_poses", 1000);
            pub_camera_pose         = this->create_publisher<nav_msgs::msg::Odometry>(PROJECT_NAME + "/vins/odometry/camera_pose", 1000);
            pub_camera_pose_visual  = this->create_publisher<visualization_msgs::msg::MarkerArray>(PROJECT_NAME + "/vins/odometry/camera_pose_visual", 1000);
            pub_keyframe_pose       = this->create_publisher<nav_msgs::msg::Odometry>(PROJECT_NAME + "/vins/odometry/keyframe_pose", 1000);
            pub_keyframe_point      = this->create_publisher<sensor_msgs::msg::PointCloud>(PROJECT_NAME + "/vins/odometry/keyframe_point", 1000);
            pub_extrinsic           = this->create_publisher<nav_msgs::msg::Odometry>(PROJECT_NAME + "/vins/odometry/extrinsic", 1000);

            br = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
            tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
            listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

            // from estimator node

            estimator.setParameter();

            sub_imu = this->create_subscription<sensor_msgs::msg::Imu>(IMU_TOPIC,100,std::bind(&EstimatorNode::imu_callback,this,std::placeholders::_1));
            sub_odom = this->create_subscription<nav_msgs::msg::Odometry>("odometry/imu",100,std::bind(&EstimatorNode::odom_callback,this,std::placeholders::_1));
            sub_image = this->create_subscription<sensor_msgs::msg::PointCloud>(PROJECT_NAME + "/vins/feature/feature",100,std::bind(&EstimatorNode::feature_callback,this,std::placeholders::_1));
            sub_restart = this->create_subscription<std_msgs::msg::Bool>(PROJECT_NAME + "/vins/feature/restart",100,std::bind(&EstimatorNode::restart_callback,this,std::placeholders::_1));
        }

        // from OdomRegister
        // convert odometry from ROS Lidar frame to VINS camera frame
        vector<float> getOdometry(deque<nav_msgs::msg::Odometry>& odomQueue, double img_time)
        {
            vector<float> odometry_channel;
            odometry_channel.resize(18, -1); // reset id(1), P(3), Q(4), V(3), Ba(3), Bg(3), gravity(1)

            nav_msgs::msg::Odometry odomCur;
            
            // pop old odometry msg
            while (!odomQueue.empty()) 
            {
                if (rclcpp::Time(odomQueue.front().header.stamp).seconds() < img_time - 0.05)
                    odomQueue.pop_front();
                else
                    break;
            }

            if (odomQueue.empty())
            {
                return odometry_channel;
            }

            // find the odometry time that is the closest to image time
            for (int i = 0; i < (int)odomQueue.size(); ++i)
            {
                odomCur = odomQueue[i];

                if (rclcpp::Time(odomCur.header.stamp).seconds() < img_time - 0.002) // 500Hz imu
                    continue;
                else
                    break;
            }

            // time stamp difference still too large
            if (abs(rclcpp::Time(odomCur.header.stamp).seconds() - img_time) > 0.05)
            {
                return odometry_channel;
            }

            // convert odometry rotation from lidar ROS frame to VINS camera frame (only rotation, assume lidar, camera, and IMU are close enough)
            tf2::Quaternion q_odom_lidar(odomCur.pose.pose.orientation.x,
                                        odomCur.pose.pose.orientation.y,
                                        odomCur.pose.pose.orientation.z,
                                        odomCur.pose.pose.orientation.w);
            // tf::quaternionMsgToTF(odomCur.pose.pose.orientation, q_odom_lidar);
            // tf2::fromMsg(odomCur,q_odom_lidar);
            
            

            // tf::Quaternion q_odom_cam = tf::createQuaternionFromRPY(0, 0, M_PI) * (q_odom_lidar * q_lidar_to_cam); // global rotate by pi // mark: camera - lidar
            tf2::Quaternion q;
            q.setRPY(0,0,M_PI);
            tf2::Quaternion q_odom_cam = q * (q_odom_lidar * q_lidar_to_cam);
            // tf::quaternionTFToMsg(q_odom_cam, odomCur.pose.pose.orientation);
            odomCur.pose.pose.orientation.x = q_odom_cam.x();
            odomCur.pose.pose.orientation.y = q_odom_cam.y();
            odomCur.pose.pose.orientation.z = q_odom_cam.z();
            odomCur.pose.pose.orientation.w = q_odom_cam.w();

            // convert odometry position from lidar ROS frame to VINS camera frame
            Eigen::Vector3d p_eigen(odomCur.pose.pose.position.x, odomCur.pose.pose.position.y, odomCur.pose.pose.position.z);
            Eigen::Vector3d v_eigen(odomCur.twist.twist.linear.x, odomCur.twist.twist.linear.y, odomCur.twist.twist.linear.z);
            Eigen::Vector3d p_eigen_new = q_lidar_to_cam_eigen * p_eigen;
            Eigen::Vector3d v_eigen_new = q_lidar_to_cam_eigen * v_eigen;

            odomCur.pose.pose.position.x = p_eigen_new.x();
            odomCur.pose.pose.position.y = p_eigen_new.y();
            odomCur.pose.pose.position.z = p_eigen_new.z();

            odomCur.twist.twist.linear.x = v_eigen_new.x();
            odomCur.twist.twist.linear.y = v_eigen_new.y();
            odomCur.twist.twist.linear.z = v_eigen_new.z();

            // odomCur.header.stamp = ros::Time().fromSec(img_time);
            // odomCur.header.frame_id = "vins_world";
            // odomCur.child_frame_id = "vins_body";
            // pub_latest_odometry.publish(odomCur);

            odometry_channel[0] = odomCur.pose.covariance[0];
            odometry_channel[1] = odomCur.pose.pose.position.x;
            odometry_channel[2] = odomCur.pose.pose.position.y;
            odometry_channel[3] = odomCur.pose.pose.position.z;
            odometry_channel[4] = odomCur.pose.pose.orientation.x;
            odometry_channel[5] = odomCur.pose.pose.orientation.y;
            odometry_channel[6] = odomCur.pose.pose.orientation.z;
            odometry_channel[7] = odomCur.pose.pose.orientation.w;
            odometry_channel[8]  = odomCur.twist.twist.linear.x;
            odometry_channel[9]  = odomCur.twist.twist.linear.y;
            odometry_channel[10] = odomCur.twist.twist.linear.z;
            odometry_channel[11] = odomCur.pose.covariance[1];
            odometry_channel[12] = odomCur.pose.covariance[2];
            odometry_channel[13] = odomCur.pose.covariance[3];
            odometry_channel[14] = odomCur.pose.covariance[4];
            odometry_channel[15] = odomCur.pose.covariance[5];
            odometry_channel[16] = odomCur.pose.covariance[6];
            odometry_channel[17] = odomCur.pose.covariance[7];

            return odometry_channel;
        }

        // from estimator node
        void predict(const sensor_msgs::msg::Imu::ConstPtr imu_msg)
        {
            double t = rclcpp::Time(imu_msg->header.stamp).seconds();
            if (init_imu)
            {
                latest_time = t;
                init_imu = 0;
                return;
            }
            double dt = t - latest_time;
            latest_time = t;

            double dx = imu_msg->linear_acceleration.x;
            double dy = imu_msg->linear_acceleration.y;
            double dz = imu_msg->linear_acceleration.z;
            Eigen::Vector3d linear_acceleration{dx, dy, dz};

            double rx = imu_msg->angular_velocity.x;
            double ry = imu_msg->angular_velocity.y;
            double rz = imu_msg->angular_velocity.z;
            Eigen::Vector3d angular_velocity{rx, ry, rz};

            Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

            Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
            tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

            Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

            Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

            tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
            tmp_V = tmp_V + dt * un_acc;

            acc_0 = linear_acceleration;
            gyr_0 = angular_velocity;
        }

        void update()
        {
            TicToc t_predict;
            latest_time = current_time;
            tmp_P = estimator.Ps[WINDOW_SIZE];
            tmp_Q = estimator.Rs[WINDOW_SIZE];
            tmp_V = estimator.Vs[WINDOW_SIZE];
            tmp_Ba = estimator.Bas[WINDOW_SIZE];
            tmp_Bg = estimator.Bgs[WINDOW_SIZE];
            acc_0 = estimator.acc_0;
            gyr_0 = estimator.gyr_0;

            queue<sensor_msgs::msg::Imu::ConstPtr> tmp_imu_buf = imu_buf;
            for (sensor_msgs::msg::Imu::ConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
                predict(tmp_imu_buf.front());
        }

        std::vector<std::pair<std::vector<sensor_msgs::msg::Imu::ConstPtr>, sensor_msgs::msg::PointCloud::ConstPtr>>
        getMeasurements()
        {
            std::vector<std::pair<std::vector<sensor_msgs::msg::Imu::ConstPtr>, sensor_msgs::msg::PointCloud::ConstPtr>> measurements;

            while (rclcpp::ok())
            {
                if (imu_buf.empty() || feature_buf.empty())
                    return measurements;

                if (!(rclcpp::Time(imu_buf.back()->header.stamp).seconds() > rclcpp::Time(feature_buf.front()->header.stamp).seconds() + estimator.td))
                {
                    return measurements;
                }

                if (!(rclcpp::Time(imu_buf.front()->header.stamp).seconds() < rclcpp::Time(feature_buf.front()->header.stamp).seconds() + estimator.td))
                {
                    RCLCPP_WARN(rclcpp::get_logger("estimator_node"),"throw img, only should happen at the beginning");
                    feature_buf.pop();
                    continue;
                }
                sensor_msgs::msg::PointCloud::ConstPtr img_msg = feature_buf.front();
                feature_buf.pop();

                std::vector<sensor_msgs::msg::Imu::ConstPtr> IMUs;
                while (rclcpp::Time(imu_buf.front()->header.stamp).seconds() < rclcpp::Time(img_msg->header.stamp).seconds() + estimator.td)
                {
                    IMUs.emplace_back(imu_buf.front());
                    imu_buf.pop();
                }
                IMUs.emplace_back(imu_buf.front());
                if (IMUs.empty())
                    RCLCPP_WARN(rclcpp::get_logger("estimator_node"),"no imu between two image");
                measurements.emplace_back(IMUs, img_msg);
            }
            return measurements;
        }

        void imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu_msg)
        {
            if (rclcpp::Time(imu_msg->header.stamp).seconds() <= last_imu_t)
            {
                RCLCPP_WARN(rclcpp::get_logger("estimator_node"),"imu message in disorder!");
                return;
            }
            last_imu_t =rclcpp::Time(imu_msg->header.stamp).seconds();

            m_buf.lock();
            imu_buf.push(imu_msg);
            m_buf.unlock();
            con.notify_one();

            last_imu_t = rclcpp::Time(imu_msg->header.stamp).seconds();

            
            std::lock_guard<std::mutex> lg(m_state);
            predict(imu_msg);
            std_msgs::msg::Header header = imu_msg->header;
            if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
                pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header, estimator.failureCount);

        }

        void odom_callback(const nav_msgs::msg::Odometry::SharedPtr odom_msg)
        {
            m_odom.lock();
            odomQueue.push_back(*odom_msg);
            m_odom.unlock();
        }

        void feature_callback(const sensor_msgs::msg::PointCloud::SharedPtr feature_msg)
        {
            if (!init_feature)
            {
                //skip the first detected feature, which doesn't contain optical flow speed
                init_feature = 1;
                return;
            }
            m_buf.lock();
            feature_buf.push(feature_msg);
            m_buf.unlock();
            con.notify_one();
        }

        void restart_callback(const std_msgs::msg::Bool::SharedPtr restart_msg)
        {
            if (restart_msg->data == true)
            {
                RCLCPP_WARN(rclcpp::get_logger("estimator_node"),"restart the estimator!");
                m_buf.lock();
                while(!feature_buf.empty())
                    feature_buf.pop();
                while(!imu_buf.empty())
                    imu_buf.pop();
                m_buf.unlock();
                m_estimator.lock();
                estimator.clearState();
                estimator.setParameter();
                m_estimator.unlock();
                current_time = -1;
                last_imu_t = 0;
            }
            return;
        }

        // thread: visual-inertial odometry
        void process()
        {
            while (rclcpp::ok())
            {
                std::vector<std::pair<std::vector<sensor_msgs::msg::Imu::ConstPtr>, sensor_msgs::msg::PointCloud::ConstPtr>> measurements;
                std::unique_lock<std::mutex> lk(m_buf);
                RCLCPP_INFO(rclcpp::get_logger("process"),"---- BEFORE con.wait ------");
                con.wait(lk, [&]
                        {
                    return (measurements = getMeasurements()).size() != 0;
                        });
                lk.unlock();

                m_estimator.lock();
                for (auto &measurement : measurements)
                {
                    auto img_msg = measurement.second;

                    // 1. IMU pre-integration
                    double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
                    for (auto &imu_msg : measurement.first)
                    {
                        double t = rclcpp::Time(imu_msg->header.stamp).seconds();
                        double img_t = rclcpp::Time(img_msg->header.stamp).seconds() + estimator.td;
                        if (t <= img_t)
                        { 
                            if (current_time < 0)
                                current_time = t;
                            double dt = t - current_time;
                            assert(dt >= 0);
                            current_time = t;
                            dx = imu_msg->linear_acceleration.x;
                            dy = imu_msg->linear_acceleration.y;
                            dz = imu_msg->linear_acceleration.z;
                            rx = imu_msg->angular_velocity.x;
                            ry = imu_msg->angular_velocity.y;
                            rz = imu_msg->angular_velocity.z;
                            estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                            //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
                        }
                        else
                        {
                            double dt_1 = img_t - current_time;
                            double dt_2 = t - img_t;
                            current_time = img_t;
                            assert(dt_1 >= 0);
                            assert(dt_2 >= 0);
                            assert(dt_1 + dt_2 > 0);
                            double w1 = dt_2 / (dt_1 + dt_2);
                            double w2 = dt_1 / (dt_1 + dt_2);
                            dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                            dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                            dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                            rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                            ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                            rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                            estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                            //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                        }
                    }

                    // 2. VINS Optimization
                    // TicToc t_s;
                    map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> image;
                    for (unsigned int i = 0; i < img_msg->points.size(); i++)
                    {
                        int v = img_msg->channels[0].values[i] + 0.5;
                        int feature_id = v / NUM_OF_CAM;
                        int camera_id = v % NUM_OF_CAM;
                        double x = img_msg->points[i].x;
                        double y = img_msg->points[i].y;
                        double z = img_msg->points[i].z;
                        double p_u = img_msg->channels[1].values[i];
                        double p_v = img_msg->channels[2].values[i];
                        double velocity_x = img_msg->channels[3].values[i];
                        double velocity_y = img_msg->channels[4].values[i];
                        double depth = img_msg->channels[5].values[i];

                        assert(z == 1);
                        Eigen::Matrix<double, 8, 1> xyz_uv_velocity_depth;
                        xyz_uv_velocity_depth << x, y, z, p_u, p_v, velocity_x, velocity_y, depth;
                        image[feature_id].emplace_back(camera_id,  xyz_uv_velocity_depth);
                    }

                    // Get initialization info from lidar odometry
                    vector<float> initialization_info;
                    m_odom.lock();
                    initialization_info = getOdometry(odomQueue, rclcpp::Time(img_msg->header.stamp).seconds() + estimator.td);
                    m_odom.unlock();


                    estimator.processImage(image, initialization_info, img_msg->header);
                    // double whole_t = t_s.toc();
                    // printStatistics(estimator, whole_t);

                    // 3. Visualization
                    std_msgs::msg::Header header = img_msg->header;
                    
                    pubOdometry(estimator, header);
                    pubKeyPoses(estimator, header);
                    pubCameraPose(estimator, header);
                    pubPointCloud(estimator, header);
                    pubTF(estimator, header);
                    pubKeyframe(estimator);
                }
                m_estimator.unlock();

                m_buf.lock();
                m_state.lock();
                if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
                    update();
                m_state.unlock();
                m_buf.unlock();
            }
        }

        // from visualization.cpp
        tf2::Transform transform_conversion(const geometry_msgs::msg::TransformStamped &t)
        {
            double xCur,yCur,zCur,rollCur,pitchCur,yawCur;
            xCur = t.transform.translation.x;
            yCur = t.transform.translation.y;
            zCur = t.transform.translation.y;

            tf2::Quaternion q(t.transform.rotation.x,
                            t.transform.rotation.y,
                            t.transform.rotation.z,
                            t.transform.rotation.w);
            
            tf2::Matrix3x3 m(q);
            m.getRPY(rollCur,pitchCur,yawCur);

            return tf2::Transform(q,tf2::Vector3(xCur,yCur,zCur));
        }

        geometry_msgs::msg::TransformStamped transformTogeometry(const tf2::Transform& transform , const std_msgs::msg::Header &header,std::string frame_id,std::string child_frame_id)
        {
            geometry_msgs::msg::TransformStamped transformstamped;

            transformstamped.header.stamp = header.stamp;
            transformstamped.header.frame_id = frame_id;
            transformstamped.child_frame_id = child_frame_id;
            transformstamped.transform.translation.x = transform.getOrigin().x();
            transformstamped.transform.translation.y = transform.getOrigin().y();
            transformstamped.transform.translation.z = transform.getOrigin().z();
            transformstamped.transform.rotation.x = transform.getRotation().x();
            transformstamped.transform.rotation.y = transform.getRotation().y();
            transformstamped.transform.rotation.z = transform.getRotation().z();
            transformstamped.transform.rotation.w = transform.getRotation().w();

            return transformstamped;
        }

        void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, const std_msgs::msg::Header &header, const int& failureId)
        {
            
            static double last_align_time = -1;

            // Quternion not normalized
            if (Q.x() * Q.x() + Q.y() * Q.y() + Q.z() * Q.z() + Q.w() * Q.w() < 0.99)
                return;

            // imu odometry in camera frame
            nav_msgs::msg::Odometry odometry;
            odometry.header = header;
            odometry.header.frame_id = "vins_world";
            odometry.child_frame_id = "vins_body";
            odometry.pose.pose.position.x = P.x();
            odometry.pose.pose.position.y = P.y();
            odometry.pose.pose.position.z = P.z();
            odometry.pose.pose.orientation.x = Q.x();
            odometry.pose.pose.orientation.y = Q.y();
            odometry.pose.pose.orientation.z = Q.z();
            odometry.pose.pose.orientation.w = Q.w();
            odometry.twist.twist.linear.x = V.x();
            odometry.twist.twist.linear.y = V.y();
            odometry.twist.twist.linear.z = V.z();
            pub_latest_odometry->publish(odometry);

            // imu odometry in ROS format (change rotation), used for lidar odometry initial guess
            odometry.pose.covariance[0] = double(failureId); // notify lidar odometry failure

            tf2::Quaternion q_odom_cam(Q.x(), Q.y(), Q.z(), Q.w());
            tf2::Quaternion q_cam_to_lidar(0, 1, 0, 0); // mark: camera - lidar
            tf2::Quaternion q_odom_ros = q_odom_cam * q_cam_to_lidar;
            // tf::quaternionTFToMsg(q_odom_ros, odometry.pose.pose.orientation);
            odometry.pose.pose.orientation.x = q_odom_ros.x();
            odometry.pose.pose.orientation.y = q_odom_ros.y();
            odometry.pose.pose.orientation.z = q_odom_ros.z();
            odometry.pose.pose.orientation.w = q_odom_ros.w();
            
            pub_latest_odometry_ros->publish(odometry);

            // TF of camera in vins_world in ROS format (change rotation), used for depth registration
            // tf::Transform t_w_body = tf::Transform(q_odom_ros, tf::Vector3(P.x(), P.y(), P.z()));
            // tf::StampedTransform trans_world_vinsbody_ros = tf::StampedTransform(t_w_body, header.stamp, "vins_world", "vins_body_ros");
            // br.sendTransform(trans_world_vinsbody_ros);
            geometry_msgs::msg::TransformStamped trans_world_vinsbody_ros;
            trans_world_vinsbody_ros.header.stamp = header.stamp;
            trans_world_vinsbody_ros.header.frame_id = "vins_world";
            trans_world_vinsbody_ros.child_frame_id = "vins_body_ros";
            trans_world_vinsbody_ros.transform.translation.x = P.x();
            trans_world_vinsbody_ros.transform.translation.y = P.y();
            trans_world_vinsbody_ros.transform.translation.z = P.z();
            trans_world_vinsbody_ros.transform.rotation.x = q_odom_ros.x();
            trans_world_vinsbody_ros.transform.rotation.y = q_odom_ros.y();
            trans_world_vinsbody_ros.transform.rotation.z = q_odom_ros.z();
            trans_world_vinsbody_ros.transform.rotation.w = q_odom_ros.w();
            // std::cout << "IN VISUALIZATION CPP " << std::endl;
            br->sendTransform(trans_world_vinsbody_ros);

            if (ALIGN_CAMERA_LIDAR_COORDINATE)
            {
                // static tf::Transform t_odom_world = tf::Transform(tf::createQuaternionFromRPY(0, 0, M_PI), tf::Vector3(0, 0, 0));
                geometry_msgs::msg::TransformStamped t_odom_world;
                tf2::Quaternion q;
                q.setRPY(0,0,M_PI);
                t_odom_world.transform.rotation.x = q.x();
                t_odom_world.transform.rotation.y = q.y();
                t_odom_world.transform.rotation.z = q.z();
                t_odom_world.transform.rotation.w = q.w();

                if (rclcpp::Time(header.stamp).seconds() - last_align_time > 1.0)
                {
                    try
                    {
                        // tf::StampedTransform trans_odom_baselink;
                        geometry_msgs::msg::TransformStamped trans_odom_baselink;

                        tf_buffer->canTransform("odom","base_link",tf2::TimePointZero,tf2::durationFromSec(1.0));

                        // listener.lookupTransform("odom","base_link", ros::Time(0), trans_odom_baselink);
                        trans_odom_baselink = tf_buffer->lookupTransform("odom","base_link",tf2::TimePointZero);

                        // t_odom_world = transformConversion(trans_odom_baselink) * transformConversion(trans_world_vinsbody_ros).inverse();

                        t_odom_world = transformTogeometry(transform_conversion(trans_odom_baselink) * transform_conversion(trans_world_vinsbody_ros).inverse(),header,"odom","vins_world");
                        

                        last_align_time = rclcpp::Time(header.stamp).seconds();
                    } 
                    catch (tf2::TransformException &ex){
                        RCLCPP_ERROR_STREAM(rclcpp::get_logger("visualization"),ex.what()<<"In visualization.cpp");
                    }
                }
                // br.sendTransform(tf::StampedTransform(t_odom_world, header.stamp, "odom", "vins_world"));
                t_odom_world.header.stamp = header.stamp;
                t_odom_world.header.frame_id = "odom";
                t_odom_world.child_frame_id = "vins_world";
                br->sendTransform(t_odom_world);
            } 
            else
            {
                // tf::Transform t_static = tf::Transform(tf::createQuaternionFromRPY(0, 0, M_PI), tf::Vector3(0, 0, 0));
                geometry_msgs::msg::TransformStamped t_static;
                t_static.header.stamp = header.stamp;
                t_static.header.frame_id = "odom";
                t_static.child_frame_id = "vins_world";
                t_static.transform.translation.x = 0;
                t_static.transform.translation.y = 0;
                t_static.transform.translation.z = 0;
                tf2::Quaternion q;
                q.setRPY(0,0,M_PI);
                t_static.transform.rotation.x = q.x();
                t_static.transform.rotation.y = q.y();
                t_static.transform.rotation.z = q.z();
                t_static.transform.rotation.w = q.w();
                
                // br.sendTransform(tf::StampedTransform(t_static, header.stamp, "odom", "vins_world"));
                br->sendTransform(t_static);
            }
        }

        void printStatistics(const Estimator &estimator, double t)
        {
            if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
                return;
            printf("position: %f, %f, %f\r", estimator.Ps[WINDOW_SIZE].x(), estimator.Ps[WINDOW_SIZE].y(), estimator.Ps[WINDOW_SIZE].z());
            RCLCPP_DEBUG_STREAM(rclcpp::get_logger("visualization"),"position: " << estimator.Ps[WINDOW_SIZE].transpose());
            RCLCPP_DEBUG_STREAM(rclcpp::get_logger("visualization"),"orientation: " << estimator.Vs[WINDOW_SIZE].transpose());
            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                //ROS_DEBUG("calibration result for camera %d", i);
                RCLCPP_DEBUG_STREAM(rclcpp::get_logger("visualization"),"extirnsic tic: " << estimator.tic[i].transpose());
                RCLCPP_DEBUG_STREAM(rclcpp::get_logger("visualization"),"extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());
                if (ESTIMATE_EXTRINSIC)
                {
                    cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
                    Eigen::Matrix3d eigen_R;
                    Eigen::Vector3d eigen_T;
                    eigen_R = estimator.ric[i];
                    eigen_T = estimator.tic[i];
                    cv::Mat cv_R, cv_T;
                    cv::eigen2cv(eigen_R, cv_R);
                    cv::eigen2cv(eigen_T, cv_T);
                    fs << "extrinsicRotation" << cv_R << "extrinsicTranslation" << cv_T;
                    fs.release();
                }
            }

            static double sum_of_time = 0;
            static int sum_of_calculation = 0;
            sum_of_time += t;
            sum_of_calculation++;
            RCLCPP_DEBUG(rclcpp::get_logger("visualization"),"vo solver costs: %f ms", t);
            RCLCPP_DEBUG(rclcpp::get_logger("visualization"),"average of time %f ms", sum_of_time / sum_of_calculation);

            sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
            last_path = estimator.Ps[WINDOW_SIZE];
            RCLCPP_DEBUG(rclcpp::get_logger("visualization"),"sum of path %f", sum_of_path);
            if (ESTIMATE_TD)
                RCLCPP_INFO(rclcpp::get_logger("visualization"),"td %f", estimator.td);
        }

        void pubOdometry(const Estimator &estimator, const std_msgs::msg::Header &header)
        {
            std::cout << "++++++++++++++++++++++ Inside pubOdometry ++++++++++++++++++++++" << std::endl;

            if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            {
                nav_msgs::msg::Odometry odometry;
                odometry.header = header;
                odometry.header.frame_id = "vins_world";
                odometry.child_frame_id = "vins_world";
                Quaterniond tmp_Q;
                tmp_Q = Quaterniond(estimator.Rs[WINDOW_SIZE]);
                odometry.pose.pose.position.x = estimator.Ps[WINDOW_SIZE].x();
                odometry.pose.pose.position.y = estimator.Ps[WINDOW_SIZE].y();
                odometry.pose.pose.position.z = estimator.Ps[WINDOW_SIZE].z();
                odometry.pose.pose.orientation.x = tmp_Q.x();
                odometry.pose.pose.orientation.y = tmp_Q.y();
                odometry.pose.pose.orientation.z = tmp_Q.z();
                odometry.pose.pose.orientation.w = tmp_Q.w();
                odometry.twist.twist.linear.x = estimator.Vs[WINDOW_SIZE].x();
                odometry.twist.twist.linear.y = estimator.Vs[WINDOW_SIZE].y();
                odometry.twist.twist.linear.z = estimator.Vs[WINDOW_SIZE].z();
                pub_odometry->publish(odometry);

                static double path_save_time = -1;
                if (rclcpp::Time(header.stamp).seconds() - path_save_time > 0.5)
                {
                    path_save_time = rclcpp::Time(header.stamp).seconds();
                    geometry_msgs::msg::PoseStamped pose_stamped;
                    pose_stamped.header = header;
                    pose_stamped.header.frame_id = "vins_world";
                    pose_stamped.pose = odometry.pose.pose;
                    path.header = header;
                    path.header.frame_id = "vins_world";
                    path.poses.push_back(pose_stamped);
                    pub_path->publish(path);
                }
            }
        }

        void pubKeyPoses(const Estimator &estimator, const std_msgs::msg::Header &header)
        {
            if (pub_key_poses->get_subscription_count() == 0)
                return;

            if (estimator.key_poses.size() == 0)
                return;
            visualization_msgs::msg::Marker key_poses;
            key_poses.header = header;
            key_poses.header.frame_id = "vins_world";
            key_poses.ns = "key_poses";
            key_poses.type = visualization_msgs::msg::Marker::SPHERE_LIST;
            key_poses.action = visualization_msgs::msg::Marker::ADD;
            key_poses.pose.orientation.w = 1.0;
            key_poses.lifetime = rclcpp::Duration(0); //ros::Duration();

            //static int key_poses_id = 0;
            key_poses.id = 0; //key_poses_id++;
            key_poses.scale.x = 0.05;
            key_poses.scale.y = 0.05;
            key_poses.scale.z = 0.05;
            key_poses.color.r = 1.0;
            key_poses.color.a = 1.0;

            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                geometry_msgs::msg::Point pose_marker;
                Vector3d correct_pose;
                correct_pose = estimator.key_poses[i];
                pose_marker.x = correct_pose.x();
                pose_marker.y = correct_pose.y();
                pose_marker.z = correct_pose.z();
                key_poses.points.push_back(pose_marker);
            }
            pub_key_poses->publish(key_poses);
        }

        void pubCameraPose(const Estimator &estimator, const std_msgs::msg::Header &header)
        {
            if (pub_camera_pose_visual->get_subscription_count() == 0)
                return;

            int idx2 = WINDOW_SIZE - 1;

            if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            {
                int i = idx2;
                Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
                Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);

                nav_msgs::msg::Odometry odometry;
                odometry.header = header;
                odometry.header.frame_id = "vins_world";
                odometry.pose.pose.position.x = P.x();
                odometry.pose.pose.position.y = P.y();
                odometry.pose.pose.position.z = P.z();
                odometry.pose.pose.orientation.x = R.x();
                odometry.pose.pose.orientation.y = R.y();
                odometry.pose.pose.orientation.z = R.z();
                odometry.pose.pose.orientation.w = R.w();

                pub_camera_pose->publish(odometry);

                cameraposevisual.reset();
                cameraposevisual.add_pose(P, R);
                cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
            }
        }

        void pubPointCloud(const Estimator &estimator, const std_msgs::msg::Header &header)
        {
            if (pub_point_cloud->get_subscription_count() != 0)
            {
                sensor_msgs::msg::PointCloud point_cloud;
                point_cloud.header = header;
                point_cloud.header.frame_id = "vins_world";

                sensor_msgs::msg::ChannelFloat32 intensity_channel;
                intensity_channel.name = "intensity";

                for (auto &it_per_id : estimator.f_manager.feature)
                {
                    int used_num;
                    used_num = it_per_id.feature_per_frame.size();
                    if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                        continue;
                    if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
                        continue;
                    
                    int imu_i = it_per_id.start_frame;
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                    Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

                    geometry_msgs::msg::Point32 p;
                    p.x = w_pts_i(0);
                    p.y = w_pts_i(1);
                    p.z = w_pts_i(2);
                    point_cloud.points.push_back(p);

                    if (it_per_id.lidar_depth_flag == false)
                        intensity_channel.values.push_back(0);
                    else
                        intensity_channel.values.push_back(1);
                }

                point_cloud.channels.push_back(intensity_channel);
                pub_point_cloud->publish(point_cloud);
            }
            
            // pub margined potin
            if (pub_margin_cloud->get_subscription_count() != 0)
            {
                sensor_msgs::msg::PointCloud margin_cloud;
                margin_cloud.header = header;
                margin_cloud.header.frame_id = "vins_world";

                sensor_msgs::msg::ChannelFloat32 intensity_channel;
                intensity_channel.name = "intensity";

                for (auto &it_per_id : estimator.f_manager.feature)
                { 
                    int used_num;
                    used_num = it_per_id.feature_per_frame.size();
                    if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                        continue;

                    if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 
                        && it_per_id.solve_flag == 1 )
                    {
                        int imu_i = it_per_id.start_frame;
                        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                        Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

                        geometry_msgs::msg::Point32 p;
                        p.x = w_pts_i(0);
                        p.y = w_pts_i(1);
                        p.z = w_pts_i(2);
                        margin_cloud.points.push_back(p);

                        if (it_per_id.lidar_depth_flag == false)
                            intensity_channel.values.push_back(0);
                        else
                            intensity_channel.values.push_back(1);
                    }
                }

                margin_cloud.channels.push_back(intensity_channel);
                pub_margin_cloud->publish(margin_cloud);
            }
        }

        void pubTF(const Estimator &estimator, const std_msgs::msg::Header &header)
        {
            if( estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
                return;
            
            geometry_msgs::msg::TransformStamped transform;
            tf2::Quaternion q;
            // body frame
            Vector3d correct_t;
            Quaterniond correct_q;
            correct_t = estimator.Ps[WINDOW_SIZE];
            correct_q = estimator.Rs[WINDOW_SIZE];

            transform.header.stamp = header.stamp;
            transform.header.frame_id = "vins_world";
            transform.child_frame_id = "vins_body";
            transform.transform.translation.x = correct_t(0);
            transform.transform.translation.y = correct_t(1);
            transform.transform.translation.z = correct_t(2);
            transform.transform.rotation.x = correct_q.x();
            transform.transform.rotation.y = correct_q.y();
            transform.transform.rotation.z = correct_q.z();
            transform.transform.rotation.w = correct_q.w(); 

            br->sendTransform(transform);

            // camera frame
            transform.header.stamp = header.stamp;
            transform.header.frame_id = "vins_body";
            transform.child_frame_id = "vins_camera";
            transform.transform.translation.x = estimator.tic[0].x();
            transform.transform.translation.y = estimator.tic[0].y();
            transform.transform.translation.z = estimator.tic[0].z();
            transform.transform.rotation.x = Quaterniond(estimator.ric[0]).x();
            transform.transform.rotation.y = Quaterniond(estimator.ric[0]).y();
            transform.transform.rotation.z = Quaterniond(estimator.ric[0]).z();
            transform.transform.rotation.w = Quaterniond(estimator.ric[0]).w(); 

            br->sendTransform(transform);

            nav_msgs::msg::Odometry odometry;
            odometry.header = header;
            odometry.header.frame_id = "vins_world";
            odometry.pose.pose.position.x = estimator.tic[0].x();
            odometry.pose.pose.position.y = estimator.tic[0].y();
            odometry.pose.pose.position.z = estimator.tic[0].z();
            Quaterniond tmp_q{estimator.ric[0]};
            odometry.pose.pose.orientation.x = tmp_q.x();
            odometry.pose.pose.orientation.y = tmp_q.y();
            odometry.pose.pose.orientation.z = tmp_q.z();
            odometry.pose.pose.orientation.w = tmp_q.w();
            pub_extrinsic->publish(odometry);
        }

        void pubKeyframe(const Estimator &estimator)
        {
            if (pub_keyframe_pose->get_subscription_count() == 0 && pub_keyframe_point->get_subscription_count() == 0)
                return;

            // pub camera pose, 2D-3D points of keyframe
            if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0)
            {
                int i = WINDOW_SIZE - 2;
                //Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
                Vector3d P = estimator.Ps[i];
                Quaterniond R = Quaterniond(estimator.Rs[i]);

                nav_msgs::msg::Odometry odometry;
                odometry.header = estimator.Headers[WINDOW_SIZE - 2];
                odometry.header.frame_id = "vins_world";
                odometry.pose.pose.position.x = P.x();
                odometry.pose.pose.position.y = P.y();
                odometry.pose.pose.position.z = P.z();
                odometry.pose.pose.orientation.x = R.x();
                odometry.pose.pose.orientation.y = R.y();
                odometry.pose.pose.orientation.z = R.z();
                odometry.pose.pose.orientation.w = R.w();

                pub_keyframe_pose->publish(odometry);


                sensor_msgs::msg::PointCloud point_cloud;
                point_cloud.header = estimator.Headers[WINDOW_SIZE - 2];
                for (auto &it_per_id : estimator.f_manager.feature)
                {
                    int frame_size = it_per_id.feature_per_frame.size();
                    if(it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag == 1)
                    {

                        int imu_i = it_per_id.start_frame;
                        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                        Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0])
                                            + estimator.Ps[imu_i];
                        geometry_msgs::msg::Point32 p;
                        p.x = w_pts_i(0);
                        p.y = w_pts_i(1);
                        p.z = w_pts_i(2);
                        point_cloud.points.push_back(p);

                        int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
                        sensor_msgs::msg::ChannelFloat32 p_2d;
                        p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.x());
                        p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.y());
                        p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.x());
                        p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.y());
                        p_2d.values.push_back(it_per_id.feature_id);
                        point_cloud.channels.push_back(p_2d);
                    }
                }
                pub_keyframe_point->publish(point_cloud);
            }
        }

        // from parameters
        void readParameters()
        {
            std::string config_file;
            
            this->declare_parameter("config_file","");
            this->get_parameter("config_file",config_file);

            std::cout << "VI CONFIG FILE : " << config_file;

            cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
            if(!fsSettings.isOpened())
            {
                std::cerr << "ERROR: Wrong path to settings" << std::endl;
            }

            fsSettings["project_name"] >> PROJECT_NAME;
            
            std::string pkg_path = ament_index_cpp::get_package_share_directory(PROJECT_NAME);

            fsSettings["imu_topic"] >> IMU_TOPIC;

            fsSettings["use_lidar"] >> USE_LIDAR;
            fsSettings["align_camera_lidar_estimation"] >> ALIGN_CAMERA_LIDAR_COORDINATE;

            SOLVER_TIME = fsSettings["max_solver_time"];
            NUM_ITERATIONS = fsSettings["max_num_iterations"];
            MIN_PARALLAX = fsSettings["keyframe_parallax"];
            MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

            ACC_N = fsSettings["acc_n"];
            ACC_W = fsSettings["acc_w"];
            GYR_N = fsSettings["gyr_n"];
            GYR_W = fsSettings["gyr_w"];
            G.z() = fsSettings["g_norm"];
            ROW = fsSettings["image_height"];
            COL = fsSettings["image_width"];
            RCLCPP_INFO(this->get_logger(),"Image dimention: ROW: %f COL: %f ", ROW, COL);

            ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
            if (ESTIMATE_EXTRINSIC == 2)
            {
                RCLCPP_INFO(this->get_logger(),"have no prior about extrinsic param, calibrate extrinsic param");
                RIC.push_back(Eigen::Matrix3d::Identity());
                TIC.push_back(Eigen::Vector3d::Zero());
                EX_CALIB_RESULT_PATH = pkg_path + "/config/extrinsic_parameter.csv";

            }
            else 
            {
                if ( ESTIMATE_EXTRINSIC == 1)
                {
                    RCLCPP_INFO(this->get_logger(),"Optimize extrinsic param around initial guess!");
                    EX_CALIB_RESULT_PATH = pkg_path + "/config/extrinsic_parameter.csv";
                }
                if (ESTIMATE_EXTRINSIC == 0)
                    RCLCPP_INFO(this->get_logger()," Fix extrinsic param.");

                cv::Mat cv_R, cv_T;
                fsSettings["extrinsicRotation"] >> cv_R;
                fsSettings["extrinsicTranslation"] >> cv_T;
                Eigen::Matrix3d eigen_R;
                Eigen::Vector3d eigen_T;
                cv::cv2eigen(cv_R, eigen_R);
                cv::cv2eigen(cv_T, eigen_T);
                Eigen::Quaterniond Q(eigen_R);
                eigen_R = Q.normalized();
                RIC.push_back(eigen_R);
                TIC.push_back(eigen_T);
                RCLCPP_INFO_STREAM(this->get_logger(),"Extrinsic_R : " << std::endl << RIC[0]);
                RCLCPP_INFO_STREAM(this->get_logger(),"Extrinsic_T : " << std::endl << TIC[0].transpose());
                
            } 

            INIT_DEPTH = 5.0;
            BIAS_ACC_THRESHOLD = 0.1;
            BIAS_GYR_THRESHOLD = 0.1;

            TD = fsSettings["td"];
            ESTIMATE_TD = fsSettings["estimate_td"];
            if (ESTIMATE_TD)
                RCLCPP_INFO_STREAM(this->get_logger(),"Unsynchronized sensors, online estimate time offset, initial td: " << TD);
            else
                RCLCPP_INFO_STREAM(this->get_logger(),"Synchronized sensors, fix time offset: " << TD);

            ROLLING_SHUTTER = fsSettings["rolling_shutter"];
            if (ROLLING_SHUTTER)
            {
                TR = fsSettings["rolling_shutter_tr"];
                RCLCPP_INFO_STREAM(this->get_logger(),"rolling shutter camera, read out time per line: " << TR);
            }
            else
            {
                TR = 0;
            }
            
            fsSettings.release();
            usleep(100);
        }

};

int main(int argc , char **argv)
{
    rclcpp::init(argc,argv);

    auto EN = std::make_shared<EstimatorNode>();

    std::thread measurement_process(&EstimatorNode::process,EN);

    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(EN);

    RCLCPP_INFO(EN->get_logger(),"\033[1;32m----> Visual Odometry Estimator Started.\033[0m");

    

    exec.spin();

    measurement_process.join();

    rclcpp::shutdown();

    return 0;
}
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <novatel_gps_msgs/NovatelXYZ.h>
#include <novatel_gps_msgs/NovatelUtmPosition.h>
#include <novatel_gps_msgs/Inspva.h>
#include <novatel_gps_msgs/NovatelPosition.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/Imu.h>
#include <gps_common/GPSFix.h>
#include <mmc_msgs/localization2D_msg.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <math.h>
#include <UTM.h>

#include <proj.h>

class gps_world_tf_node
{
private:
  /* data */
public:

  ros::NodeHandle node;
  ros::Subscriber sub1;
  ros::Subscriber sub2;
  ros::Subscriber sub3;
  ros::Subscriber sub4;
  ros::Publisher pub1;
  ros::Publisher pub2;
  ros::Publisher pub3;
  ros::Publisher pub4;
  ros::Publisher pub5;
  ros::Publisher pub6;
  ros::Publisher pub7;
  tf::Vector3 xyz_init;

  sensor_msgs::NavSatFix fix_msg;
  double r, theta_init, s;
  double N_init,E_init,H_init;
  double yaw_stored;

  tf2::Quaternion quat_init;
  geometry_msgs::TransformStamped static_init_transformStamped;
  geometry_msgs::TransformStamped gps_transformStamped;
  int Z_init;
  bool initialized = false;
  bool initialized_quat = false;
  int count;
  tf2::Quaternion q, q_;
  double height_current;
  gps_world_tf_node(/* args */);
  ~gps_world_tf_node();
  void fixCallback(const gps_common::GPSFixConstPtr& msg);
  void fixCallback_navsat(const sensor_msgs::NavSatFixConstPtr& msg);
  void inspvaCallback(const novatel_gps_msgs::InspvaConstPtr& msg);
  void bestutmCallback(const novatel_gps_msgs::NovatelUtmPositionConstPtr& msg);
  void latlontoEPSG5179XY(double lat, double lon, double &x, double &y);
  void latlonalttoEPSG5179XYZ(double lat, double lon,double alt, double &x, double &y, double &z);
  void imuCallback(const sensor_msgs::ImuConstPtr & msg);

  PJ_CONTEXT *C_proj;
  PJ *P_proj;
  
  ros::Time past_time;
};

gps_world_tf_node::gps_world_tf_node(/* args */)
{
  sub1 = node.subscribe("/sensors/gps/gps", 10, &gps_world_tf_node::fixCallback, this);
  sub3 = node.subscribe("/sensors/gps/fix_extra", 10, &gps_world_tf_node::fixCallback_navsat, this);
  sub2 = node.subscribe("/sensors/gps/inspva", 10, &gps_world_tf_node::inspvaCallback, this);
  sub4 = node.subscribe("/sensors/gps/imu_extra", 10, &gps_world_tf_node::imuCallback, this);


  pub1 = node.advertise<sensor_msgs::NavSatFix>("/sensors/gps/xyzinit",10000,true);
  pub2 = node.advertise<std_msgs::Float64>("/localization/height",  1);
  pub3 = node.advertise<mmc_msgs::localization2D_msg>("/localization/pose_2d_gps",  1);
  pub4 = node.advertise<geometry_msgs::PoseWithCovarianceStamped>("/localization/pose_6dof_gps",  1);
  pub5 = node.advertise<geometry_msgs::PoseWithCovarianceStamped>("/localization/pose_6dof_imu",  1);
  pub6 = node.advertise<geometry_msgs::Vector3Stamped>("/localization/pose_enu_vel_gps",  1);
  pub7 = node.advertise<geometry_msgs::PoseWithCovarianceStamped>("/localization/pose_6dof_gps_init",  1);

  H_init = 1.5;
  count = 0;
  C_proj = proj_context_create();
  P_proj = proj_create_crs_to_crs (C_proj, "EPSG:4326", "epsg:5179", NULL);

}

gps_world_tf_node::~gps_world_tf_node()
{
}

void gps_world_tf_node::latlontoEPSG5179XY(double lat, double lon, double &x, double &y)
{

  PJ *norm_proj;
  PJ_COORD a, b;
  
  /* This will ensure that the order of coordinates for the input CRS */
  /* will be longitude, latitude, whereas EPSG:4326 mandates latitude, */
  /* longitude */
  norm_proj = proj_normalize_for_visualization(C_proj, P_proj);

  /* a coordinate union representing Copenhagen: 55d N, 12d E    */
  /* Given that we have used proj_normalize_for_visualization(), the order of
  /* coordinates is longitude, latitude, and values are expressed in degrees. */
  a = proj_coord(lon, lat, 0, 0);

  /* transform to UTM zone 32, then back to geographical */
  b = proj_trans(norm_proj, PJ_FWD, a);

  /* Clean up */
  // proj_destroy(P_proj);
  // proj_context_destroy(C_proj);
  x = b.enu.e;
  y = b.enu.n;
}

void gps_world_tf_node::latlonalttoEPSG5179XYZ(double lat, double lon, double alt, double &x, double &y, double &z)
{
  PJ *norm_proj;
  PJ_COORD a, b;
  
  /* This will ensure that the order of coordinates for the input CRS */
  /* will be longitude, latitude, whereas EPSG:4326 mandates latitude, */
  /* longitude */
  norm_proj = proj_normalize_for_visualization(C_proj, P_proj);

  /* a coordinate union representing Copenhagen: 55d N, 12d E    */
  /* Given that we have used proj_normalize_for_visualization(), the order of
  /* coordinates is longitude, latitude, and values are expressed in degrees. */
  a = proj_coord(lon, lat, alt, 0);

  /* transform to UTM zone 32, then back to geographical */
  b = proj_trans(norm_proj, PJ_FWD, a);

  /* Clean up */
  proj_destroy(P_proj);
  proj_context_destroy(C_proj);
  x = b.enu.e;
  y = b.enu.n;
  z = b.enu.u;
}

void gps_world_tf_node::fixCallback(const gps_common::GPSFixConstPtr& msg){
  if (initialized_quat && initialized){
    
    double N,E;
    // auto before = ros::Time::now();
    latlontoEPSG5179XY(msg->latitude, msg->longitude, E, N);
    // std::cout << "proj time is: " << ros::Time::now() - before <<std::endl;
    double E_test, N_test, U_test;
    // latlonalttoEPSG5179XYZ(msg->latitude, msg->longitude, msg->altitude ,E_test, N_test,U_test);
    // std::cout << "E: "<< E_test <<", N: "<< N_test <<", U: "<<U_test <<std::endl;
    mmc_msgs::localization2D_msg msg_pose;
    msg_pose.time = msg->header.stamp;
    msg_pose.EPSG = 5179;
    msg_pose.east = E;
    msg_pose.north = N;
    msg_pose.gps_initialized = true;
    msg_pose.v = msg->speed;
    msg_pose.yaw = yaw_stored;
    msg_pose.sig_east = msg->position_covariance[0];
    msg_pose.sig_north = msg->position_covariance[4];
    // msg_pose.sig_east = msg->position_covariance[0,0];
    pub3.publish(msg_pose);

    geometry_msgs::PoseWithCovarianceStamped pose_6dof;
    pose_6dof.header.stamp = msg->header.stamp;
    pose_6dof.header.frame_id = "epsg5179";
    pose_6dof.pose.pose.position.x = E;
    pose_6dof.pose.pose.position.y = N;
    pose_6dof.pose.pose.position.z = height_current;
    pose_6dof.pose.pose.orientation.w = q.w();
    pose_6dof.pose.pose.orientation.x = q.x();
    pose_6dof.pose.pose.orientation.y = q.y();
    pose_6dof.pose.pose.orientation.z = q.z();

    pose_6dof.pose.covariance[0] = msg->position_covariance[0];
    pose_6dof.pose.covariance[7] = msg->position_covariance[4];
    pose_6dof.pose.covariance[14] = msg->position_covariance[8];

    pub4.publish(pose_6dof);

    static tf2_ros::TransformBroadcaster br_;
    gps_transformStamped.header.stamp = msg->header.stamp;
    gps_transformStamped.header.frame_id = "epsg5179";
    gps_transformStamped.child_frame_id = "gps";
    gps_transformStamped.transform.translation.x = E;
    gps_transformStamped.transform.translation.y = N;
    gps_transformStamped.transform.translation.z = height_current;
    if (count%2 == 0)
    {
      br_.sendTransform(gps_transformStamped);
      count++;
    }else{
      count = 0;
    }
      

  } 
  else
  {
    geometry_msgs::PoseWithCovarianceStamped pose_6dof;
    pose_6dof.header.stamp = msg->header.stamp;
    pose_6dof.header.frame_id = "epsg5179";
    pose_6dof.pose.pose.position.x = E_init;
    pose_6dof.pose.pose.position.y = N_init;
    pose_6dof.pose.pose.position.z = height_current;
    pose_6dof.pose.pose.orientation.w = q.w();
    pose_6dof.pose.pose.orientation.x = q.x();
    pose_6dof.pose.pose.orientation.y = q.y();
    pose_6dof.pose.pose.orientation.z = q.z();

    pose_6dof.pose.covariance[0] = msg->position_covariance[0];
    pose_6dof.pose.covariance[7] = msg->position_covariance[4];
    pose_6dof.pose.covariance[14] = msg->position_covariance[8];

    pub7.publish(pose_6dof);

    static tf2_ros::StaticTransformBroadcaster static_broadcaster;
    latlontoEPSG5179XY(msg->latitude, msg->longitude, E_init, N_init);
    if (initialized_quat)
      initialized = true;
    static_init_transformStamped.header.stamp = ros::Time::now();
    static_init_transformStamped.header.frame_id = "epsg5179";
    static_init_transformStamped.child_frame_id = "odom";
    static_init_transformStamped.transform.translation.x = E_init;
    static_init_transformStamped.transform.translation.y = N_init;
    static_init_transformStamped.transform.translation.z = height_current;

    static_broadcaster.sendTransform(static_init_transformStamped);

  }
}

void gps_world_tf_node::fixCallback_navsat(const sensor_msgs::NavSatFixConstPtr& msg){
  if (!initialized_quat)
    return;
  double N,E;
  // auto before = ros::Time::now();
  latlontoEPSG5179XY(msg->latitude, msg->longitude, E, N);
  // std::cout << "proj time is: " << ros::Time::now() - before <<std::endl;
  double E_test, N_test, U_test;
  // latlonalttoEPSG5179XYZ(msg->latitude, msg->longitude, msg->altitude ,E_test, N_test,U_test);
  // std::cout << "E: "<< E <<", N: "<< N <<", U: "<<U_test <<std::endl;
  mmc_msgs::localization2D_msg msg_pose;
  msg_pose.time = msg->header.stamp;
  msg_pose.EPSG = 5179;
  msg_pose.east = E;
  msg_pose.north = N;
  msg_pose.gps_initialized = true;
  msg_pose.v = -1;
  msg_pose.yaw = yaw_stored;
  msg_pose.sig_east = msg->position_covariance[0];
  msg_pose.sig_north = msg->position_covariance[4];
  // msg_pose.sig_east = msg->position_covariance[0,0];
  pub3.publish(msg_pose);

  geometry_msgs::PoseWithCovarianceStamped pose_6dof;
  pose_6dof.header.stamp = msg->header.stamp;
  pose_6dof.header.frame_id = "epsg5179";
  pose_6dof.pose.pose.position.x = E;
  pose_6dof.pose.pose.position.y = N;
  pose_6dof.pose.pose.position.z = msg->altitude-24;
  pose_6dof.pose.pose.orientation.w = q.getW();
  pose_6dof.pose.pose.orientation.x = q.getX();
  pose_6dof.pose.pose.orientation.y = q.getY();
  pose_6dof.pose.pose.orientation.z = q.getZ();

  pose_6dof.pose.covariance[0] = msg->position_covariance[0];
  pose_6dof.pose.covariance[7] = msg->position_covariance[4];
  pose_6dof.pose.covariance[14] = msg->position_covariance[8];
  pub4.publish(pose_6dof);

  static tf2_ros::TransformBroadcaster br;
  static tf2_ros::StaticTransformBroadcaster br_;
  gps_transformStamped.header.stamp = msg->header.stamp;
  gps_transformStamped.header.frame_id = "epsg5179";
  gps_transformStamped.child_frame_id = "gps";
  if (!initialized)
    gps_transformStamped.child_frame_id = "global_aligned";
  gps_transformStamped.transform.translation.x = E;
  gps_transformStamped.transform.translation.y = N;
  gps_transformStamped.transform.translation.z = msg->altitude-24;
  gps_transformStamped.transform.rotation.w = q.getW();
  gps_transformStamped.transform.rotation.x = q.getX();
  gps_transformStamped.transform.rotation.y = q.getY();
  gps_transformStamped.transform.rotation.z = q.getZ();

  if (!initialized)
  {
    br_.sendTransform(gps_transformStamped);
    gps_transformStamped.child_frame_id = "global";
    auto q_rotate = q;
    q_rotate.setRPY(0,0,3.14);
    q = q_rotate * q;
    gps_transformStamped.transform.rotation.w = q.getW();
    gps_transformStamped.transform.rotation.x = q.getX();
    gps_transformStamped.transform.rotation.y = q.getY();
    gps_transformStamped.transform.rotation.z = q.getZ();
    br_.sendTransform(gps_transformStamped);
  }
  else
    br.sendTransform(gps_transformStamped);
  initialized = true;
}

void gps_world_tf_node::imuCallback(const sensor_msgs::ImuConstPtr& msg){
  q.setX(msg->orientation.x);
  q.setY(msg->orientation.y);
  q.setZ(msg->orientation.z);
  q.setW(msg->orientation.w);
  q.normalize();
  q_.setRPY(0,0,0.36);
  q = q_*q;
  q.normalize();
  initialized_quat = true;
}

void gps_world_tf_node::inspvaCallback(const novatel_gps_msgs::InspvaConstPtr& msg){
  if (!initialized_quat){
    
    fix_msg.latitude = msg->latitude;
    fix_msg.longitude = msg->longitude;
    fix_msg.header.frame_id = "odom";
    
    pub1.publish(fix_msg);
    // LatLonToUTMXY(msg->latitude, msg->longitude, 52, E_init, N_init);

    

    
    quat_init.setRPY(-msg->roll/180*3.141592, -msg->pitch/180*3.141592, 1.57-msg->azimuth/180*3.141592);
    static_init_transformStamped.transform.rotation.x = quat_init.x();
    static_init_transformStamped.transform.rotation.y = quat_init.y();
    static_init_transformStamped.transform.rotation.z = quat_init.z();
    static_init_transformStamped.transform.rotation.w = quat_init.w();
    initialized_quat = true;
    yaw_stored = 1.57-msg->azimuth/180*3.141592;
    q.setRPY(-msg->roll/180*3.141592, -msg->pitch/180*3.141592,1.57-msg->azimuth/180*3.141592);
    return;
    
  }
  static tf::TransformBroadcaster br_ins;
  tf::Transform transform;
  height_current = msg->height;
  std_msgs::Float64 height_;
  height_.data = height_current;
  pub2.publish(height_);
  yaw_stored = 1.57-msg->azimuth/180*3.141592;
  q.setRPY(-msg->roll/180*3.141592, -msg->pitch/180*3.141592,1.57-msg->azimuth/180*3.141592);
  gps_transformStamped.transform.rotation.x = q.x();
  gps_transformStamped.transform.rotation.y = q.y();
  gps_transformStamped.transform.rotation.z = q.z();
  gps_transformStamped.transform.rotation.w = q.w();
  // transform.setRotation(q);
  double N,E;
  int Z;
  latlontoEPSG5179XY(msg->latitude, msg->longitude, E, N);
  transform.setOrigin(tf::Vector3(E, N, height_current));
  geometry_msgs::PoseWithCovarianceStamped pose_6dof_;
  pose_6dof_.header.stamp = msg->header.stamp;
  pose_6dof_.header.frame_id = "epsg5179";
  pose_6dof_.pose.pose.position.x = E;
  pose_6dof_.pose.pose.position.y = N;
  pose_6dof_.pose.pose.position.z = height_current;
  pose_6dof_.pose.pose.orientation.w = q.w();
  pose_6dof_.pose.pose.orientation.x = q.x();
  pose_6dof_.pose.pose.orientation.y = q.y();
  pose_6dof_.pose.pose.orientation.z = q.z();

  geometry_msgs::Vector3Stamped gps_vel;
  gps_vel.header.frame_id = "epsg5179";
  gps_vel.header.stamp = msg->header.stamp;
  gps_vel.vector.x = msg->east_velocity;
  gps_vel.vector.y = msg->north_velocity;
  gps_vel.vector.z = msg->up_velocity;

  // pose_6dof.pose.covariance[0] = msg->position_covariance[0];
  // pose_6dof.pose.covariance[7] = msg->position_covariance[4];
  // pose_6dof.pose.covariance[14] = msg->position_covariance[8];

  pub5.publish(pose_6dof_);
  pub6.publish(gps_vel);
}

int main(int argc, char** argv){
  ros::init(argc, argv, "gps_world_tf");
  gps_world_tf_node gps_world_tf = gps_world_tf_node();
  ros::spin();
  return 0;
};
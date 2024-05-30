#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <novatel_gps_msgs/NovatelXYZ.h>
#include <novatel_gps_msgs/NovatelUtmPosition.h>
#include <novatel_gps_msgs/Inspva.h>
#include <sensor_msgs/NavSatFix.h>
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
  ros::Publisher pub1;
  tf::Vector3 xyz_init;
  sensor_msgs::NavSatFix fix_msg;
  double r, theta_init, s;
  double N_init,E_init,H_init;
  tf2::Quaternion quat_init;
  geometry_msgs::TransformStamped static_init_transformStamped;
  int Z_init;
  bool initialized = false;
  bool initialized_quat = false;
  tf::Quaternion q;
  gps_world_tf_node(/* args */);
  ~gps_world_tf_node();
  void fixCallback(const sensor_msgs::NavSatFixConstPtr& msg);
  void inspvaCallback(const novatel_gps_msgs::InspvaConstPtr& msg);
  void bestutmCallback(const novatel_gps_msgs::NovatelUtmPositionConstPtr& msg);
  ros::Time past_time;
  tf::Transform transform_init;
};

gps_world_tf_node::gps_world_tf_node(/* args */)
{
  sub1 = node.subscribe("/sensors/gps/fix", 10, &gps_world_tf_node::fixCallback, this);
  sub2 = node.subscribe("/sensors/gps/inspva", 10, &gps_world_tf_node::inspvaCallback, this);
  sub3 = node.subscribe("/sensors/gps/bestutm", 10, &gps_world_tf_node::bestutmCallback, this);

  pub1 = node.advertise<sensor_msgs::NavSatFix>("/sensors/gps/xyzinit",10000,true);
  // q.setW(1.0);
}

gps_world_tf_node::~gps_world_tf_node()
{
}

void gps_world_tf_node::fixCallback(const sensor_msgs::NavSatFixConstPtr& msg){
  // if (initialized_quat && initialized){
  //   tf::Transform transform_;
  //   static tf::TransformBroadcaster br_;
  //   transform_.setRotation(q);
  //   double N,E;
  //   LatLonToUTMXY(msg->latitude, msg->longitude, 52, E, N);
  //   transform_.setOrigin(tf::Vector3(E, N, 0));
  //   br_.sendTransform(tf::StampedTransform(transform_init.inverse() * transform_, ros::Time::now(), "map", "gps_fix"));
  // }
  // else
  // {
  //   static tf2_ros::StaticTransformBroadcaster static_broadcaster;
  //   LatLonToUTMXY(msg->latitude, msg->longitude, 52, E_init, N_init);
  //   if (initialized_quat)
  //     initialized = true;
  //   static_init_transformStamped.header.stamp = ros::Time::now();
  //   static_init_transformStamped.header.frame_id = "utm";
  //   static_init_transformStamped.child_frame_id = "map";
  //   static_init_transformStamped.transform.translation.x = E_init;
  //   static_init_transformStamped.transform.translation.y = N_init;
  //   static_init_transformStamped.transform.translation.z = H_init;
  //   static_broadcaster.sendTransform(static_init_transformStamped);
  //   transform_init.setRotation(q);   
  //   transform_init.setOrigin(tf::Vector3(E_init, N_init, H_init));
  // }
}

void gps_world_tf_node::bestutmCallback(const novatel_gps_msgs::NovatelUtmPositionConstPtr& msg){
  if (initialized_quat && initialized){
    tf::Transform transform_;
    static tf::TransformBroadcaster br_bestutm;
    transform_.setRotation(q);
    transform_.setOrigin(tf::Vector3(msg->easting, msg->northing, msg->height));
    br_bestutm.sendTransform(tf::StampedTransform(transform_init.inverse() * transform_, ros::Time::now(), "map", "gps_utm"));
  }
  else
  {
    static tf2_ros::StaticTransformBroadcaster static_broadcaster;
    E_init = msg->easting;
    N_init = msg->northing;
    H_init = msg->height;
    if (initialized_quat)
      initialized = true;
    static_init_transformStamped.header.stamp = ros::Time::now();
    static_init_transformStamped.header.frame_id = "utm";
    static_init_transformStamped.child_frame_id = "map";
    static_init_transformStamped.transform.translation.x = E_init;
    static_init_transformStamped.transform.translation.y = N_init;
    static_init_transformStamped.transform.translation.z = H_init;
    static_broadcaster.sendTransform(static_init_transformStamped);
    transform_init.setRotation(q);   
    transform_init.setOrigin(tf::Vector3(E_init, N_init, H_init));
  }
}

void gps_world_tf_node::inspvaCallback(const novatel_gps_msgs::InspvaConstPtr& msg){
  if (!initialized_quat){
    
    fix_msg.latitude = msg->latitude;
    fix_msg.longitude = msg->longitude;
    fix_msg.header.frame_id = "map";
    initialized_quat = true;
    pub1.publish(fix_msg);
    // LatLonToUTMXY(msg->latitude, msg->longitude, 52, E_init, N_init);

    

    
    quat_init.setRPY(msg->pitch/180*3.141592, msg->roll/180*3.141592, -msg->azimuth/180*3.141592);
    static_init_transformStamped.transform.rotation.x = quat_init.x();
    static_init_transformStamped.transform.rotation.y = quat_init.y();
    static_init_transformStamped.transform.rotation.z = quat_init.z();
    static_init_transformStamped.transform.rotation.w = quat_init.w();

    
    
     
  }
  static tf::TransformBroadcaster br_ins;
  tf::Transform transform;
  
  q.setRPY(msg->pitch/180*3.141592, msg->roll/180*3.141592,-msg->azimuth/180*3.141592);
  transform.setRotation(q);
  double N,E;
  int Z;
  LatLonToUTMXY(msg->latitude, msg->longitude, 52, E, N);
  transform.setOrigin(tf::Vector3(E, N, 0));
  if (past_time != ros::Time::now())
    br_ins.sendTransform(tf::StampedTransform(transform_init.inverse()*transform, ros::Time::now(), "map", "imu"));
  past_time = ros::Time::now();
}

int main(int argc, char** argv){
  ros::init(argc, argv, "gps_world_tf");
  gps_world_tf_node gps_world_tf = gps_world_tf_node();
  ros::spin();
  return 0;
};
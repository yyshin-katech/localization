#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <ros/ros.h>

#include <mmc_msgs/chassis_msg.h>
#include <mmc_msgs/to_control_team_from_local_msg.h>
#include <mmc_msgs/localization2D_msg.h>
#include <std_msgs/Bool.h>

#include <visualization_msgs/MarkerArray.h>

#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/TimeReference.h>
#include <geometry_msgs/TwistStamped.h>

#include <novatel_gps_msgs/Inspva.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float32.h>

// #include <math.h>
#include <vector>
#include <numeric>

#include <proj.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
// #include <turtlesim/Pose.h>

using namespace std;
using namespace Eigen;

#define PI 3.141592f
#define GRAVITY 9.80655f

ros::Publisher pub1, pub2, pub3, pub4, pub5, pub6;

mmc_msgs::chassis_msg chassis = mmc_msgs::chassis_msg();
sensor_msgs::NavSatFix gps_fix = sensor_msgs::NavSatFix();
geometry_msgs::TwistStamped gps_vel = geometry_msgs::TwistStamped();
mmc_msgs::localization2D_msg gps_fix_2d = mmc_msgs::localization2D_msg();
mmc_msgs::localization2D_msg msg_2d_pub = mmc_msgs::localization2D_msg();

// initializing flags
bool gps_fix_init = false;
bool gps_vel_init = false;
bool gps_init = false;
bool var_init = false;
bool gps_stabilized = false;

bool fix_topic_received = false;

double to_rad = 3.141592 / 180;
double heading_gps;
double heading_gps_from_vel = 0;

double sYawrate;
double sA;

double heading_gps_antenna;

// kalman filter variables
MatrixXd Phi, Gamma, H, Q, R, P;
VectorXd x, z;
int estimation_frequency = 10;
double dt = 1/(double)estimation_frequency;

// moving average variables
vector<double> east_history;
vector<double> north_history;
vector<double> ve_history;
vector<double> vn_history;
vector<double> time_history;

// proj package variables
PJ_CONTEXT *C_proj;
PJ *P_proj;
PJ *norm_proj;
PJ_COORD a, b;

int window_size;
std::string localizing_method;

sensor_msgs::NavSatFix fix_msg;
sensor_msgs::Imu IMU;

geometry_msgs::TransformStamped gps_transformStamped;

int cnt;

tf::Quaternion q;

template <class T>
T sign(T input){
  if(input < 0){
    return T(-1);

  }else if(input > 0){
    return T(1);

  }else{
    return T(0);
  }
}

void chassis_callback(const mmc_msgs::chassis_msg& data){
  if(data.speed > 0){
    chassis = data;

  }else{
    chassis.yawrate = 0;
    chassis.ax = 0;
    chassis.ay = 0;
  }
}

void IMU_callback(const sensor_msgs::Imu& data){
  IMU = data;
  
}


void EQ900_gps_time_callback(const sensor_msgs::TimeReference& data){
  
}


void EQ900_gps_fix_callback(const sensor_msgs::NavSatFix& data){
  fix_topic_received = true;

  gps_fix = data;

  norm_proj = proj_normalize_for_visualization(C_proj, P_proj);

  // proj_destroy(P_proj);
  // P_proj = norm_proj;

  a = proj_coord(data.longitude, data.latitude, 0, 0);
  b = proj_trans(norm_proj, PJ_FWD, a);
  // printf("easting: %.3f, northing: %.3f\n", b.enu.e, b.enu.n);

  // b = proj_trans(P_proj, PJ_INV, b);
  // printf("longitude: %g, latitude: %g\n", b.lp.lam, b.lp.phi);

  // proj_destroy(P_proj);
  // proj_context_destroy(C_proj); /* may be omitted in the single threaded case */

  // cout<<"e = "<<b.enu.e<<endl;
  // cout<<"n = "<<b.enu.n<<endl;
  
  /* Clean up */


  // //////////////////////// TF broadcaster ///////////////////////
  // static tf::TransformBroadcaster br;
  // tf::Transform transform;
  // // transform.setOrigin( tf::Vector3(b.enu.e, b.enu.n, 0.0) );
  // transform.setOrigin( tf::Vector3(msg_2d_pub.east, msg_2d_pub.north, 0.0) );
  // tf::Quaternion q;
  // q.setRPY(0, 0, msg_2d_pub.yaw);
  // transform.setRotation(q);
  // br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "gps"));


  gps_fix_2d.east = b.enu.e;
  gps_fix_2d.north = b.enu.n;

  if(gps_fix_init == false){
    gps_fix_init = true;
    gps_init = gps_fix_init && gps_vel_init;
  }

  if(isnan(gps_fix.latitude) != 0 || isnan(gps_fix.longitude) != 0){
    gps_init = false;
    gps_fix_init = false;
  }

  mmc_msgs::localization2D_msg test;
  test.east  = b.enu.e;
  test.north = b.enu.n;
  // test.yaw = atan2(gps_vel.twist.linear.y, gps_vel.twist.linear.x); // 찬욱이 코드 없던 시절

  

  int min_idx;
  ArrayXd heading_candidate(5);
  heading_candidate << heading_gps, heading_gps-2*PI, heading_gps+2*PI, heading_gps-4*PI, heading_gps+4*PI;
  (heading_candidate - msg_2d_pub.yaw).abs().minCoeff(&min_idx);
  heading_gps = heading_candidate(min_idx);



  test.yaw = heading_gps;
  test.sig_yaw = IMU.angular_velocity.z;


  

  /// 헤딩 360 예외 처리 코드 2번 돌린다
  if(test.yaw < -2*PI){
    test.yaw += 2*PI;
    
  }else if(test.yaw > 2*PI){
    test.yaw += -2*PI;
  }

  if(test.yaw < -2*PI){
    test.yaw += 2*PI;
    
  }else if(test.yaw > 2*PI){
    test.yaw += -2*PI;
  } 

  pub2.publish(test);
}


void EQ900_gps_vel_callback(const geometry_msgs::TwistStamped& data){
  

  // 특정 속도 이상일 때에만 update를 한다.
  gps_stabilized = sqrt( pow(data.twist.linear.x, 2) + pow(data.twist.linear.y, 2)) > 0.7; 

  if(gps_stabilized == true){
    gps_vel = data;
    heading_gps_antenna = atan2(data.twist.linear.y, data.twist.linear.x);
  }
  

  // if(gps_stabilized == false){
  //   gps_vel.twist.linear.x = 0;
  //   gps_vel.twist.linear.y = 0;
  // }

  if(gps_vel_init == false){
    gps_vel_init = true;
    gps_init = gps_fix_init && gps_vel_init;
  }
  
  if(isnan(gps_vel.twist.linear.x) != 0 || isnan(gps_vel.twist.linear.y) != 0){
    gps_init = false;
    gps_vel_init = false;
  }


}


void inspvaCallback(const novatel_gps_msgs::InspvaConstPtr& msg){
  // cout<<"fix_topic_received = "<<fix_topic_received<<endl;
  if(fix_topic_received == false){
    gps_fix.header = msg->header;
    gps_fix.latitude = msg->latitude;
    gps_fix.longitude = msg->longitude;
    
    norm_proj = proj_normalize_for_visualization(C_proj, P_proj);
    a = proj_coord(msg->longitude, msg->latitude, 0, 0);
    b = proj_trans(norm_proj, PJ_FWD, a);

    gps_fix_2d.east = b.enu.e;
    gps_fix_2d.north = b.enu.n;

    if(gps_fix_init == false){
      gps_fix_init = true;
      gps_init = gps_fix_init && gps_vel_init;
    }

    if(isnan(gps_fix.latitude) != 0 || isnan(gps_fix.longitude) != 0){
      gps_init = false;
      gps_fix_init = false;
    }

    mmc_msgs::localization2D_msg test;
    test.east  = b.enu.e;
    test.north = b.enu.n;
    // test.yaw = atan2(gps_vel.twist.linear.y, gps_vel.twist.linear.x); // 찬욱이 코드 없던 시절

    

    int min_idx;
    ArrayXd heading_candidate(5);
    heading_candidate << heading_gps, heading_gps-2*PI, heading_gps+2*PI, heading_gps-4*PI, heading_gps+4*PI;
    (heading_candidate - msg_2d_pub.yaw).abs().minCoeff(&min_idx);
    heading_gps = heading_candidate(min_idx);



    test.yaw = heading_gps;
    test.sig_yaw = IMU.angular_velocity.z;


    

    /// 헤딩 360 예외 처리 코드 2번 돌린다
    if(test.yaw < -2*PI){
      test.yaw += 2*PI;
      
    }else if(test.yaw > 2*PI){
      test.yaw += -2*PI;
    }

    if(test.yaw < -2*PI){
      test.yaw += 2*PI;
      
    }else if(test.yaw > 2*PI){
      test.yaw += -2*PI;
    } 

    pub2.publish(test);
  }
  

  gps_stabilized = true;

  gps_vel.header.stamp = ros::Time::now();
  gps_vel.twist.linear.x = msg->east_velocity;
  gps_vel.twist.linear.y = msg->north_velocity;

  if(gps_vel_init == false){
    gps_vel_init = true;
    gps_init = gps_fix_init && gps_vel_init;
  }
  
  if(isnan(gps_vel.twist.linear.x) != 0 || isnan(gps_vel.twist.linear.y) != 0){
    gps_init = false;
    gps_vel_init = false;
  }
  
  
  // heading_gps = 0.5f*PI - msg->azimuth/180.f*PI; // IMU yaw

  if(isnan(gps_vel.twist.linear.x) == 0 && isnan(gps_vel.twist.linear.y) == 0){
    heading_gps_from_vel = atan2(msg->north_velocity, msg->east_velocity);
  }

  heading_gps = heading_gps_from_vel; // IMU yaw
  // heading_gps = heading_gps_antenna; // Antenna yaw

  std_msgs::Float32 heading_float;  
  heading_float.data = heading_gps_from_vel;
  pub5.publish(heading_float);

  // tf::Transform transform;
  // q.setRPY(msg->pitch/180*3.141592, msg->roll/180*3.141592,1.57-msg->azimuth/180*3.141592);
  // gps_transformStamped.transform.rotation.x = q.x();
  // gps_transformStamped.transform.rotation.y = q.y();
  // gps_transformStamped.transform.rotation.z = q.z();
  // gps_transformStamped.transform.rotation.w = q.w();
  // transform.setRotation(q);

}




void kalman_filter_CV_without_yaw(){
  if(gps_init == true && var_init == false){
    // state = [X Y Vx Vy]
    int state_num = 4;

    Phi.setZero(state_num, state_num);
    Phi << 1, 0, dt,  0,
           0, 1,  0, dt,
           0, 0,  1,  0,
           0, 0,  0,  1;

    H.setZero(state_num, state_num);
    H << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1;

    P.setZero(Phi.innerSize(), Phi.outerSize());
    P << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1;

    Q.setZero(Phi.innerSize(), Phi.outerSize());
    Q << 0.01,    0,   0,   0,
            0, 0.01,   0,   0,
            0,    0, 0.2,   0,
            0,    0,   0, 0.2;

    R.setZero(H.innerSize(), H.innerSize());
    R << 1.5,   0,   0,   0,
           0, 1.5,   0,   0,
           0,   0, 6.0,   0,
           0,   0,   0, 6.0;

    x.setZero(Phi.innerSize(), 1);
    x << gps_fix_2d.east,
         gps_fix_2d.north,
         gps_vel.twist.linear.x,
         gps_vel.twist.linear.y;

    var_init = true;

  }else if(gps_init == true && var_init == true){ 
    // Prediction
    VectorXd xp;
    MatrixXd Pp;
    xp.setZero(Phi.innerSize(), 1);
    Pp.setZero(Phi.innerSize(), Phi.outerSize());

    xp = Phi*x;
    Pp = Phi*P*Phi.transpose().eval() + Q;
    
    // Kalman Gain
    MatrixXd K;
    K.setZero(Phi.innerSize(), Phi.outerSize());
    K = Pp*H.transpose().eval() * (H*Pp*H.transpose().eval() + R).inverse();
    
    // Correction
    VectorXd z;
    z.setZero(R.innerSize(), 1);
    z << gps_fix_2d.east,
         gps_fix_2d.north,
         gps_vel.twist.linear.x,
         gps_vel.twist.linear.y;

    x = xp + K*(z - H*xp);
    P = Pp - K*H*Pp;

    msg_2d_pub.localizer_type  = "KF_CV";
    msg_2d_pub.gps_initialized = gps_init;
    msg_2d_pub.time            = ros::Time::now();
    msg_2d_pub.east            = x(0);
    msg_2d_pub.north           = x(1);
    // msg_2d_pub.yaw             = atan2(x(3), x(2));
    msg_2d_pub.yaw             = heading_gps;
    msg_2d_pub.v               = sqrt(x(2)*x(2) + x(3)*x(3));
  }
}


void kalman_filter_CV_with_yaw(){
  if(gps_init == true && var_init == false){
    // state = [X Y Vx Vy Yaw Yawrate]
    int state_num = 6;

    Phi.setZero(state_num, state_num);
    Phi << 1, 0, dt,  0, 0,  0,
           0, 1,  0, dt, 0,  0,
           0, 0,  1,  0, 0,  0,
           0, 0,  0,  1, 0,  0,
           0, 0,  0,  0, 1, dt,
           0, 0,  0,  0, 0,  1;

    H.setZero(state_num, state_num);
    H << 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1;

    P.setZero(Phi.innerSize(), Phi.outerSize());
    P << 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1;

    Q.setZero(Phi.innerSize(), Phi.outerSize());
    Q << 0.01,    0,   0,   0,          0,        0,
            0, 0.01,   0,   0,          0,        0,
            0,    0, 0.2,   0,          0,        0,
            0,    0,   0, 0.2,          0,        0,
            0,    0,   0,   0, 0.1*to_rad,        0,
            0,    0,   0,   0,          0, 1*to_rad;

    R.setZero(H.innerSize(), H.innerSize());
    R << 1.5,   0,   0,   0,          0,        0,
           0, 1.5,   0,   0,          0,        0,
           0,   0, 7.0,   0,          0,        0,
           0,   0,   0, 7.0,          0,        0,
           0,   0,   0,   0, 4.0*to_rad,        0,
           0,   0,   0,   0,          0, 15*to_rad;

    x.setZero(Phi.innerSize(), 1);
    // x << gps_fix_2d.east,
    //      gps_fix_2d.north,
    //      gps_vel.twist.linear.x,
    //      gps_vel.twist.linear.y,
    //      atan2(gps_vel.twist.linear.y, gps_vel.twist.linear.x),
    //      chassis.yawrate;

    x << gps_fix_2d.east,
         gps_fix_2d.north,
         gps_vel.twist.linear.x,
         gps_vel.twist.linear.y,
         heading_gps,
         chassis.yawrate;

    var_init = true;

  }else if(gps_init == true && var_init == true){ 
    // Prediction
    VectorXd xp;
    MatrixXd Pp;
    xp.setZero(Phi.innerSize(), 1);
    Pp.setZero(Phi.innerSize(), Phi.outerSize());

    xp = Phi*x;
    Pp = Phi*P*Phi.transpose().eval() + Q;
    
    // Kalman Gain
    MatrixXd K;
    K.setZero(Phi.innerSize(), Phi.outerSize());
    K = Pp*H.transpose().eval() * (H*Pp*H.transpose().eval() + R).inverse();
    
    // Correction
    VectorXd z;
    z.setZero(R.innerSize(), 1);
    // z << gps_fix_2d.east,
    //      gps_fix_2d.north,
    //      gps_vel.twist.linear.x,
    //      gps_vel.twist.linear.y,
    //      atan2(gps_vel.twist.linear.y, gps_vel.twist.linear.x),
    //      chassis.yawrate;

    z << gps_fix_2d.east,
         gps_fix_2d.north,
         gps_vel.twist.linear.x,
         gps_vel.twist.linear.y,
         heading_gps,
         chassis.yawrate;

    x = xp + K*(z - H*xp);
    P = Pp - K*H*Pp;

    msg_2d_pub.localizer_type  = "KF_CV_yaw_simple";
    msg_2d_pub.gps_initialized = gps_init;
    msg_2d_pub.time            = ros::Time::now();
    msg_2d_pub.east            = x(0);
    msg_2d_pub.north           = x(1);
    msg_2d_pub.yaw             = x(4);
    msg_2d_pub.v               = sqrt(x(2)*x(2) + x(3)*x(3));
  }
}


void kalman_filter_CTRA(){
  // cout<<setprecision(8)<<endl;
  if(gps_init == true && var_init == false){
    // state = [X Y v a Yaw Yawrate]
    int state_num = 6;

    Phi.setZero(state_num, state_num);

    H.setZero(state_num, state_num);
    H << 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1;

    P.setZero(Phi.innerSize(), Phi.outerSize());
    P << 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1;

    Q.setZero(Phi.innerSize(), Phi.outerSize());
    // Q << 0.04,    0,   0,   0,          0,        0,
    //         0, 0.04,   0,   0,          0,        0,
    //         0,    0, 0.8,   0,          0,        0,
    //         0,    0,   0, 8.0,          0,        0,
    //         0,    0,   0,   0, 0.1*to_rad,        0,
    //         0,    0,   0,   0,          0, 2*to_rad;

    // double sXY      = 0.5*(0.4*9.81)*dt*dt; 
    // double sV       = (0.4*9.81)*dt; 
    // double sA       = (0.4*9.81);
    // double sYaw     = 5.0*to_rad*dt; 
    // double sYawrate = 5.0*to_rad;

    double limit_gravity = 0.25;
    double limit_yawrate_deg = 4;

    sA       = (limit_gravity*GRAVITY);
    sYawrate = limit_yawrate_deg*to_rad; 
    
    Q.setZero(state_num, state_num);
    // Q << pow(sXY,2),          0,         0,         0,           0,               0,
    //               0, pow(sXY,2),         0,         0,           0,               0,
    //               0,          0, pow(sV,2),         0,           0,               0,
    //               0,          0,         0, pow(sA,2),           0,               0,
    //               0,          0,         0,         0, pow(sYaw,2),               0,
    //               0,          0,         0,         0,           0, pow(sYawrate,2);

    

    R.setZero(H.innerSize(), H.innerSize());
    R << 0.25,    0,   0,     0,          0,         0,
            0, 0.25,   0,     0,          0,         0,
            0,    0, 1.0,     0,          0,         0,
            0,    0,   0,   5.0,          0,         0,
            0,    0,   0,     0, 0.5*to_rad,         0,
            0,    0,   0,     0,          0, 1.5*to_rad;
    
    // R*=1;

    x.setZero(Phi.innerSize(), 1);
    // x << gps_fix_2d.east,
    //      gps_fix_2d.north,
    //      chassis.speed,
    //      sqrt(pow(chassis.ax, 2) + pow(chassis.ay, 2)),         
    //      atan2(gps_vel.twist.linear.y, gps_vel.twist.linear.x),
    //      chassis.yawrate;

    // x << gps_fix_2d.east,
    //      gps_fix_2d.north,
    //      chassis.speed,
    //      sqrt(pow(chassis.ax, 2) + pow(chassis.ay, 2)),         
    //      heading_gps,
    //      chassis.yawrate;

    x << gps_fix_2d.east,
         gps_fix_2d.north,
         chassis.speed,
         sqrt(pow(chassis.ax, 2) + pow(chassis.ay, 2)),         
         heading_gps,
         IMU.angular_velocity.z;

    // x << IMU.,
    //      gps_fix_2d.north,
    //      chassis.speed,
    //      sqrt(pow(chassis.ax, 2) + pow(chassis.ay, 2)),         
    //      heading_gps,
    //      IMU.angular_velocity.z;


    var_init = true;

  }else if(gps_init == true && var_init == true && gps_stabilized == true){

    // for(int i=0; i!=Phi.innerSize(); i++){
    //   cout<<"x "<<i<<" = "<<x[i]<<endl;
    // }

    dt = (ros::Time::now() - msg_2d_pub.time).toSec();
    cout<<"dt = "<<dt*1000<<" ms"<<endl;

    if(dt > 1 || dt < 0){
      dt = 0;
    }

    // Prediction
    VectorXd xp;
    MatrixXd Pp;
    xp.setZero(Phi.innerSize(), 1);
    Pp.setZero(Phi.innerSize(), Phi.outerSize());

    double X   = x(0);
    double Y   = x(1);
    double v   = x(2);
    double a   = x(3);
    double yaw = x(4);
    double r   = x(5);

    // if(abs(chassis.yawrate) <= 0.005){
    if(abs(IMU.angular_velocity.z) <= 0.0001){
      r = 0;
    }

    
    if(abs(r) > 0.0001){
      
      cout<<"enough yawrate"<<endl;

      xp << X + (1/(r*r)) * (( v*r + a*r*dt) * sin(yaw+r*dt) + a*cos(yaw+r*dt) - v*r*sin(yaw) - a*cos(yaw)),
            Y + (1/(r*r)) * ((-v*r - a*r*dt) * cos(yaw+r*dt) + a*sin(yaw+r*dt) + v*r*cos(yaw) - a*sin(yaw)),
            v + a*dt,
            a,
            yaw + r*dt,
            r;

      // double a13, a14, a15, a16, a23, a24, a25, a26;

      // a13 = (-r*sin(yaw) + r*sin(dt*r+yaw))/(r*r);
      // a14 = ( dt*r*sin(dt*r+yaw) - cos(yaw) + cos(dt*r+yaw))/(r*r);
      // a15 = (-r*v*cos(yaw) + a*sin(yaw) - a*sin(dt*r+yaw) + (dt*r*a+r*v)*cos(dt*r+yaw))/(r*r);
      // a16 = (-dt*a*sin(dt*r+yaw) + dt*( dt*r*a+r*v)*cos(dt*r+yaw) - v*sin(yaw) + ( dt*a+v)*sin(dt*r+yaw))/(r*r) - \
      //         2*(-r*v*sin(yaw) - a*cos(yaw) + a*cos(dt*r+yaw) + ( dt*r*a+r*v)*sin(dt*r+yaw))/(r*r*r);

      // a23 = ( r*cos(yaw) - r*cos(dt*r+yaw))/(r*r);
      // a24 = (-dt*r*cos(dt*r+yaw) - sin(yaw) + sin(dt*r+yaw))/(r*r);
      // a25 = (-r*v*sin(yaw) - a*cos(yaw) + a*cos(dt*r+yaw) - (-dt*r*a-r*v) * sin(dt*r + yaw))/(r*r);
      // a26 = ( dt*a*cos(dt*r+yaw) - dt*(-dt*r*a-r*v)*sin(dt*r+yaw) + v*cos(yaw) + (-dt*a-v)*cos(dt*r+yaw))/(r*r) - \
      //         2*( r*v*cos(yaw) - a*sin(yaw) + a*sin(dt*r+yaw) + (-dt*r*a-r*v)*cos(dt*r+yaw))/(r*r*r);

      // Phi << 1, 0, a13, a14, a15, a16,
      //        0, 1, a23, a24, a25, a26,
      //        0, 0,   1,  dt,   0,   0,
      //        0, 0,   0,   1,   0,   0,
      //        0, 0,   0,   0,   1,  dt,
      //        0, 0,   0,   0,   0,   1;

    }else{

      cout<<"NOT enough yawrate"<<endl;
      xp << X + v*cos(yaw)*dt + 0.5*a*cos(yaw)*pow(dt,2),
            Y + v*sin(yaw)*dt + 0.5*a*sin(yaw)*pow(dt,2),
            v + a*dt,
            a,
            yaw + r*dt,
            r;
    }
    
    Q << pow(dt,5)*(0.05*pow(sYawrate,2)*pow(v,2)*pow(sin(yaw),2) + 0.05*pow(sA,2)*pow(cos(yaw),2)), pow(dt,5)*(-0.05*pow(sYawrate,2)*pow(v,2)*sin(yaw)*cos(yaw) + 0.05*pow(sA,2)*sin(yaw)*cos(yaw)), 0.125*pow(dt,4)*pow(sA,2)*cos(yaw), 0.1667*pow(dt,3)*pow(sA,2)*cos(yaw), -0.125*pow(dt,4)*pow(sYawrate,2)*v*sin(yaw), -0.1667*pow(dt,3)*pow(sYawrate,2)*v*sin(yaw),
         pow(dt,5)*(-0.05*pow(sYawrate,2)*pow(v,2)*sin(yaw)*cos(yaw) + 0.05*pow(sA,2)*sin(yaw)*cos(yaw)), pow(dt,5)*(0.05*pow(sYawrate,2)*pow(v,2)*pow(cos(yaw),2) + 0.05*pow(sA,2)*pow(sin(yaw),2)), 0.125*pow(dt,4)*pow(sA,2)*sin(yaw), 0.1667*pow(dt,3)*pow(sA,2)*sin(yaw), 0.125*pow(dt,4)*pow(sYawrate,2)*v*cos(yaw), 0.1667*pow(dt,3)*pow(sYawrate,2)*v*cos(yaw),
         0.125*pow(dt,4)*pow(sA,2)*cos(yaw), 0.125*pow(dt,4)*pow(sA,2)*sin(yaw), pow(dt,3)*pow(sA,2)/3, pow(dt,2)*pow(sA,2)/2, 0, 0,
         0.1667*pow(dt,3)*pow(sA,2)*cos(yaw), 0.1667*pow(dt,3)*pow(sA,2)*sin(yaw), pow(dt,2)*pow(sA,2)/2, dt*pow(sA,2), 0, 0,
         -0.125*pow(dt,4)*pow(sYawrate,2)*v*sin(yaw), 0.125*pow(dt,4)*pow(sYawrate,2)*v*cos(yaw), 0, 0, pow(dt,3)*pow(sYawrate,2)/3, pow(dt,2)*pow(sYawrate,2)/2,
         -0.1667*pow(dt,3)*pow(sYawrate,2)*v*sin(yaw), 0.1667*pow(dt,3)*pow(sYawrate,2)*v*cos(yaw), 0, 0, pow(dt,2)*pow(sYawrate,2)/2, dt*pow(sYawrate,2);
         
    // cout<<"Q"<<endl;
    // cout<<Q<<endl;

    double a13, a14, a15, a16, a23, a24, a25, a26;
    a13 = dt*cos(yaw);
    a14 = 0.5*dt*dt*cos(yaw);
    a15 = -dt*v*sin(yaw);
    a16 = -0.5*dt*dt*v*sin(yaw);

    a23 = dt*sin(yaw);
    a24 = 0.5*dt*dt*sin(yaw);
    a25 = dt*v*cos(yaw);
    a26 = 0.5*dt*dt*v*cos(yaw);

    Phi << 1, 0, a13, a14, a15, a16,
           0, 1, a23, a24, a25, a26,
           0, 0,   1,  dt,   0,   0,
           0, 0,   0,   1,   0,   0,
           0, 0,   0,   0,   1,  dt,
           0, 0,   0,   0,   0,   1;

    Pp = Phi*P*Phi.transpose().eval() + Q;

    // Kalman Gain
    MatrixXd K;
    K.setZero(Phi.innerSize(), Phi.outerSize());
    K = Pp*H.transpose().eval() * (H*Pp*H.transpose().eval() + R).inverse();
    
    // heading measurement setup(현재 헤딩과 2pi 차이나면 한바퀴 돌려줘라)
    ArrayXd heading_candidate(5);

    // double heading_meas = atan2(gps_vel.twist.linear.y, gps_vel.twist.linear.x);
    double heading_meas = heading_gps;
    // cout<<"heading_stat = "<<yaw<<endl;
    // cout<<"heading_meas = "<<heading_meas<<endl;
    int min_idx;
    heading_candidate << heading_meas, heading_meas-2*PI, heading_meas+2*PI, heading_meas-4*PI, heading_meas+4*PI;
    (heading_candidate - yaw).abs().minCoeff(&min_idx);
    heading_meas = heading_candidate(min_idx);
    // cout<<"heading_meas = "<<heading_meas<<endl;

    double dt_sync_gps = (ros::Time::now() - gps_fix.header.stamp).toSec(); // dy_sync는 correction 순간에 gps measure 와 state 간의 시간차이

    // cout<<"dt_sync_gps = "<<dt_sync_gps<<endl;
    // cout<<"dt_sync_gps*chassis.speed*cos(heading_meas) = "<<dt_sync_gps*chassis.speed*cos(heading_meas)<<endl;

    double dt_sync_chassis = (ros::Time::now() - chassis.time).toSec();
    VectorXd z;
    z.setZero(R.innerSize(), 1);
    // z << gps_fix_2d.east  + dt_sync_gps*chassis.speed*cos(heading_meas) + 0.5*dt_sync_gps*dt_sync_gps*chassis.ax*cos(heading_meas),
    //      gps_fix_2d.north + dt_sync_gps*chassis.speed*sin(heading_meas) + 0.5*dt_sync_gps*dt_sync_gps*chassis.ax*sin(heading_meas),
    //      chassis.speed + dt_sync_pub*chassis.ax,
    //      sqrt(pow(chassis.ax, 2) + pow(chassis.ay, 2)),
    //      heading_meas + dt_sync_pub*(chassis.yawrate),
    //      chassis.yawrate;

    z << gps_fix_2d.east + dt_sync_gps*chassis.speed*cos(yaw),
         gps_fix_2d.north + dt_sync_gps*chassis.speed*sin(yaw),
         chassis.speed,
         sqrt(pow(chassis.ax, 2) + pow(chassis.ay, 2)),
         heading_meas,
         IMU.angular_velocity.z;

    // Correction
    x = xp + K*(z - H*xp);
    P = Pp - K*H*Pp;

    // cout<<P<<endl;
    
    // cout<<"pred X = "<<xp(0)<<" Y = "<<xp(1)<<endl;
    // cout<<"meas X = "<<z(0)<<" Y = "<<z(1)<<endl;
    // cout<<"stat X = "<<x(0)<<" Y = "<<x(1)<<endl;

    // cout<<"pred yaw = "<<xp(4)<<endl;
    // cout<<"meas yaw = "<<z(4)<<endl;
    // cout<<"stat yaw = "<<x(4)<<endl;

    msg_2d_pub.localizer_type  = "KF_CTRA";
    msg_2d_pub.gps_initialized = gps_init;
    msg_2d_pub.time            = ros::Time::now();
    msg_2d_pub.east            = x(0);
    msg_2d_pub.north           = x(1);
    msg_2d_pub.v               = x(2);
    msg_2d_pub.yaw             = x(4);
    msg_2d_pub.sig_yaw         = x(5); // yaw_rate

    /// 헤딩 360 예외 처리 코드 2번 돌린다
    if(msg_2d_pub.yaw < -2*PI){
      msg_2d_pub.yaw += 2*PI;
      
    }else if(msg_2d_pub.yaw > 2*PI){
      msg_2d_pub.yaw += -2*PI;
    }

    if(msg_2d_pub.yaw < -2*PI){
      msg_2d_pub.yaw += 2*PI;
      
    }else if(msg_2d_pub.yaw > 2*PI){
      msg_2d_pub.yaw += -2*PI;
    }

    // if(chassis.speed <= 1.5){
    //   x(3) = 0;
    //   x(5) = 0;
    // }
    
  }else if(gps_init == true && var_init == true && gps_stabilized == false){
    msg_2d_pub.time            = ros::Time::now();
  }
}


void moving_average(){
  if(gps_init == true && var_init == false){

    if(east_history.size() == 0 || north_history.size() == 0){

      for(int i=0; i!=window_size; i++){
        east_history.push_back(gps_fix_2d.east);
        north_history.push_back(gps_fix_2d.north);
        ve_history.push_back(gps_vel.twist.linear.x);
        vn_history.push_back(gps_vel.twist.linear.y);
        // time_history.push_back(gps_fix.header.stamp.toSec());
        time_history.push_back(ros::Time::now().toSec());
      }

      var_init = true;
    }

  }else if(gps_init == true && var_init == true && gps_stabilized == true){

    if(east_history.size() != 0 && north_history.size() != 0){

      for(int i=0; i!=window_size-1; i++){
        east_history[i] = east_history[i+1];
        north_history[i] = north_history[i+1];
        ve_history[i] = ve_history[i+1];
        vn_history[i] = vn_history[i+1];
        time_history[i] = time_history[i+1];
      }

      east_history.back() = gps_fix_2d.east;
      north_history.back() = gps_fix_2d.north;
      ve_history.back() = gps_vel.twist.linear.x;
      vn_history.back() = gps_vel.twist.linear.y;
      // time_history.back() = gps_fix.header.stamp.toSec();
      time_history.back() = ros::Time::now().toSec();

      
    }else{
      var_init = false;
    }

  }else if(gps_init == true && var_init == true && gps_stabilized == false){

    for(int i=0; i!=window_size-1; i++){
        time_history[i] = time_history[i+1];
      }

    time_history.back() = ros::Time::now().toSec();
  }

  double east      = accumulate(east_history.begin(),  east_history.end(),  0.0) / window_size;
  double north     = accumulate(north_history.begin(), north_history.end(), 0.0) / window_size;
  double ve        = accumulate(ve_history.begin(),    ve_history.end(),    0.0) / window_size;
  double vn        = accumulate(vn_history.begin(),    vn_history.end(),    0.0) / window_size;
  double time_aver = accumulate(time_history.begin(),  time_history.end(),  0.0) / window_size;

  // cout<<"ros::Time::now().toSec()-time_aver "<<ros::Time::now().toSec()-time_aver<<endl;

  // double east      = accumulate(east_history.begin(),  east_history.end(),  0.0);
  // double north     = accumulate(north_history.begin(), north_history.end(), 0.0);
  // double ve        = accumulate(ve_history.begin(),    ve_history.end(),    0.0);
  // double vn        = accumulate(vn_history.begin(),    vn_history.end(),    0.0);
  // double time_aver = accumulate(time_history.begin(),  time_history.end(),  0.0);

  // east      /= window_size;
  // north     /= window_size;
  // ve        /= window_size;
  // vn        /= window_size;
  // time_aver /= window_size;

  msg_2d_pub.localizer_type  = "Moving_average";
  msg_2d_pub.gps_initialized = gps_init;
  msg_2d_pub.east            = east;
  msg_2d_pub.north           = north;
  // msg_2d_pub.yaw             = atan2(vn, ve);
  msg_2d_pub.yaw             = heading_gps;
  msg_2d_pub.v               = sqrt(ve*ve + vn*vn);
  msg_2d_pub.time.fromSec(time_aver);
}


void loop(){
  ros::Rate rate(estimation_frequency);
  cout<<"Localizing method is ..."<<localizing_method<<endl;

  while(ros::ok()){

    if(gps_init == true && (ros::Time::now()-gps_fix_2d.time).toSec() < -5){
      gps_init = false;
      gps_fix_init = false;
      gps_vel_init = false;
    }



    if(localizing_method == "KF_CTRA"){
      kalman_filter_CTRA();

    }else if(localizing_method == "KF_CV_yaw"){
      kalman_filter_CV_with_yaw();
      
    }else if(localizing_method == "KF_CV"){
      kalman_filter_CV_without_yaw();

    }else if(localizing_method == "Moving_Average"){
      moving_average();
    
    }else{
      kalman_filter_CTRA();
    }

    
    // cout<<"localizing_method = "<<localizing_method<<endl;

    // cout<<"msg_2d_pub.east  = "<<msg_2d_pub.east<<endl;
    // cout<<"msg_2d_pub.north = "<<msg_2d_pub.north<<endl;
    // cout<<"gps_init = "<<gps_init<<endl;

    // msg_2d_pub.yaw = msg_2d_pub.yaw*1000;
    // msg_2d_pub.yaw = round(msg_2d_pub.yaw);
    // msg_2d_pub.yaw = msg_2d_pub.yaw*0.001;

    if(msg_2d_pub.east != 0 && msg_2d_pub.north != 0){
      // cout<<"good"<<endl;
      pub1.publish(msg_2d_pub);
    }

    //////////////////////// TF broadcaster ///////////////////////
    // static tf::TransformBroadcaster br;
    // tf::Transform transform;
    // // transform.setOrigin( tf::Vector3(msg_2d_pub.east, msg_2d_pub.north, 0.0) );
    // transform.setOrigin( tf::Vector3(msg_2d_pub.east, msg_2d_pub.north, 1.5) );
    // tf::Quaternion q;
    
    // q.setRPY(0, 0, msg_2d_pub.yaw);
    // transform.setRotation(q);
    // // br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "gps"));
    // // br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "gps_fix_ref"));
    // // br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "gps"));
    // br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "epsg5179", "gps"));


    static tf2_ros::TransformBroadcaster br;
    tf::Transform transform;

    gps_transformStamped.header.stamp = msg_2d_pub.time;
    gps_transformStamped.header.frame_id = "epsg5179";
    gps_transformStamped.child_frame_id = "gps";
    gps_transformStamped.transform.translation.x = msg_2d_pub.east;
    gps_transformStamped.transform.translation.y = msg_2d_pub.north;
    gps_transformStamped.transform.translation.z = 1.5; // 왜 1.5?

    q.setRPY(0, 0, msg_2d_pub.yaw);
    gps_transformStamped.transform.rotation.x = q.x();
    gps_transformStamped.transform.rotation.y = q.y();
    gps_transformStamped.transform.rotation.z = q.z();
    gps_transformStamped.transform.rotation.w = q.w();
    transform.setRotation(q);

    br.sendTransform(gps_transformStamped);

    // if (cnt%2 == 0){
    //   br.sendTransform(gps_transformStamped);
    //   cnt++;

    // }else{
    //   cnt = 0;
    // }

    
    rate.sleep();
  }
}





int main(int argc, char **argv){
  ros::init(argc, argv, "gps_to_tf");
  ros::NodeHandle node("~");

  ros::AsyncSpinner spinner(0);
  spinner.start();

  ros::param::get("gps_moving_average_window_size", window_size);
  ros::param::get("gps_localizing_method_EQ", localizing_method);

  C_proj = proj_context_create();
  // P_proj = proj_create_crs_to_crs (C_proj, "EPSG:4326", "epsg:5186", NULL);
  P_proj = proj_create_crs_to_crs (C_proj, "EPSG:4326", "epsg:5179", NULL);

  // cout<<"windows_size = "<<window_size<<endl;
  // cout<<"localizing_method = "<<localizing_method<<endl;

  std::cout<<setprecision(8);

  ros::Subscriber sub3 = node.subscribe("/sensors/chassis",              1, &chassis_callback);
  ros::Subscriber sub4 = node.subscribe("/sensors/gps/fix",              1, &EQ900_gps_fix_callback);
  ros::Subscriber sub8 = node.subscribe("/sensors/gps/vel",              1, &EQ900_gps_vel_callback);
  ros::Subscriber sub5 = node.subscribe("/sensors/gps/inspva",           1, &inspvaCallback);
  ros::Subscriber sub6 = node.subscribe("/sensors/gps/time_reference",   1, &EQ900_gps_time_callback);
  ros::Subscriber sub7 = node.subscribe("/sensors/gps/imu",              1, &IMU_callback);
 
  pub1 = node.advertise<mmc_msgs::localization2D_msg>("/localization/pose_2d_gps",  1);
  pub2 = node.advertise<mmc_msgs::localization2D_msg>("/localization/pose_2d_raw",  1);
  // pub3 = node.advertise<std_msgs::Bool>("/yawrate_choongboon", 1);
  pub4 = node.advertise<sensor_msgs::NavSatFix>("/sensors/gps/xyzinit", 10000, true);
  pub5 = node.advertise<std_msgs::Float32>("/monitor_EQ_yaw_from_vel",1);

  loop();
  
  ros::waitForShutdown();   
}



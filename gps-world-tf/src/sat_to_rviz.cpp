#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <curl/curl.h>
#include <mmc_msgs/localization2D_msg.h>

size_t write_data(char *ptr, size_t size, size_t nmemb, void *userdata)
{
    std::vector<uchar> *stream = (std::vector<uchar>*)userdata;
    size_t count = size * nmemb;
    stream->insert(stream->end(), ptr, ptr + count);
    return count;
}

cv::Mat curlImg(const std::string img_url, int timeout=10)
{
    ROS_WARN("Map Loading...");
    std::vector<uchar> stream;
    CURL *curl = curl_easy_init();
    auto url = img_url.c_str();
    curl_easy_setopt(curl, CURLOPT_URL, url); 
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream); 
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);
    CURLcode res = curl_easy_perform(curl);
    if (res > 0)
        throw(res);
    curl_easy_cleanup(curl);
    ROS_WARN("Map Load success!");
    return cv::imdecode(stream, -1);
}

class SattoRviz
{
private:
    ros::NodeHandle node;
    ros::Subscriber sub1;
    ros::Publisher pub1;
    ros::Publisher pub2;
    double x,y,v,yaw;
    double x_pub=0;
    double y_pub=0;
    bool initialized = false;

public:
    SattoRviz(/* args */);
    ~SattoRviz();
    double getDistance();
    double offset_x_, offset_y_;
    void PoseCallback(const mmc_msgs::localization2D_msgConstPtr& msg);
    void plot_image(double offset_x = 0.0, double offset_y = 0.0, std::string frame_name="map_sat");
};

SattoRviz::SattoRviz(/* args */)
{
    pub1 = node.advertise<sensor_msgs::Image>("/satellite",10000,true);
    pub2 = node.advertise<sensor_msgs::Image>("/satellite_2",10000,true);
    sub1 = node.subscribe("/localization/pose_2d_gps",10, &SattoRviz::PoseCallback, this);
}

SattoRviz::~SattoRviz()
{
}

void SattoRviz::PoseCallback(const mmc_msgs::localization2D_msgConstPtr& msg)
{
    x = msg->east;
    y = msg->north;
    v = msg->v;
    yaw = msg->yaw;
    
    double v_x_norm, v_y_norm;
    v_x_norm = x-x_pub;
    v_y_norm = y-y_pub;
    if (abs(v_x_norm) > abs(v_y_norm))
    {
        if(v_x_norm > 0)
        {
            offset_x_ = 245.76;
            offset_y_ = 0;
        }
        else
        {
            offset_x_ = -245.76;
            offset_y_ = 0;
        }
    }
    else
    {
        if(v_y_norm > 0)
        {
            offset_x_ = 0;
            offset_y_ = 245.76;
        }
        else
        {
            offset_x_ = 0;
            offset_y_ = -245.76;
        }
    }
    if(!initialized)
    {
        x_pub = x;
        y_pub = y;
        plot_image();
    }
    
}

double SattoRviz::getDistance()
{
    return sqrt(pow(x_pub-x,2) + pow(y_pub-y,2));
}

void SattoRviz::plot_image(double offset_x, double offset_y, std::string frame_name)
{
    sensor_msgs::Image msg = sensor_msgs::Image();
    std::string URL, x_, y_;
    x_ = std::to_string(x+offset_x);
    y_ = std::to_string(y+offset_y);
    URL = "https://naveropenapi.apigw.ntruss.com/map-static/v2/raster?w=1024&h=1024&maptype=satellite_base&level=18&crs=epsg:5179&scale=2&X-NCP-APIGW-API-KEY=xbLCWlYOScxR1nOxQ3x0o6uf2SiMGSLzw4gUHQLP&&X-NCP-APIGW-API-KEY-ID=n1e71r5j8z&center=";
    URL += x_ + "," + y_;
    cv::Mat test1;
    try{
        test1 = curlImg(URL);
        if(!initialized)
            ROS_WARN("Map initialized!");

        initialized = true;
    }
    catch(CURLcode error_)
    {
        ROS_WARN("Map does not loaded");
        std::string string_;
        string_ = std::to_string(error_);
        string_ = "error code is " + string_;
        ROS_WARN(string_.c_str());
        return;
    }
    // cv_ptr->image = test;
    cv_bridge::CvImage cv_brid;
    cv_brid.image = test1;
    cv_brid.encoding = "bgr8";
    cv_brid.header.stamp = ros::Time::now();
    cv_brid.toImageMsg(msg);
    if (frame_name=="map_sat_2")
        pub2.publish(msg);
    else
        pub1.publish(msg);
    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "epsg5179";
    transformStamped.child_frame_id = frame_name;
    tf2::Quaternion quat_init;
    quat_init.setRPY(3.141592, 0, 0);
    transformStamped.transform.rotation.x = quat_init.x();
    transformStamped.transform.rotation.y = quat_init.y();
    transformStamped.transform.rotation.z = quat_init.z();
    transformStamped.transform.rotation.w = quat_init.w();
    transformStamped.transform.translation.x = x+offset_x;
    transformStamped.transform.translation.y = y+offset_y;
    transformStamped.transform.translation.z = 0;
    static tf2_ros::StaticTransformBroadcaster br_map;
    br_map.sendTransform(transformStamped);
    x_pub = x;
    y_pub = y;
}

int main(int argc, char** argv){
  ros::init(argc, argv, "SattoRviz");
  SattoRviz map_to_rviz_ = SattoRviz();
  auto rate = ros::Rate(0.5);
  while(ros::ok())
  {
    ros::spinOnce();
    // std::cout <<"distance is: " << map_to_rviz_.getDistance() <<std::endl;
    if(map_to_rviz_.getDistance() >50)
    {
        map_to_rviz_.plot_image();
        map_to_rviz_.plot_image(map_to_rviz_.offset_x_,map_to_rviz_.offset_y_,"map_sat_2");
    }
    // rate.sleep();
  }
    
  return 0;
};
#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import rospy
import numpy as np

import time
import tf
import matplotlib.pyplot as plt

from pyproj import Transformer

from mmc_msgs.msg import localization2D_msg
from mmc_msgs.msg import to_control_team_from_local_msg
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from filterpy.kalman import KalmanFilter

GPS_EPSG = "epsg:4326" # WSG84
MAP_EPSG = "epsg:5186" # map shapefile epsg

try:
    MAP_EPSG_NUMBER = int(rospy.get_param('EPSG'))
    window_size_rosparam = int(rospy.get_param('gps_moving_average_window_size'))

except:
    MAP_EPSG_NUMBER = 5186
    window_size_rosparam = 5


class Localizer(object):
    def __init__(self):
        # gps
        self.gps_fix_initialized = False
        self.gps_vel_initialized = False
        self.gps_initialized = False
        self.variable_initialized = False

        self.time_gps_fix_last_received = None

        # tracking state
        self.e = 0.0 # east
        self.n = 0.0 # north
        self.v = 0.0 # speed       
        self.ve = 0 
        self.vn = 0
        
        self.dt = 0.05

        self.gps_fix = NavSatFix()
        self.gps_vel = TwistStamped()

        self.time_for_pub = rospy.Time.now()

        self.time_state_last_updated = None

        # tf broadcaster
        self.gps_tf_broadcaster = tf.TransformBroadcaster()

        # measurement noise
        self.e_sigma = 10.0
        self.n_sigma = 10.0
        self.yaw_sigma = np.radians(180.0)
        self.v_sigma = 3

        # moving average for east and north
        self.number_of_moving_average = window_size_rosparam
        self.east_history = []
        self.north_history = []
        self.ve_history = []
        self.vn_history = []
        self.time_stamp_history = []
        self.yaw_smoothed = 0.0

        # GPS 점들로부터 heading 구하기
        self.previous_east = None
        self.previous_north = None

        # 디버깅용 왜 한번씩 튀나?
        self.times = []
        self.easts = []
        self.norths = []
        self.marker_list = []

        self.kf = KalmanFilter(4, 4)

        self.pose_2d_pub = rospy.Publisher('/localization/pose_2d', localization2D_msg, queue_size=1)
        self.gps_ellipse_pub = rospy.Publisher('/localization/rviz/gps_ellipse', Marker, queue_size=1)
        self.gps_ellipse_hisory_pub = rospy.Publisher('/localization/rviz/gps_ellipse_history', MarkerArray, queue_size=1)

        
    def kalman_filter_CV(self):
        if self.gps_initialized == False:
            pass

        elif self.gps_initialized == True and self.variable_initialized == False:
            self.kf = KalmanFilter(4,4)
            self.kf.P = np.eye(4)
            
            self.kf.Q = np.array([[0.01,    0,   0,   0],
                                  [0,    0.01,   0,   0],
                                  [0,    0,    0.2,   0],
                                  [0,    0,      0, 0.2]])

            self.kf.R = np.array([[1.5,   0,   0,   0],
                                  [0,   1.5,   0,   0],
                                  [0,     0, 6.0,   0],
                                  [0,     0,   0, 6.0]])
            
            self.kf.F = np.array([[1, 0, self.dt,       0],\
                                  [0, 1,       0, self.dt],\
                                  [0, 0,       1,       0],\
                                  [0, 0,       0,       1]])
            
            self.kf.H = np.eye(4)

            self.kf.x = np.array([self.east_raw,\
                                  self.north_raw,\
                                  self.gps_vel.twist.linear.x,\
                                  self.gps_vel.twist.linear.y])

            self.variable_initialized = True

        elif self.gps_initialized == True and self.variable_initialized == True:
            z = np.array([self.east_raw,\
                          self.north_raw,\
                          self.gps_vel.twist.linear.x,\
                          self.gps_vel.twist.linear.y])

            self.kf.predict()
            self.kf.update(z)
            
            self.e = self.kf.x[0]
            self.n = self.kf.x[1]
            self.ve = self.kf.x[2]
            self.vn = self.kf.x[3]
            self.time_for_pub = rospy.Time.now()
            self.yaw_smoothed = np.arctan2(self.vn, self.ve)


    def moving_average(self):
        if self.gps_initialized == False:
            pass

        elif self.gps_initialized == True and self.variable_initialized == False:

            if len(self.east_history) == 0 or len(self.north_history) == 0:
                self.east_history       = [self.east_raw]                      * self.number_of_moving_average
                self.north_history      = [self.north_raw]                     * self.number_of_moving_average
                self.ve_history         = [self.gps_vel.twist.linear.x]        * self.number_of_moving_average
                self.vn_history         = [self.gps_vel.twist.linear.y]        * self.number_of_moving_average
                self.time_stamp_history = [self.gps_fix.header.stamp.to_sec()] * self.number_of_moving_average
                self.variable_initialized = True
                
            else:
                pass

        elif self.gps_initialized == True and self.variable_initialized == True:

            if len(self.east_history) != 0 and len(self.north_history) != 0:
                self.east_history.append(self.east_raw)
                self.east_history.pop(0)

                self.north_history.append(self.north_raw)
                self.north_history.pop(0)

                self.ve_history.append(self.gps_vel.twist.linear.x)
                self.ve_history.pop(0)

                self.vn_history.append(self.gps_vel.twist.linear.y)
                self.vn_history.pop(0)

                self.time_stamp_history.append(self.gps_fix.header.stamp.to_sec())
                self.time_stamp_history.pop(0)

            else:
                self.variable_initialized = False

        
        self.e            = sum(self.east_history)       / self.number_of_moving_average
        self.n            = sum(self.north_history)      / self.number_of_moving_average
        self.ve           = sum(self.ve_history)         / self.number_of_moving_average
        self.vn           = sum(self.vn_history)         / self.number_of_moving_average
        time_average      = sum(self.time_stamp_history) / self.number_of_moving_average
        self.yaw_smoothed = np.arctan2(self.vn, self.ve)
        self.time_for_pub = rospy.Time.from_sec(time_average)
        

    def callback_gps_fix(self, msg):
        self.gps_fix = msg

        check_data = [msg.longitude, msg.latitude, msg.position_covariance[0], msg.position_covariance[4]]
        healty = np.all(np.isfinite(check_data)) # 위 값이 Nan이나 Inf가 아니면 healty

        if healty:
            tftf = Transformer.from_crs("epsg:4326", "epsg:5186")
            self.north_raw, self.east_raw = tftf.transform(msg.latitude, msg.longitude)

            east_sigma = np.clip(np.sqrt(msg.position_covariance[0]), 0.2, np.inf)
            north_sigma = np.clip(np.sqrt(msg.position_covariance[4]), 0.2, np.inf)

            self.e_sigma = east_sigma
            self.n_sigma = north_sigma

            if not self.gps_fix_initialized:
                self.time_gps_fix_last_received = msg.header.stamp
                self.gps_fix_initialized = True
                self.gps_initialized = (self.gps_fix_initialized and self.gps_vel_initialized)
                self.time_state_last_updated = rospy.get_rostime()

            if self.gps_initialized:
                self.time_state_last_updated = rospy.get_rostime()
                self.time_gps_fix_last_received = msg.header.stamp # GPS 받은 시간은 이때 이지만 측위에 쓸지는 아직 결정안됨

            self.e_sigma = east_sigma
            self.n_sigma = north_sigma


            # print("0-1 elapsed = %.2f" % ((t1 - t0)*1000.0))
            # print("1-2 elapsed = %.2f" % ((t2 - t1)*1000.0))
            # print("2-3 elapsed = %.2f" % ((t3 - t2)*1000.0))



    def callback_gps_vel(self, msg):
        self.gps_vel = msg

        ve = msg.twist.linear.x
        vn = msg.twist.linear.y

        v = np.sqrt(ve*ve + vn*vn)
        healty = (v > 0.5)

        if healty:
            if not self.gps_vel_initialized:
                self.gps_vel_initialized = True
                self.gps_initialized = (self.gps_fix_initialized and self.gps_vel_initialized)
                self.time_state_last_updated = rospy.get_rostime()

        ## 감속해서 healty가 다시 False로 바뀐 후에 어떻게 할 지 고민해야 함


    def publish(self):
        p = localization2D_msg()
        
        p.localizer_type = "GPS"
        p.gps_initialized = self.gps_initialized
        p.EPSG = MAP_EPSG_NUMBER

        if self.time_gps_fix_last_received != None:
            p.without_gps = (rospy.get_rostime()-self.time_gps_fix_last_received).to_sec()

        p.east = self.e
        p.north = self.n
        p.yaw = self.yaw_smoothed
        p.v = np.sqrt(self.ve*self.ve + self.vn*self.vn)
        p.time = self.time_for_pub

        p.sig_east = self.e_sigma
        p.sig_north = self.n_sigma
        p.sig_yaw = self.yaw_sigma
        p.sig_v = self.v_sigma
        
        self.pose_2d_pub.publish(p)


    def ellipse(self, x0, y0, sigx, sigy):
        # 1sigma = 68%, 2sigma = 95%, 3sigma = 99.7%
        tt = np.linspace(0, 2*np.pi, 20)
        xx = sigx*np.cos(tt) + x0
        yy = sigy*np.sin(tt) + y0
        return xx, yy


    def set_marker_array(self, xx, yy):
        m = Marker()
        m.id = 1
        m.header.frame_id = "world"
        m.type = m.LINE_STRIP
        m.action = m.ADD

        m.scale.x = 0.05

        m.color.r = 123/255.0
        m.color.g = 239/255.0
        m.color.b = 178/255.0
        m.color.a = 0.8

        m.pose.orientation.x = 0
        m.pose.orientation.y = 0
        m.pose.orientation.z = 0
        m.pose.orientation.w = 1.0

        m.points = []
        for i,(x,y) in enumerate(zip(xx, yy)):
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            m.points.append(p)

        return m



    def loop(self):
        r = rospy.Rate(20)

        while not rospy.is_shutdown():
            quat = tf.transformations.quaternion_from_euler(0, 0, self.yaw_smoothed)

            print('timetime = ', self.time_gps_fix_last_received)
            if self.time_gps_fix_last_received is not None:
                
                # self.kalman_filter_CV()
                self.moving_average()

                self.publish()

                self.gps_tf_broadcaster.sendTransform((self.e, self.n, 1.5),
                                                        quat,
                                                        self.time_gps_fix_last_received,
                                                        "gps", "world")
                
                # rviz에 covariance ellipse 표시하기
                xx, yy = self.ellipse(self.e, self.n, 2*self.e_sigma, 2*self.n_sigma)
                m = self.set_marker_array(xx, yy)

                self.gps_ellipse_pub.publish(m)
            r.sleep()


if __name__ == '__main__':
    try:
        rospy.init_node('GPS_System_Localizer')

        f = Localizer()

        rospy.Subscriber('/sensors/gps/fix', NavSatFix, f.callback_gps_fix, queue_size = 1)
        rospy.Subscriber('/sensors/gps/vel', TwistStamped, f.callback_gps_vel, queue_size = 1)
        # rospy.Subscriber('/localization/to_control_team', to_control_team_from_local_msg, self.callback_to_control_team_topic, queue_size=1)

        f.loop()

        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start Localizer() node.')
        pass

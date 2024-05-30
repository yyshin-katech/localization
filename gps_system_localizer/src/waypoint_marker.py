#!/usr/bin/env python

import rospy
import numpy as np

import glob
import scipy.io as sio
import matplotlib.pyplot as plt


from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
MAPFILE_PATH = rospy.get_param("MAPFILE_PATH")
# from LaneModels.msg import LaneModels

def waypoint_data_initialize(dir):
    matfiles = glob.glob(dir)
    wps = []
    j = 0
    for matfile in matfiles:
        mat = sio.loadmat(matfile)
        easts = mat["east"][0]
        norths = mat["north"][0]
        # print(easts[0], norths[0])
        stations = mat["station"][0]
        wp = {'id': j, 'e': easts, 'n': norths}
        j = j + 1
        wps.append(wp)
    # print("loaded mat file num is " + str(j))
    return wps

class waypoint_marker(object):
    def __init__(self):
        rospy.init_node('waypoint_marker')

        # initialize lanelet dataset
        mat = MAPFILE_PATH + "/*.mat"
        # mat = "/home/mmc-chanuk/Ros_Project/System-Localization/lanelet_waypoint_visualizer/src/waypoint_marker/database/*.mat"
        self.wps = waypoint_data_initialize(mat)
        # print(self.wps)
        # define publisher
        self.lanelet_strip_pub = rospy.Publisher('/rviz/waypoint_marker', MarkerArray, queue_size=100)

        # e_ego = 326223.961293405
        # n_ego = 340777.214972443
        e_ego, n_ego = 0, 0
        marker = MarkerArray()
        for wp in self.wps:
            marker_single = Marker()
            marker_single.id = wp['id']
            marker_single.header.frame_id = "/world"
            marker_single.type = marker_single.SPHERE_LIST
            marker_single.action = marker_single.ADD

            # marker_single scale
            marker_single.scale.x = 0.2
            marker_single.scale.y = 0.2
            marker_single.scale.z = 0.2

            # marker_single color
            marker_single.color.r = 210/255.0
            marker_single.color.g = 215/255.0
            marker_single.color.b = 211/255.0
            marker_single.color.a = 0.8

            # marker_single orientaiton
            marker_single.pose.orientation.x = 0.0
            marker_single.pose.orientation.y = 0.0
            marker_single.pose.orientation.z = 0.0
            marker_single.pose.orientation.w = 1.0

            # marker line pointsW
            marker_single.points = []
            for i in range(0,len(wp['e']),1):
                point = Point()
                point.x = wp['e'][i] - e_ego
                point.y = wp['n'][i] - n_ego
                # print("x is " + str(point.x) + " and y is " + str(point.y))
                marker_single.points.append(point)
            marker.markers.append(marker_single)


        # Publish the Marker after 5 seconds later
        rospy.sleep(5)
        self.lanelet_strip_pub.publish(marker)



if __name__ == '__main__':
    try:
        waypoint_marker = waypoint_marker()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint_marker() node.')

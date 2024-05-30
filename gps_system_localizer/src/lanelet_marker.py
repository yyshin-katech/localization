#!/usr/bin/env python3

import rospy
import numpy as np
import shapefile
import pyproj


from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
MAPFILE_PATH = rospy.get_param("MAPFILE_PATH_kiapi")
# from LaneModels.msg import LaneModels

def lanelet_data_initialize(dir):
    sf = shapefile.Reader(dir)
    # print(sf.fields)
    # print (' .shp file readed.')

    shapes = sf.shapes()
    shape_type = shapes[0].shapeType
    shape_parts = shapes[0].parts
    print('shape_type = ', shape_type)
    print('shape_parts = ', shape_parts)

    wps = []

    for j, shape in enumerate(shapes):

        if(len(shape.points) == 0):
            continue

        norths, easts = [], []
        lats, lons = [], []
        _index = j
        wp = {'id': j, 'e': [], 'n': []}
        # print(shape)
        #
        # print("%d/%d, len=%d" %(j, len(shapes), len(shape.points)))
        for i, points in enumerate(shape.points):
            location = [coord for coord in points]

            east, north = location

            # lon, lat = pyproj.transform(utm52n, wsg84, east, north)

            norths.append(north)
            easts.append(east)
            # lats.append(lat)
            # lons.append(lon)

            wp['e'].append(east)
            wp['n'].append(north)
            # wp['lon'].append(lon)
            # wp['lat'].append(lat)

        wps.append(wp)
    return wps

class lanelet_marker(object):
    def __init__(self):
        rospy.init_node('lanelet_marker')

        # initialize lanelet dataset
        shp = MAPFILE_PATH + '/A1_LANE.shp'
        self.wps = lanelet_data_initialize(shp)

        # define publisher
        self.lanelet_strip_pub = rospy.Publisher('/rviz/lanelet_marker', MarkerArray, queue_size=1)

        # e_ego = 326223.961293405
        # n_ego = 340777.214972443
        e_ego, n_ego = 0, 0
        marker = MarkerArray()
        for wp in self.wps:
            marker_single = Marker()
            marker_single.id = wp['id']
            # marker_single.header.frame_id = "world"
            marker_single.header.frame_id = "map"
            marker_single.type = marker_single.LINE_STRIP
            marker_single.action = marker_single.ADD

            # marker_single scale
            marker_single.scale.x = 0.15
            marker_single.scale.y = 0.03
            marker_single.scale.z = 0.03

            # marker_single color
            marker_single.color.r = 242.0/255.0
            marker_single.color.g = 217/255.0
            marker_single.color.b = 132/255.0
            marker_single.color.a = 0.8

            # marker_single orientaiton
            marker_single.pose.orientation.x = 0.0
            marker_single.pose.orientation.y = 0.0
            marker_single.pose.orientation.z = 0.0
            marker_single.pose.orientation.w = 1.0

            # marker line points
            marker_single.points = []
            for i in range(0,len(wp['e']),1):
                point = Point()
                point.x = wp['e'][i] - e_ego
                point.y = wp['n'][i] - n_ego
                point.z = 0
                # print("x is " + str(point.x) + " and y is " + str(point.y))
                marker_single.points.append(point)
            marker.markers.append(marker_single)
            # print("============================")
        # Publish the Marker after 2 seconds later
        rospy.sleep(1)

        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.lanelet_strip_pub.publish(marker)
            r.sleep()


if __name__ == '__main__':
    try:
        lanelet_marker = lanelet_marker()
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start lanelet_marker() node.')

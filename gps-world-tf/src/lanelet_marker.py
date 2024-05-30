#!/usr/bin/env python3

import rospy
import numpy as np
import shapefile


from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
# MAPFILE_PATH = '/root/mcar_ws/src/localization/gps-world-tf/map_data'
MAPFILE_PATH = rospy.get_param("MAPFILE_PATH")
OFFLINE = rospy.get_param("offline")
# from LaneModels.msg import LaneModels

def lanelet_data_initialize(dir, dir_dbf):
    sf = shapefile.Reader(dir, dir_dbf,encoding="EUC-KR")
    # print(sf.fields)
    # print (' .shp file readed.')

    shapes = sf.shapes()

    shape_type = shapes[0].shapeType
    shape_parts = shapes[0].parts
    records_ = sf.records()
    
    wps = []
    records =[]
    for j, shape in enumerate(shapes):
        record = records_[j]
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
        records.append(record)
    return wps, records

class lanelet_marker(object):
    def __init__(self):
        rospy.init_node('lanelet_marker')

        # initialize lanelet dataset
        shp = MAPFILE_PATH + '/B2_SURFACELINEMARK.shp'
        dbf = MAPFILE_PATH + '/B2_SURFACELINEMARK.dbf'
        self.wps, self.records = lanelet_data_initialize(shp,dbf)
        # print(self.records)
        # define publisher
        self.lanelet_strip_pub = rospy.Publisher('/rviz/lanelet_marker', MarkerArray, queue_size=1, latch=True)
        marker = MarkerArray()
        for idx, wp in enumerate(self.wps):
            record = self.records[idx]
            line_type = int(record[2])
            marker_single = Marker()
            marker_single.id = wp['id']
            marker_single.header.frame_id = "epsg5179"

            if (line_type%100%10 ==1):
                marker_single.type = marker_single.LINE_STRIP
            elif (line_type%100%10 ==2):
                marker_single.type = marker_single.LINE_STRIP
            else:
                marker_single.type = marker_single.POINTS
            marker_single.action = marker_single.ADD

            # marker_single scale
            marker_single.scale.x = 0.25
            marker_single.scale.y = 0.03
            marker_single.scale.z = 0.03

            # marker_single color
            if (line_type//100 == 1):
                marker_single.color.r = 242.0/255.0
                marker_single.color.g = 217/255.0
                marker_single.color.b = 30/255.0
                marker_single.color.a = 0.8
            elif (line_type//100 == 2):
                marker_single.color.r = 1.0
                marker_single.color.g = 1.0
                marker_single.color.b = 1.0
                marker_single.color.a = 0.8
            elif (line_type//100 == 3):
                marker_single.color.r = 0.0
                marker_single.color.g = 0.0
                marker_single.color.b = 1.0
                marker_single.color.a = 0.8
            else:
                marker_single.color.r = 1.0
                marker_single.color.g = 0.0
                marker_single.color.b = 0.0
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
                point.x = wp['e'][i]
                point.y = wp['n'][i]
                point.z = 0
                # print("x is " + str(point.x) + " and y is " + str(point.y))
                marker_single.points.append(point)
            if (len(wp['e'])%2 == 1):
                point = Point()
                point.x = wp['e'][-1]
                point.y = wp['n'][-1]
                point.z = 0
                marker_single.points.append(point)
            marker.markers.append(marker_single)
            # print("============================")
        # Publish the Marker after 2 seconds later
        # rospy.sleep(1)

        if OFFLINE:
            r = rospy.Rate(20)
            while not rospy.is_shutdown():
                self.lanelet_strip_pub.publish(marker)    
                r.sleep()

        else:
            self.lanelet_strip_pub.publish(marker)
            r = rospy.Rate(1)
            while not rospy.is_shutdown():
                
                r.sleep()


if __name__ == '__main__':
    try:
        lanelet_marker = lanelet_marker()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start lanelet_marker() node.')

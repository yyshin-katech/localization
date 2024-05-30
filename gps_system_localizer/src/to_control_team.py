#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
localization2d msg를 받아서
  1. 내가 현재 속한 차선
  2. 합류로 종료 전까지 남은 거리
  3. 분기로 까지 남은 거리
 를 출력해주는게 목표
 """

import rospy
import numpy as np
import scipy.io as sio
import time
import pyproj
# from scipy.spatial import distance

from mmc_msgs.msg import localization2D_msg, to_control_team_from_local_msg
from sensor_msgs.msg import NavSatFix
from utils import distance2curve
from utils_cython import find_closest, compute_current_lane, xy2frenet_with_closest_waypoint

MAPFILE_PATH = rospy.get_param('MAPFILE_PATH')


GPS_EPSG = pyproj.Proj(init='epsg:4326') # WSG84
MAP_EPSG_NUMBER = 5186
# MAP_EPSG_NUMBER = int(rospy.get_param('EPSG'))
MAP_EPSG = pyproj.Proj(init='epsg:%d' % MAP_EPSG_NUMBER) # map shapefile epsg


class DistanceCalculator(object):
    def __init__(self):
        rospy.init_node('To_Control_Team_Node')
        self.init_variable()
        self.load_centerline_map()

        self.set_subscriber()
        self.set_publisher()

        rospy.spin()

    def init_variable(self):
        # 맵 centerlines
        self.main_1 = None
        self.main_2 = None
        self.merge_in = None
        self.merge_out = None
        self.map_loaded = False

        self.east = 0.0
        self.north = 0.0
        self.yaw = 0.0

        self.previous_index = np.zeros((4,), dtype=np.int32)
        self.set_target_roads = False

    def set_subscriber(self):
        # localization 정보 들어오면 바로 control team에 필요한 메세지 publish
        # callback 안에 publish 명령어까지 같이 들어있음
        rospy.Subscriber('/localization/pose_2d', localization2D_msg, self.pose_2d_cb, queue_size=1)
        rospy.Subscriber('/sensors/gps/fix', NavSatFix, self.gps_position_update_cb, queue_size=1)

    def set_publisher(self):
        self.to_control_team_pub = rospy.Publisher('/localization/to_control_team', to_control_team_from_local_msg, queue_size=1)

    def load_centerline_map(self):
        self.main_1 = sio.loadmat(MAPFILE_PATH + '/mainlane_1.mat')
        self.main_2 = sio.loadmat(MAPFILE_PATH + '/mainlane_2.mat')
        self.merge_in = sio.loadmat(MAPFILE_PATH + '/merge_in.mat')
        self.merge_out = sio.loadmat(MAPFILE_PATH + '/merge_out.mat')
        self.map_loaded = True

    def gps_position_update_cb(self, msg):
        check_data = [msg.longitude, msg.latitude, msg.position_covariance[0], msg.position_covariance[4]]
        healty = np.all(np.isfinite(check_data)) # 위 값이 Nan이나 Inf가 아니면 healty

        if healty:
            east, north = pyproj.transform(GPS_EPSG, MAP_EPSG, msg.longitude, msg.latitude)
            east_sigma = np.sqrt(msg.position_covariance[0])
            north_sigma = np.sqrt(msg.position_covariance[4])

            self.east = east
            self.north = north

    def compute_my_lane(self, e, n):
        ''' distance to cureve 함수를 이용해서 내 차선 찾기 '''
        if self.map_loaded:
            target_roads = [self.main_1, self.main_2, self.merge_in, self.merge_out]

        if self.map_loaded:
            minimum_lateral_offset = 9999
            start = time.time()
            distances = []
            for i, target_road in enumerate(target_roads):

                mapx = target_road['east'][0][::10]
                mapy = target_road['north'][0][::10]
                maps = target_road['station'][0][::10]

                waypoints = np.vstack((mapx, mapy)).T # (N x 2)
                pose = np.array([[e, n]])

                xy, distance, t = distance2curve(waypoints, pose)

                distances.append(distance[0])
            end = time.time()

            # print('time elpased = {}ms'.format((end-start)*1000))
            # print(distances)
            # print(np.argmax(distances))
            # print(maps[:10])

    def compute_my_lane_cy(self, e, n):
        ''' cython 버전 '''
        lane_names = ['main_1', 'main_2', 'merge_in', 'merge_out', 'none']
        current_lane_id = -1
        current_lane_name = 'none'
        current_closest_waypoint_index = -1
        current_closest_waypoint_in_MATLAB = 0
        distance_to_entry_end = -1
        distance_to_exit_start = -1
        current_s = 0
        current_d = 0

        if self.map_loaded and not self.set_target_roads:
            self.target_roads = [self.main_1, self.main_2, self.merge_in, self.merge_out]
            self.set_target_roads = True

        if self.set_target_roads:
            # start = time.time()
            distances, indexs = compute_current_lane(self.target_roads, e, n)
            # print(distances)
            min_abs_d = 100.0

            for i, (dist, closest_waypoint) in enumerate(zip(distances, indexs)):
                if dist > 4.0:
                    pass
                else:
                    mapx = self.target_roads[i]['east'][0]
                    mapy = self.target_roads[i]['north'][0]
                    maps = self.target_roads[i]['station'][0]

                    s, d = xy2frenet_with_closest_waypoint(e, n, closest_waypoint, mapx, mapy, maps)
                    if abs(d) < min_abs_d:
                        current_lane_id = i
                        current_s = s
                        current_d = d
                        min_abs_d = abs(d)
                        current_closest_waypoint_index = closest_waypoint
                    # print('[%d/%s] lateral offset = %.2f' % (i, lane_names[i], d))

            if current_closest_waypoint_index > 0:
                ''' 가장 최근에 지난 waypoint index 던져주기'''
                # matlab은 index가 1부터 시작하는 것에 조심하기
                maps = self.target_roads[current_lane_id]['station'][0]
                n_waypoints_in_map = len(maps)
                if maps[current_closest_waypoint_index] > current_s:
                    # 최근에 지난 waypoint index로 설정
                    closest_waypoint -= 1

                current_closest_waypoint_index = np.clip(current_closest_waypoint_index, 0, n_waypoints_in_map-1)
                current_closest_waypoint_in_MATLAB = current_closest_waypoint_index + 1

                current_lane_name = lane_names[current_lane_id]
                # print("current_lane: {}".format(current_lane_name))

                if current_lane_name == 'merge_in':
                    distance_to_entry_end = self.merge_in['station'][0][-1] - current_s

                if current_lane_name == 'main_1' or current_lane_name == 'main_2':
                    # 분기로 시작점
                    _e = self.merge_out['east'][0][0]
                    _n = self.merge_out['north'][0][0]
                    _exit_0 = np.array([_e, _n])

                    # 현재 달리고 있는 차선
                    l = [self.main_1, self.main_2][current_lane_id]
                    waypoints_east = l['east'][0]
                    waypoints_north = l['north'][0]

                    # 현재 달리는 차선에서 분기로 시작점에 가장 가까운점 찾기
                    if current_lane_name == 'main_1':
                        min_index = 442
                    else:
                        min_index = 432
                    distance_to_exit_start = l['station'][0][min_index] - current_s
            # print('[control_team_msg_pub.py] id=%d, name=%s' % (current_lane_id, current_lane_name))
            # print('-------------------------------')

            # end = time.time()
            # elapsed_in_ms = (end-start)*1000
            # print("time = {:.2f}ms".format(elapsed_in_ms))
            # self.times.append(elapsed_in_ms)
            # print("maximum = {:.2f}ms".format(max(self.times)))

        return current_lane_id, current_lane_name, distance_to_entry_end, distance_to_exit_start, current_s, current_d, current_closest_waypoint_in_MATLAB


    def pose_2d_cb(self, msg):
        """
        localization 메세지를 받아서 control team에 필요한 메세지 publish
        """
        e = msg.east
        n = msg.north
        # e = self.east
        # n = self.north
        yaw = np.arctan2(np.tan(msg.yaw), 1)

        # t0 = time.time()
        # mylane_ind, mylane_str, distance_to_entry_end, distance_to_exit_start, frenet_s, frenet_d = self.get_my_lane(e, n)
        # self.compute_my_lane(e, n)
        current_lane_id, current_lane_name, distance_to_entry_end, distance_to_exit_start, current_s, current_d, current_closest_waypoint_in_MATLAB = self.compute_my_lane_cy(e, n)
        # print("[control_team_msg_pub.py] computation_time = %.1f (ms)" % ((time.time() - t0)*1000) )

        p = to_control_team_from_local_msg()
        p.time = msg.time
        p.lane_id = current_lane_id + 1 # 제어팀에서 요구한 부분 1: 본선1차로, 2: 본선2차로, 3: 합류로, 4: 분기로
        p.lane_name = current_lane_name

        p.distance_to_entry_end = distance_to_entry_end
        p.distance_to_exit_start = distance_to_exit_start

        p.host_east = e
        p.host_north = n
        p.host_yaw = yaw # radian

        p.station = current_s
        p.lateral_offset = current_d

        p.waypoint_index = current_closest_waypoint_in_MATLAB

        self.to_control_team_pub.publish(p)


if __name__ == "__main__":
    DistanceCalculator()

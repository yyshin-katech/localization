#!/usr/bin/env python3.8
#-*- coding: utf-8 -*-
"""
to_control_team.py의 12월 데모를 위한 버전
"""

import rospy
import numpy as np
import scipy.io as sio

from scipy.spatial import cKDTree as KDTree
import time
import pyproj
# from scipy.spatial import distance

from mmc_msgs.msg import localization2D_msg, to_control_team_from_local_msg
from sensor_msgs.msg import NavSatFix
from utils import distance2curve

from utils_cython import find_closest, compute_current_lane, xy2frenet_with_closest_waypoint_loop, xy2frenet_with_closest_waypoint


MAPFILE_PATH = rospy.get_param('MAPFILE_PATH_kiapi')


GPS_EPSG = pyproj.Proj(init='epsg:4326') # WSG84
MAP_EPSG_NUMBER = 5186
# MAP_EPSG_NUMBER = int(rospy.get_param('EPSG'))
MAP_EPSG = pyproj.Proj(init='epsg:%d' % MAP_EPSG_NUMBER) # map shapefile epsg


class DistanceCalculator(object):
    def __init__(self):
        rospy.init_node('To_Control_Team_Node')
        self.init_variable()
        self.load_centerline_map()
        # self.init_kdtree()
        self.set_subscriber()
        self.set_publisher()

        rospy.spin()

    def init_variable(self):
        # 맵 centerlines
        self.road_1 = None
        self.road_2 = None
        self.road_3 = None
        self.road_4 = None
        self.map_loaded = False


        self.east = 0.0
        self.north = 0.0
        self.yaw = 0.0

        self.previous_index = np.zeros((4,), dtype=np.int32)

    def init_kdtree(self):
        self.trees=[]
        if self.map_loaded:
            self.Waypoint_set_s=[]
            for road_n in self.target_roads:
                Waypoint_set = []
                Waypoint_set.append(road_n['east'][0])
                Waypoint_set.append(road_n['north'][0])
                Waypoint_set = np.array(Waypoint_set).transpose()
                tree = KDTree(Waypoint_set,leafsize = 3000)
                self.trees.append(tree)
                self.Waypoint_set_s.append(road_n['station'][0])
            print("KDTREE initialization is sucessful")
        else:
            print("map is not loaded!!!")

    def set_subscriber(self):
        # localization 정보 들어오면 바로 control team에 필요한 메세지 publish
        # callback 안에 publish 명령어까지 같이 들어있음
        rospy.Subscriber('/localization/pose_2d', localization2D_msg, self.pose_2d_cb, queue_size=1)

    def set_publisher(self):
        self.to_control_team_pub = rospy.Publisher('/localization/to_control_team', to_control_team_from_local_msg, queue_size=1)

    def load_centerline_map(self):
        self.road_1 = sio.loadmat(MAPFILE_PATH + '/waypoint_1.mat')
        self.road_2 = sio.loadmat(MAPFILE_PATH + '/waypoint_2.mat')
        self.road_3 = sio.loadmat(MAPFILE_PATH + '/waypoint_3.mat')
        self.road_4 = sio.loadmat(MAPFILE_PATH + '/waypoint_4.mat')
        self.target_roads = [self.road_1, self.road_2, self.road_3, self.road_4]
        self.map_loaded = True



    def compute_my_lane_kdtree(self, e, n):
        ''' kdtree 로 변경'''
        lane_names = ['road_1', 'road_2', 'road_3', 'road_4', 'none']
        current_lane_id = -1
        current_lane_name = 'none'
        distance_to_entry_end = -1
        distance_to_exit_start = -1
        current_closest_waypoint_index = -1
        current_closest_waypoint_in_MATLAB = 0
        current_s = 0
        current_d = 0
        # start = time.time()
        distances = []
        indexs = []
        for tree in self.trees:
            # index, distance = tree.nn_index(np.array([e, n]), k=1)
            e = self.target_roads[0]['east'][0][0]
            n = self.target_roads[0]['north'][0][0]
            distance,index  = tree.query(np.array([e, n]), k=1)
            print (distance)
            distances.append(distance)
            indexs.append(index)


        for i in range(len(distances)):
            closest_waypoint = indexs[i]
            mapx = self.target_roads[i]['east'][0]
            mapy = self.target_roads[i]['north'][0]
            maps = self.target_roads[i]['station'][0]
            min_abs_d = 100.
            # if i is 3:
            if i == 3:
                ''' road_4 의 경우에는 loop가 없음 '''
                s, d = xy2frenet_with_closest_waypoint(e, n, closest_waypoint, mapx, mapy, maps)
            else:
                ''' loop가 있는 경우에는 station을 연속적으로 출력할 수 있게 '''
                s, d = xy2frenet_with_closest_waypoint_loop(e, n, closest_waypoint, mapx, mapy, maps)
            if abs(d) < min_abs_d:
                current_lane_id = i
                current_s = s
                current_d = d
                min_abs_d = abs(d)
                current_closest_waypoint_index = closest_waypoint

        ''' 가장 최근에 지난 waypoint index 던져주기'''
        if current_closest_waypoint_index > 0:
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

            # end = time.time()
            # elapsed_in_ms = (end-start)*1000
            # print("time = {:.2f}ms".format(elapsed_in_ms))
            # self.times.append(elapsed_in_ms)
            # print("maximum = {:.2f}ms".format(max(self.times)))

        return current_lane_id, current_lane_name, distance_to_entry_end, distance_to_exit_start, current_s, current_d, current_closest_waypoint_in_MATLAB

    def compute_my_lane_cy(self, e, n):
        ''' cython 버전 '''
        lane_names = ['road_1', 'road_2', 'road_3', 'road_4', 'none']
        current_lane_id = -1
        current_lane_name = 'none'
        distance_to_entry_end = -1
        distance_to_exit_start = -1
        current_closest_waypoint_index = -1
        current_closest_waypoint_in_MATLAB = 0
        current_s = 0
        current_d = 0

        if self.map_loaded:
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
                    # if i is 3:
                    if i == 3:
                        ''' road_4 의 경우에는 loop가 없음 '''
                        s, d = xy2frenet_with_closest_waypoint(e, n, closest_waypoint, mapx, mapy, maps)
                    else:
                        ''' loop가 있는 경우에는 station을 연속적으로 출력할 수 있게 '''
                        s, d = xy2frenet_with_closest_waypoint_loop(e, n, closest_waypoint, mapx, mapy, maps)
                    if abs(d) < min_abs_d:
                        current_lane_id = i
                        current_s = s
                        current_d = d
                        min_abs_d = abs(d)
                        current_closest_waypoint_index = closest_waypoint

            ''' 가장 최근에 지난 waypoint index 던져주기'''
            if current_closest_waypoint_index > 0:
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
        t0 = time.time()

        e = msg.east
        n = msg.north
        # e = self.east
        # n = self.north
        self.yaw = msg.yaw
        yaw = msg.yaw

        # mylane_ind, mylane_str, distance_to_entry_end, distance_to_exit_start, frenet_s, frenet_d = self.get_my_lane(e, n)
        # self.compute_my_lane(e, n)
        # current_lane_id, current_lane_name, distance_to_entry_end, distance_to_exit_start, current_s, current_d, current_closest_waypoint_in_MATLAB = self.compute_my_lane_kdtree(e, n)
        current_lane_id, current_lane_name, distance_to_entry_end, distance_to_exit_start, current_s, current_d, current_closest_waypoint_in_MATLAB = self.compute_my_lane_cy(e, n)


        p = to_control_team_from_local_msg()
        p.time = msg.time

        ## KIAPI 에서 곡선부 돌때 lane id가 튀는 현상을 잡기 위한 임시방편 코드
        if 1380 < current_s and current_s < 1490:
            current_lane_id = 0

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
        print("[control_pub] computation_time = %.4f (ms)" % ((time.time() - t0)*1000) )

        self.to_control_team_pub.publish(p)


if __name__ == "__main__":
    DistanceCalculator()

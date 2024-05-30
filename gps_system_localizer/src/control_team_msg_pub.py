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
from scipy.spatial import distance

from mmc_msgs.msg import localization2D_msg, to_control_team_from_local_msg
from sensor_msgs.msg import NavSatFix

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

    def set_subscriber(self):
        # localization 정보 들어오면 바로 control team에 필요한 메세지 publish
        # callback 안에 publish 명령어까지 같이 들어있음
        rospy.Subscriber('/localization/pose_2d', localization2D_msg, self.pose_2d_cb)
        rospy.Subscriber('/sensors/gps/fix', NavSatFix, self.gps_position_update_cb)

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

    def get_my_lane(self, e, n):
        # 초기값
        mylane_str = 'map_not_initialized_yet'
        distance_to_entry_end = -1
        distance_to_exit_start = -1
        current_lane = -2
        frenet_s = 0
        frenet_d = 0
        current_frenet_s = 0
        current_frenet_d = 0
        # print("self.map_loaded"   , self.map_loaded)

        if self.map_loaded:
            target_roads = [self.main_1, self.main_2, self.merge_in, self.merge_out]
            # target_raods = [self.main_1]

        if self.map_loaded:
            """
            find frenet for all target_road
            """
            minimum_lateral_offset = 9999
            for i, target_road in enumerate(target_roads):

                mapx = target_road['east'][0][::2]
                mapy = target_road['north'][0][::2]
                maps = target_road['station'][0][::2]


                # next waypoint 찾기
                waypoints = np.vstack((mapx, mapy)).T # (N x 2)
                pose = np.array([e, n])

                t0 = time.time()
                # min_dist, closest_waypoint = cKDTree(waypoints).query(pose,1)
                closest_len = 1000000.0
                closest_waypoint = 0
                for j in range(len(mapx)):
                    map_x = mapx[j]
                    map_y = mapy[j]
                    dx = e - map_x
                    dy = n - map_y
                    dist = np.sqrt(dx*dx + dy*dy)

                    if dist < closest_len:
                        closest_len = dist
                        closest_waypoint = j

                t_elapsed = time.time() - t0
                if t_elapsed > 0.1:
                    print("[control_team_msg_pub.py] LANE %d: COMPUTE_MINIMUM_DISTANCE = %.1f ms" % (i, t_elapsed * 1000))

                if closest_len > 5:
                    pass
                else:
                    t0 = time.time()
                    if closest_waypoint < 0:
                        closest_waypoint = 0

                    if closest_waypoint >= len(mapx) -1:
                        closest_waypoint = len(mapx) -2

                    map_vec = [mapx[closest_waypoint + 1] - mapx[closest_waypoint],
                               mapy[closest_waypoint + 1] - mapy[closest_waypoint]]
                    ego_vec = [e - mapx[closest_waypoint], n - mapy[closest_waypoint]]
                    direction = np.sign(np.dot(map_vec, ego_vec))
                    if direction >= 0:
                        next_wp = closest_waypoint + 1
                    else:
                        next_wp = closest_waypoint

                    prev_wp = next_wp - 1

                    if (prev_wp == -1) or (next_wp == len(mapx)):
                        frenet_d = 0
                        frenet_s = -1
                        print('[control_team_msg_pub.py] cannot get frenet (s,d)')
                    else:
                        n_x = mapx[next_wp] - mapx[prev_wp]
                        n_y = mapy[next_wp] - mapy[prev_wp]
                        x_x = e-mapx[prev_wp]
                        x_y = n-mapy[prev_wp]

                        # find the projection of [x,y] onto n
                        proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y)
                        proj_x = proj_norm*n_x
                        proj_y = proj_norm*n_y

                        # get frenet d
                        dx = x_x - proj_x
                        dy = x_y - proj_y
                        frenet_d = np.sqrt(dx*dx + dy*dy)

                        # set sign of frenet_d
                        """
                        얘 대구 가서 체크해봐야함
                        """
                        ego_vec = [e-mapx[prev_wp], n-mapy[prev_wp], 0]
                        map_vec = [n_x, n_y, 0]
                        d_cross = np.cross(ego_vec,map_vec)
                        if d_cross[-1] > 0:
                            frenet_d = -frenet_d

                        # get frenet s
                        frenet_s = maps[prev_wp] + np.sqrt(proj_x*proj_x + proj_y*proj_y)

                        t_elapsed = time.time() - t0
                        if t_elapsed > 0.1:
                            print("[control_team_msg_pub.py] COMPUTE LOCATION IN FRENET: %.1f (ms)" % (t_elapsed * 1000))

                        if np.abs(frenet_d) < minimum_lateral_offset:
                            minimum_lateral_offset = np.abs(frenet_d)
                            current_frenet_d = frenet_d
                            current_frenet_s = frenet_s
                            current_lane = i

            if current_lane < 0:
                mylane_str = "far_away_from_map"
                current_lane = -2
                return current_lane, mylane_str, -1, -1, 0, 0

            # 현재 내 차선의 이름 (string)
            if current_lane == 0:
                mylane_str = 'main_1'
            if current_lane == 1:
                mylane_str = 'main_2'
            if current_lane == 2:
                mylane_str = 'merge_in'
            if current_lane == 3:
                mylane_str = 'merge_out'

            # 합류로 종료 지점까지 남은 거리 계산
            if mylane_str == 'merge_in':
                distance_to_entry_end = self.merge_in['station'][0][-1] - current_frenet_s

            # 분기로 진입 지점까지 남은 거리 계산
            if mylane_str == 'main_1' or mylane_str == 'main_2':
                t0 = time.time()
                # 분기로 시작점
                _e = self.merge_out['east'][0][0]
                _n = self.merge_out['north'][0][0]
                _exit_0 = np.array([_e, _n])

                # 현재 달리고 있는 차선
                l = [self.main_1, self.main_2][current_lane]
                waypoints_east = l['east'][0]
                waypoints_north = l['north'][0]
                waypoints = np.vstack((waypoints_east, waypoints_north)).T # (N x 2)

                # 현재 달리는 차선에서 분기로 시작점에 가장 가까운점 찾기
                # min_dist, min_index = cKDTree(waypoints).query(_exit_0,1)
                if mylane_str == 'main_1':
                    min_index = 442
                else:
                    min_index = 432
                # print("min index = ", min_index)
                # print(mylane_str)
                distance_to_exit_start = l['station'][0][min_index] - current_frenet_s

                t_elapsed = time.time() - t0
                    # print("[control_team_msg_pub.py] COMPUTE DISTANCE LEFT TO .MERGE OUT.: %.1f (ms)" % (t_elapsed * 1000))


            # print(" [%s] distance_to_entry_end = %.1f (m), distance_to_exit_start = %.1f (m)" % (mylane_str,
            #                                                                                      distance_to_entry_end,
            #                                                                                      distance_to_exit_start))
        return current_lane, mylane_str, distance_to_entry_end, distance_to_exit_start, current_frenet_s, current_frenet_d




    def pose_2d_cb(self, msg):
        """
        localization 메세지를 받아서 control team에 필요한 메세지 publish
        """
        # e = msg.east
        # n = msg.north
        e = self.east
        n = self.north
        self.yaw = msg.yaw
        yaw = msg.yaw

        # t0 = time.time()
        mylane_ind, mylane_str, distance_to_entry_end, distance_to_exit_start, frenet_s, frenet_d = self.get_my_lane(e, n)
        # print("[control_team_msg_pub.py] computation_time = %.1f (ms)" % ((time.time() - t0)*1000) )

        p = to_control_team_from_local_msg()
        p.time = msg.time
        p.lane_id = mylane_ind + 1 # 제어팀에서 요구한 부분 1: 본선1차로, 2: 본선2차로, 3: 합류로, 4: 분기로
        p.lane_name = mylane_str

        p.distance_to_entry_end = distance_to_entry_end
        p.distance_to_exit_start = distance_to_exit_start

        p.host_east = e
        p.host_north = n
        p.host_yaw = yaw # radian

        p.station = frenet_s
        p.lateral_offset = frenet_d

        self.to_control_team_pub.publish(p)


if __name__ == "__main__":
    DistanceCalculator()

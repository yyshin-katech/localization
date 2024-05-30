#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
RVIZ에 VILS차량 올리기

inputs
- use_VILS: VILS 사용하는지 여부
- n_cars: 몇 대의 VILS 차량이 있는지
- (s,d,lane,v) for every cars: Frenet coordinate 상에서 (s,d,lane,v)

work
- (s,d,lane,v) -> (X,Y,theta)
    - 이 때 theta는 차선 방향
- (X,Y,theta) VILS 차량을 RVIZ에 직육면체로 올리기

"""

import rospy
import tf
import numpy as np
import scipy.io as sio
from visualization_msgs.msg import MarkerArray, Marker
from mmc_msgs.msg import control_team_veh_array_msg

MAPFILE_PATH = rospy.get_param("MAPFILE_PATH")

ZERO_LANE = sio.loadmat(MAPFILE_PATH + '/mainlane_1.mat')

N_MAX_CAR = 20

def frenet_to_cartesian(s, d, lane):
    """
    차선이 주어졌을 때 s,d를 (E, N, yaw)로 바꾸기
    e = e - e0
    n = n - n0 이거
    """
    mapx = lane['east'][0]
    mapy = lane['north'][0]
    # mapx = lane['east'][0] - ZERO_LANE['east'][0][0]
    # mapy = lane['north'][0] - ZERO_LANE['north'][0][0]
    maps = lane['station'][0]

    # print(maps[-1])

    prev_wp = 0

    while (s > maps[prev_wp]) and (prev_wp < len(maps)-2):
        prev_wp += 1

    next_wp = np.maximum(prev_wp, len(maps)-1)

    dydx = (mapy[next_wp] - mapy[prev_wp]) / (mapx[next_wp] - mapx[prev_wp])

    if (dydx >= 0) and (mapx[next_wp] >= mapx[prev_wp]):
        position = '1'
    elif (dydx >= 0) and (mapx[next_wp] < mapx[prev_wp]):
        position = '2'
    elif (dydx < 0) and (mapx[next_wp] <= mapx[prev_wp]):
        position = '3'
    else:
        position = '4'

    heading = np.arctan(dydx)
    if position == '2':
        heading += np.pi

    if position == '3':
        heading -= np.pi

    seg_s = s - maps[prev_wp];
    seg_x = mapx[prev_wp] + seg_s * np.cos(heading)
    seg_y = mapy[prev_wp] + seg_s * np.sin(heading)

    perp_heading = heading + np.pi/2.0
    # perp_heading = heading - np.pi/2.0 # d 방향이 맞도록 수정
    x = seg_x + d * np.cos(perp_heading)
    y = seg_y + d * np.sin(perp_heading)

    E = x
    N = y
    yaw = heading

    return E, N, yaw


def set_marker(east, north, yaw, _id, l=5, w=2, h=0.5):
    o = Marker()
    o.type = Marker.CUBE
    o.header.frame_id = "world"
    o.id = _id
    o.lifetime = rospy.Time(0.05)

    o.scale.x = l
    o.scale.y = w
    o.scale.z = h

    o.pose.position.x = east
    o.pose.position.y = north
    o.pose.position.z = o.scale.z / 2.0

    quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)

    o.pose.orientation.x = quaternion[0]
    o.pose.orientation.y = quaternion[1]
    o.pose.orientation.z = quaternion[2]
    o.pose.orientation.w = quaternion[3]
    # rgba(241, 169, 160, 1)
    o.color.r = 230/255.0
    o.color.g = 169/255.0
    o.color.b = 160/255.0
    o.color.a = 0.8

    return o


class RVIZ_VILS_Manager(object):
    def __init__(self):
        rospy.init_node("rviz_vils_manager")

        self.init_variables()
        self.load_centerline_map()
        self.set_subscriber()
        self.set_publisher()

        # tmp set object states
        # self.set_tmp_car_states()

        while not rospy.is_shutdown():
            # self.check_VILS_health()
            self.transfrom_Frenet_to_Cartesian_and_publish()
            rospy.sleep(0.05)

    def init_variables(self):
        # 맵 centerlines
        self.main_1 = None
        self.main_2 = None
        self.merge_in = None
        self.merge_out = None
        self.map_loaded = False

        # VILS object
        self.use_VILS = False
        self.n_cars = 0
        self.objects = [] # array of object (car_id, lane_id, s,d,v)
        self.time_VILS_message_received = rospy.Time(0)


    def load_centerline_map(self):
        self.main_1 = sio.loadmat(MAPFILE_PATH + '/mainlane_1.mat')
        self.main_2 = sio.loadmat(MAPFILE_PATH + '/mainlane_2.mat')
        self.merge_in = sio.loadmat(MAPFILE_PATH + '/merge_in.mat')
        self.merge_out = sio.loadmat(MAPFILE_PATH + '/merge_out.mat')
        self.target_roads = [self.main_1, self.main_2, self.merge_in, self.merge_out]
        self.map_loaded = True
        # print("[-] rviz: centerline map loaded. ")

    def set_subscriber(self):
        rospy.Subscriber('/from_control_team/VILS_objects', control_team_veh_array_msg, self.VILS_object_cb)

    def set_publisher(self):
        self.VILS_pub = rospy.Publisher("rviz/VILS_objects", MarkerArray, queue_size=1)

    def VILS_object_cb(self, msg):
        """
        VILS object 메세지 받아서 파싱
            1: 본선1차로
            2: 본선2차로
            3: 합류로
            4: 분기로
            그래서 시각화때 index는 0부터 시작하므로 lane_id에서 1 빼줘야.
        """
        self.time_VILS_message_received = msg.time
        self.use_VILS = msg.obj_check
        self.n_cars = msg.n_car
        self.objects = []

        for i in range(N_MAX_CAR):
            try:
                data = msg.data[i]
                _object = {'car_id': data.car_id,
                           'lane_id': data.lane_id - 1,
                           's': data.s,
                           'd': data.d,
                           'v': data.v}
                self.objects.append(_object)
            except:
                pass
        # if len(msg.data) != msg.n_car:
        #     rospy.logwarn('[rviz_VILS.py] len(msg.data) != msg.n_car')

    def check_VILS_health(self):
        """
        VILS 메세지가 죽었는지 체크하기
        """
        self.use_VILS = True
        # dt = (rospy.get_rostime() - self.time_VILS_message_received).to_sec()
        #
        # if abs(dt) > 0.5:
        #     # 최근에 들어온 메시지 아니면 VILS 죽었다고 판단
        #     self.use_VILS = False


    def set_tmp_car_states(self):
        self.n_cars = 3
        self.objects = []
        for i in range(self.n_cars):
            _object = {'s': np.random.randint(10) * 4, 'd': np.random.randint(10) * 0.1, 'lane_id': 1}
            self.objects.append(_object)
            # print(_object)


    def transfrom_Frenet_to_Cartesian_and_publish(self):
        """
        1. VILS object들을 rviz에 올리기 위해 cartesian frame으로 바꾸기
        2. Marker로 publish하기
        """
        if self.use_VILS:
            oa = MarkerArray()
            # print(self.n_cars)
            for i in range(len(self.objects)):
                _object = self.objects[i]
                lane_id = _object['lane_id']
                if lane_id >=0:
                    lane = self.target_roads[lane_id]

                    s = _object['s']
                    d = _object['d']

                    east, north, yaw = frenet_to_cartesian(s, d, lane)
                    marker = set_marker(east, north, yaw, _id=i)
                    oa.markers.append(marker)
                else:
                    pass


            self.VILS_pub.publish(oa)


if __name__ == "__main__":

    manager = RVIZ_VILS_Manager()

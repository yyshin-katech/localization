#-*- coding: utf-8 -*-
import numpy as np

def motion_model(x, u, dt):
    '''
    x: east, north, theta
    u: vx, yawrate
    '''
    vx, yawrate = u[0], u[1]
    e_prev, n_prev, theta_prev = x[0,0], x[1,0], x[2,0]

    if np.abs(yawrate) < 1e-4:
        theta = theta_prev + yawrate * dt
        e = e_prev + vx * np.cos(theta_prev) * dt
        n = n_prev + vx * np.sin(theta_prev) * dt

        G = np.matrix([[1, 0, -vx * np.sin(theta_prev)*dt],
                       [0, 1, vx * np.cos(theta_prev)*dt],
                       [0, 0, 1]])

        V = np.matrix([[np.cos(theta_prev)*dt, 0],
                       [np.sin(theta_prev)*dt, 0],
                       [0, dt]])

    else:
        xc = e_prev - vx/yawrate * np.sin(theta_prev)
        yc = n_prev + vx/yawrate * np.cos(theta_prev)
        theta = theta_prev + yawrate * dt

        e = xc + vx/yawrate * np.sin(theta)
        n = yc - vx/yawrate * np.cos(theta)

        G = np.matrix([[1, 0, -vx/yawrate*np.cos(theta_prev) + vx/yawrate*np.cos(theta)],
                       [0, 1, -vx/yawrate*np.sin(theta_prev) + vx/yawrate*np.sin(theta)],
                       [0, 0, 1]])

        V = np.matrix([[-(np.sin(theta_prev)+np.sin(theta))/yawrate, (vx*(np.sin(theta_prev)-np.sin(theta)))/(yawrate*yawrate)+(vx*np.cos(theta)*dt)/yawrate],
                       [(np.cos(theta_prev)-np.cos(theta))/yawrate, -(vx*(np.cos(theta_prev)-np.cos(theta)))/(yawrate*yawrate)+vx*np.sin(theta)*dt/yawrate],
                       [0, dt]])
    # print(e, n, theta)
    x = np.matrix([e, n, theta]).T

    return x, G, V

def normalize_angle(theta):
    '''
    angle이 +/- pi 사이 값을 가지게 하기
    '''
    # print ' not yet normalized = {}'.format(self.theta)
    return np.arctan2(np.sin(theta), np.cos(theta))

def calculate_angle_error(prev_angle, current_angle):
    try:
        prev_angle = prev_angle[0,0]
        current_angle = current_angle[0,0]
    except:
        pass
    cp = np.cos(prev_angle)
    sp = np.sin(prev_angle)
    p = [cp, sp]


    cc = np.cos(current_angle)
    sc = np.sin(current_angle)
    c = [cc, sc]

    dot = np.dot(p, c)
    cross = np.cross(p, c, axis=0)
    return np.arctan2(cross, dot)

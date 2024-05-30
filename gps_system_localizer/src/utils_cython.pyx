import numpy as np
cimport numpy as c_np
from libc.math cimport sqrt

cpdef list find_closest(c_np.ndarray[double, ndim=1] mapx, c_np.ndarray[double, ndim=1] mapy, double x, double y):
  cdef double closest_len, map_x, map_y, dx, dy, dist
  cdef int closest_waypoint, j

  closest_len = 1000000.0
  closest_waypoint = 0

  for j in range(len(mapx)):
      map_x = mapx[j]
      map_y = mapy[j]
      dx = x - map_x
      dy = y - map_y
      dist = sqrt(dx*dx + dy*dy)

      if dist < closest_len:
          closest_len = dist
          closest_waypoint = j

  return [closest_len, closest_waypoint]


cpdef list compute_current_lane(object target_roads, double e, double n):
  cdef double closest_len
  cdef int closest_waypoint, i
  cdef c_np.ndarray[double, ndim=2] waypoints
  cdef c_np.ndarray[double, ndim=1] pose, mapx, mapy

  cdef list dists, waypoints_list

  dists = []
  waypoints_list = []

  for i, target_road in enumerate(target_roads):
    mapx = target_road['east'][0]
    mapy = target_road['north'][0]

    waypoints = np.vstack((mapx, mapy)).T # (N x 2)
    pose = np.array([e, n])

    closest_len, closest_waypoint = find_closest(mapx, mapy, e, n)
    dists.append(closest_len)
    waypoints_list.append(closest_waypoint)

  return [dists, waypoints_list]

cpdef list xy2frenet_with_closest_waypoint(double e, double n, int closest_waypoint, c_np.ndarray[double, ndim=1] mapx, c_np.ndarray[double, ndim=1] mapy, c_np.ndarray[double, ndim=1] maps):
  cdef c_np.ndarray[double, ndim=1] ego_vec, map_vec, d_cross
  cdef double direction, frenet_d, frenet_s, n_x, n_y, x_x, x_y, proj_norm, proj_x, proj_y, dx, dy
  cdef int next_wp, prev_wp

  if closest_waypoint < 0:
    closest_waypoint = 0

  if closest_waypoint >= len(mapx) -1:
    closest_waypoint = len(mapx) -2

  map_vec = np.array([mapx[closest_waypoint + 1] - mapx[closest_waypoint],
                      mapy[closest_waypoint + 1] - mapy[closest_waypoint]])
  ego_vec = np.array([e - mapx[closest_waypoint], n - mapy[closest_waypoint]])
  direction = np.sign(np.dot(map_vec, ego_vec))
  if direction >= 0:
    next_wp = closest_waypoint + 1
  else:
    next_wp = closest_waypoint

  prev_wp = next_wp - 1


  if (prev_wp == -1) or (next_wp == len(mapx)):
    frenet_d = 101.0
    frenet_s = -1
  else:
    n_x = mapx[next_wp] - mapx[prev_wp]
    n_y = mapy[next_wp] - mapy[prev_wp]
    if n_x == 0:
      next_wp += 1
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
    frenet_d = sqrt(dx*dx + dy*dy)

    ego_vec = np.array([e-mapx[prev_wp], n-mapy[prev_wp], 0])
    map_vec = np.array([n_x, n_y, 0])
    d_cross = np.cross(ego_vec, map_vec)
    if d_cross[-1] > 0:
        frenet_d = -frenet_d

    # get frenet s
    frenet_s = maps[prev_wp] + sqrt(proj_x*proj_x + proj_y*proj_y)


  return [frenet_s, frenet_d]


cpdef list xy2frenet_with_closest_waypoint_loop(double e, double n, int closest_waypoint, c_np.ndarray[double, ndim=1] mapx, c_np.ndarray[double, ndim=1] mapy, c_np.ndarray[double, ndim=1] maps):
  cdef c_np.ndarray[double, ndim=1] ego_vec, map_vec, d_cross
  cdef double direction, frenet_d, frenet_s, n_x, n_y, x_x, x_y, proj_norm, proj_x, proj_y, dx, dy
  cdef int next_wp, prev_wp

  if closest_waypoint < 0:
    closest_waypoint = 0

  if closest_waypoint >= len(mapx) -1:
    closest_waypoint = len(mapx) -2

  map_vec = np.array([mapx[closest_waypoint + 1] - mapx[closest_waypoint],
                      mapy[closest_waypoint + 1] - mapy[closest_waypoint]])
  ego_vec = np.array([e - mapx[closest_waypoint], n - mapy[closest_waypoint]])
  direction = np.sign(np.dot(map_vec, ego_vec))
  if direction >= 0:
    next_wp = closest_waypoint + 1
  else:
    next_wp = closest_waypoint

  prev_wp = next_wp - 1

  if next_wp == len(mapx):
    next_wp = 0

  n_x = mapx[next_wp] - mapx[prev_wp]
  n_y = mapy[next_wp] - mapy[prev_wp]
  if n_x == 0:
    next_wp += 1
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
  frenet_d = sqrt(dx*dx + dy*dy)

  ego_vec = np.array([e-mapx[prev_wp], n-mapy[prev_wp], 0])
  map_vec = np.array([n_x, n_y, 0])
  d_cross = np.cross(ego_vec, map_vec)
  if d_cross[-1] > 0:
      frenet_d = -frenet_d

  # get frenet s
  frenet_s = maps[prev_wp] + sqrt(proj_x*proj_x + proj_y*proj_y)


  return [frenet_s, frenet_d]

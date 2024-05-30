import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from utils import interpolate_map

MAPFILE_PATH = '/home/mmc_note_18/Desktop/kiapi_localization/mcar_ws/src/localization/gps_system_localizer/mapfiles/KIAPI_CONTROL_TEAM'

main_1 = sio.loadmat(MAPFILE_PATH + '/mainlane_1.mat')
main_2 = sio.loadmat(MAPFILE_PATH + '/mainlane_2.mat')
merge_in = sio.loadmat(MAPFILE_PATH + '/merge_in.mat')
merge_out = sio.loadmat(MAPFILE_PATH + '/merge_out.mat')

target_roads = [main_1, main_2, merge_in, merge_out]

fig = plt.figure(figsize=(10, 5))

for road in target_roads:
    e = road['east'][0]
    n = road['north'][0]
    s = road['station'][0]

    plt.subplot(1,2,1)
    plt.plot(e, n, '.')

    plt.subplot(1,2,2)
    e_interp, n_interp, s_interp = interpolate_map(e,n,s,interp_gap=0.5)
    print(s_interp)
    plt.plot(e_interp, n_interp, '.')


plt.axis('equal')
plt.grid()

plt.subplot(1,2,1)
plt.axis('equal')
plt.grid()

plt.show()

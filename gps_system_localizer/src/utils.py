#-*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt

def interpolate_map(x, y, s, interp_gap):
    max_s = s[-1]
    num = int(max_s/interp_gap) + 1
    s_interp = np.linspace(0.0, max_s, num, endpoint=True)
    x_interp = np.interp(s_interp, s, x)
    y_interp = np.interp(s_interp, s, y)

    return x_interp, y_interp, s_interp


def distance2curve(curvexy, mapxy):
    # get the dimension of the space our points live in
    n, p = np.shape(curvexy)
    m = np.shape(mapxy)[0]

    # compute the chordal linear arclengths, and scale to [0,1].
    seglen = np.sqrt(np.sum(np.power(np.diff(curvexy, axis=0), 2.0), axis=1))
    t0 = np.insert(np.cumsum(seglen)/np.sum(seglen), 0, 0.0)

    # We need to build some parametric splines.
    # compute the splines, storing the polynomials in one 3-d array
    ppsegs = np.zeros((p, (n-1), 2))

    # the breaks for the splines will be t0, unless spline got fancy
    # on us here.
    breaks = t0
    for i in range(p):
        dt = np.diff(t0)
        ind = np.arange(n-1)
        a0 = (curvexy[ind+1,i] - curvexy[ind,i]) / dt
        a1 = curvexy[ind, i]
        ppsegs[i] = np.vstack((a0, a1)).T

    # how many breaks did we find in the spline? This is
    # only a thing to worry about for a spline based on few points,
    # when the function spline.m may choose to use only two breaks.
    nbr = len(breaks)

    # for each point in mapxy, find the closest point to those
    # in curvexy. This part we can do in a vectorized form.
    prowindex, pcolindex, distance = ipdm(mapxy, curvexy)

    # initialize the return variables, using the closest point
    # found in the set curvexy.
    xy = curvexy[pcolindex]
    t = t0[pcolindex]

    for i in range(n-1):
        # the i'th segment of the curve
        t1 = t0[i]
        t2 = t0[i+1]

        # Compute the location (in t) of the minimal distance
        # point to mapxy, for all points.
        tnum = np.zeros(m)
        tden = 0.0

        for j in range(p):
            ppj = ppsegs[j]
            tden = tden + np.power(ppj[i,0], 2.0)
            tnum = tnum + ppj[i,0] * (mapxy[:,j] - ppj[i,1])

        tmin = tnum / tden

        # We only care about those points for this segment where there
        # is a minimal distance to the segment that is internal to the
        # segment.
        k = [j for j, _bool in enumerate((tmin > 0) & (tmin < (t2-t1))) if _bool == True]
        nk = len(k)

        if nk > 0:
            # for any points with a valid minimum distance inside the
            # segment itself, compute that distance.
            dmin = np.zeros(nk)
            xymin = np.zeros((nk, p))
            for j in range(p):
                ppj = ppsegs[j]
                xymin[:,j] = ppj[i,0] * tmin[k] + ppj[i,1]
                dmin = dmin + np.power(xymin[:,j] - mapxy[k,j], 2.0)
            dmin = np.sqrt(dmin)
            L = dmin < distance[k]

            # this segment has a closer point
            # closest point from curvexy.
            if any(L):
                ik = [j for j, _bool in enumerate(L) if _bool == True]
                kl = [k[i] for i in ik]
                distance[kl] = dmin[ik]
                t[kl] = tmin[kl] + t0[i]
                xy[kl,:] = xymin[ik, :]
    # for the linear case, t is identical to the fractional arc length
    # along the curve.
    t_a = t

    return xy, distance, t_a

def ipdm(data1, data2):
    """ipdm: Inter-Point Distance Matrix"""
    n1, dim = np.shape(data1)
    n2 = len(data2)

    # Compute inter-point distances
    a1 = np.reshape(data1, [n1, 1, dim])
    a2 = np.reshape(data2, [1, n2, dim])
    bsx = np.power((a1[...,:] - a2), 2.0)
    dist = np.sqrt(np.sum(bsx, axis=2))

    # nearest
    columnindex = np.argmin(dist, axis=1)
    distance = np.array([dist[i,j] for i,j in enumerate(columnindex)])
    rowindex = np.arange(n1)

    return rowindex, columnindex, distance

def test_example():
    curvexy = np.array([[0,0], [1,0], [2,1], [0, 0.5], [0, 0]])
    mapxy = np.array([[3,4], [0.5, 0.5], [3, -1]])
    xy, distance, t = distance2curve(curvexy, mapxy)
    print("xy")
    print(xy)
    print("distance")
    print(distance)
    print("t")
    print(t)

    plt.plot(curvexy[:,0], curvexy[:,1], 'b.-', label='curvexy')
    plt.plot(mapxy[:,0], mapxy[:,1], 'ro', label='mapxy')
    plt.plot(xy[:,0], xy[:,1], 'kx', label='maped on curve',
             markersize=10, markeredgewidth=2)
    plt.xlim([-2, 5])
    plt.ylim([-2, 5])
    plt.grid()
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    test_example()

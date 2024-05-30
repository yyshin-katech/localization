import glob
import scipy.io as sio
import matplotlib.pyplot as plt

matfiles = glob.glob("./*.mat")

for matfile in matfiles:
    mat = sio.loadmat(matfile)
    easts = mat["east"][0]
    norths = mat["north"][0]
    stations = mat["station"][0]

    plt.plot(easts, norths, '.-')
    print("[%s] station[0] = %.2f, stations[-1] = %.2f" % (matfile, stations[0], stations[-1]))

plt.axis("equal")
plt.show()

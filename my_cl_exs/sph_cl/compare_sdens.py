import numpy as np
import pylab as pl
import sys

def main():
    acpu = np.fromfile(sys.argv[1],dtype=np.float32)
    agpu = np.fromfile(sys.argv[2],dtype=np.float32)

    levels = [1e2,1e3,1e4,1e5,1e6,1e7]

    pl.figure(figsize=(10,10))
    pl.contour(acpu.reshape((np.sqrt(len(acpu)),np.sqrt(len(acpu)))),levels)
    #pl.figure(figsize=(10,10))
    #pl.contour(agpu.reshape((np.sqrt(len(agpu)),np.sqrt(len(agpu)))),levels)
    pl.show()

if __name__ == '__main__':
    main()



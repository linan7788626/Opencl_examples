import numpy as np
import pylab as pl
#import scipy.signal as ss
#from mycosmology import *
#from scipy import integrate
#import alens_arr as aa

#--------------------------------------------------------------------
boxsize = 1.0
nnn = 1024
dbx = boxsize/nnn
zl = 0.2
zs = 1.0
#--------------------------------------------------------------------
#file_in = "snfw_1e5.dat"
#x1,x2,x3 = np.loadtxt("../data/"+file_in,usecols = (0,1,2),unpack=True)
#print np.max(x1)
#print np.max(x2)
#print np.max(x3)

file_in = "cnfw_2e4.dat"
x1,x2,x3 = np.loadtxt("../input_files/"+file_in,usecols = (0,1,2),unpack=True)
print np.max(x1)
print np.max(x2)
print np.max(x3)

#np.savetxt("cnfw_2e4.dat",np.transpose((x1[::10],x2[::10],x3[::10])),fmt="%.8e",comments="#")

##xall = np.zeros((len(x1),3))
##xall[:,0] = x1
##xall[:,1] = x2
##xall[:,2] = x3
##xall = xall.reshape((3*len(x1)))
##print np.max(xall),np.min(xall)
##xallt = np.array(xall,dtype=np.float32)
##xallt.tofile("snfw_1e5_for_test.bin")
##print np.max(xallt),np.min(xallt)
##
##xall0=np.fromfile('./snfw_1e5_for_test.bin',dtype=np.float32)
##print np.max(xall0),np.min(xall0)
##xall0=xall0.reshape((len(x1),3))
##print np.shape(xall0)
##
pl.figure()
pl.plot(x1[::10],x2[::10],'r.')
pl.figure()
pl.plot(x2[::10],x3[::10],'g.')
pl.figure()
pl.plot(x3[::10],x1[::10],'b.')
pl.show()

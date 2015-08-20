import pyopencl as cl
import pyopencl.clrandom
import numpy as np
nsamples = int (12e6)
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

xy = pyopencl.clrandom.rand(ctx,queue,(nsamples,2),np.float32)


xy = xy**2

print(4.0*np.sum(np.sum(xy.get(),1)<1)/nsamples)



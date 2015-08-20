#!/usr/bin/env python

'''
Testing the efficiency of different implimentations of the Deeming
periodogram. Its an O(N*N) algorithm for calculating a periodogram but
it works for unevenly sampled data. Used frequently in astronomical
time-series.

cyperiodogram contains a cython implimentation

periodogram.f90 contains two Fortran 90 implimentations
  1 : Using the same array syntax as numpy
  2 : Hand coded loops (Much faster in this case)

The fortran subroutines use openmp and are wrapped using f2py. Look in
build_deeming.sh for the compilation commands.

The OpenCL version uses deeming_kernel.cl for computations on the opencl device
(A GPU in my case)

To run this example (in Linux):
  sh build_deeming.sh
  python deeming.py

The cython extension will be compiled when running the deeming.py file.



'''

import numpy as np
import deemingomp
import time
import pyximport; pyximport.install()
import cyperiodogram as cy
import pyopencl as cl
import matplotlib.pyplot as plt
import matplotlib.style

matplotlib.style.use('ggplot')


def timeit(method):
    '''
    From: http://www.samuelbosch.com/2012/02/timing-functions-in-python.html

    '''
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        # print('%r %2.2f sec' % (method.__name__, te-ts))
        return result, (te-ts)

    return timed


# Deeming periodogram done with numpy
@timeit
def periodogram_numpy(t, m, freqs):
    ''' Calculate the Deeming periodogram using numpy
    '''
    pi = np.pi
    amps = np.zeros(freqs.size, dtype='float')
    cos = np.cos
    sin = np.sin
    for i, f in enumerate(freqs):
        real = (m*cos(2*pi*f*t)).sum()
        imag = (m*sin(2*pi*f*t)).sum()
        amps[i] = real**2 + imag**2

    amps = 2.0*np.sqrt(amps)/t.size
    return amps


@timeit
def periodogram_cython(t, m, f):
    ''' Calculate Deeming periodogram using cython: see cyperiodogram.pyx
    '''
    cyamps = cy.periodogram(t, m, f)
    return cyamps


@timeit
def periodogram_fortran_openmp(t, m, f, threads):
    ''' Calculate the Deeming periodogram using Fortran with OpenMP
    '''
    ampsf90omp_2 = deemingomp.periodogram2(t, m, f, t.size, f.size, threads)

    return ampsf90omp_2


@timeit
def periodogram_opencl(t, m, f):
    ''' Calculate the Deeming periodogram using OpenCL on the GPU
    '''

    # create a context and a job queue
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # create buffers to send to device
    mf = cl.mem_flags
    # input buffers
    times_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t)
    mags_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m)
    freqs_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)

    # output buffers
    amps_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, f.nbytes)
    amps_g = np.empty_like(f)

    # read and compile the opencl kernel
    with open('deeming_kernel.cl') as kernel:
        prg = cl.Program(ctx, kernel.read()).build()
        try:
            prg.build()
        except:
            print("Error:")
            print(prg.get_build_info(ctx.devices[0],
                                     cl.program_build_info.LOG))
            raise

    # call the function and copy the values from the buffer to a numpy array
    prg.periodogram(queue, amps_g.shape, None,
                    times_g,
                    mags_g,
                    freqs_g,
                    amps_buffer,
                    np.int32(t.size))
    cl.enqueue_copy(queue, amps_g, amps_buffer)

    return amps_g


@timeit
def run_benchmark(N, M):
    ''' Run the Deeming benchmark using all the implementations using N data
    points and M frequencies

    '''

    pi = np.pi
    # Data
    t = np.linspace(0.0, 0.25, N)
    m = 1.75*np.sin(2*pi*150*t) + 0.75*np.sin(2*pi*277*t) + np.sin(2*pi*333*t)

    # Frequencies
    freqs = np.linspace(0.0, 1000.0, M)

    # Run the different versions

    _times = {}

    # numpy
    amps, t_numpy = periodogram_numpy(t, m, freqs)

    # cython
    cyamps, t_cython = periodogram_cython(t, m, freqs)
    assert(np.allclose(cyamps, amps))

    # Fortran with openmp
    for threads in [1, 2, 4, 8]:
        ampsf90omp_2, t_fortran = periodogram_fortran_openmp(t,
                                                             m,
                                                             freqs,
                                                             threads)
        _times['fortran {}'.format(threads)] = t_fortran
        assert(np.allclose(ampsf90omp_2, amps))

    # opencl
    opencl_amps, t_opencl = periodogram_opencl(t, m, freqs)
    assert(np.allclose(opencl_amps, amps))

    _times['numpy'] = t_numpy
    _times['cython'] = t_cython
    _times['opencl'] = t_opencl

    return _times


def make_bar_plot(N, times_dict):
    ''' Make a bar plot using the timing dict obtained from run_benchmarks
    '''
    speedups = []
    times = []
    methods = ['numpy', 'cython', 'fortran 1', 'fortran 2', 'fortran 4',
               'fortran 8', 'opencl']

    for i, key in enumerate(methods):
        speedups.append(times_dict['numpy']/times_dict[key])
        times.append(times_dict[key])

    index = np.arange(len(methods))
    fig, ax = plt.subplots()
    rect1 = ax.bar(index, speedups)
    plt.xlabel('Method', fontsize=18)
    plt.ylabel('Speedup (higher is better)', fontsize=18)
    plt.xticks(index + 0.4, methods, fontsize=14)

    def autolabel(rects, times):
        # attach some text labels
        for rect, _time in zip(rects, times):
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.02*height, '%1.1fs'% _time,
                    ha='center', va='bottom')

    autolabel(rect1, times)
    # save to file
    plt.savefig('{}x{}-barchart.jpg'.format(N, N))


if __name__ == "__main__":

    for N in [1000, 2000, 4000, 8000, 16000, 32000, 64000]:
        _times, _time_all = run_benchmark(N, N)
        make_bar_plot(N, _times)
        print("\n{} datapoints, {} frequencies in {}s".format(N, N, round(_time_all, 1)))
        for key in sorted(_times):
            print("{}: {} {} {}".format(key,
                                        " "*(20-len(key)),
                                        round(_times[key], 3),
                                        round(_times['numpy']/_times[key], 1)))

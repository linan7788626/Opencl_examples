#!/bin/bash
f2py -c -m deeming periodogram.f90 -lgomp
f2py -c -m deemingomp periodogram.f90 --f90flags="-fopenmp " -lgomp
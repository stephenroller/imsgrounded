#!/bin/bash
# on OS X 10.7
#gcc -fPIC -I /usr/include/python2.7 -I /Library/Python/2.7/site-packages/numpy-1.8.0.dev_436a28f_20120710-py2.7-macosx-10.7-x86_64.egg/numpy/core/include -c xmod.c -o xmod.o -O3 -finline-functions -ffast-math
# on OS X 10.8
gcc -fPIC -I /usr/include/python2.7 -I /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include -c xmod.c -o xmod.o -O3 -finline-functions -ffast-math
gcc -shared -o xmod.so xmod.o -lgsl -lgslcblas -lm -lpython

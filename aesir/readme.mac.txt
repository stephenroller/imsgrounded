#!/bin/bash
gcc -fPIC -I /usr/include/python2.7 -I /Library/Python/2.7/site-packages/numpy-1.8.0.dev_436a28f_20120710-py2.7-macosx-10.7-x86_64.egg/numpy/core/include -c xmod.c -o xmod.o
gcc -shared -o xmod.so xmod.o -lgsl -lgslcblas -lm -lpython

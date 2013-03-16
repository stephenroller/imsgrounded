gcc -fPIC -I $TACC_PYTHON_DIR/include/python2.7 -I $TACC_PYTHON_DIR/lib/python2.7/site-packages/numpy/core/include -I $TACC_GSL_INC -c xmod.c -o xmod.o -O3 -finline-functions -ffast-math
gcc -shared -o xmod.so xmod.o -lgsl -L$TACC_GSL_DIR/lib -lgslcblas -lm -L$TACC_PYTHON_LIB -lpython2.7

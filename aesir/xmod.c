#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <limits.h>
#include <gsl/gsl_rng.h>
#include "fastapprox.h"

//#define MAGIC_GAMMA_CONSTANT (1.7976931348623157e+308)
#define MAGIC_GAMMA_CONSTANT FLT_MAX
#define GAMMA_THRESH (1e-9)

/* create a random number generator object */
gsl_rng *random_number_generator;

static inline float fasttrigamma(float x) {
  float p;
  x=x+6;
  p=1/(x*x);
  p=(((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
       *p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
  int i;
  for (i=0; i<6 ;i++) {
    x=x-1;
    p=1/(x*x)+p;
  }
  return p;
}

static PyObject* a_update(PyObject *self, PyObject *args) {
  double old_a, m, sum_logdatamean, iteration_eps;
  int max_iter, J;

  if (!PyArg_ParseTuple(args, "dddiid", &old_a, &m, &sum_logdatamean, &J, &max_iter, &iteration_eps)) {
    return NULL;
  }

  int i;
  double delta = DBL_MAX;
  double a = old_a;
  double d1, d2, am;
  double inva;

  for (i=0; i<max_iter && delta > iteration_eps; i++) {
    old_a = a;
    if (a < FLT_MIN) {
      a = FLT_MIN;
    }
    am = a * m;
    d1 = J * fastdigamma(a) - m * fastdigamma(am) + m * sum_logdatamean;
    d2 = J * fasttrigamma(a) - (m * m * fasttrigamma(am));

    inva =((1.0 / a) + (d1 / d2) * a * a);
    a = 1.0/inva;
    if (a < FLT_MIN) {
      a = old_a;
    }
    delta = abs(old_a - a);
  }

  return PyFloat_FromDouble(a);
}


double lnsumexp(double xarray[], int n) {
  int i;
  double x,m,y;

  m = xarray[0];
  for (i=1; i<n; i++) {
    if (xarray[i] > m)
      m = xarray[i];
  }

  /* add up the exp of every element minux max */
  y = 0;
  for (i=0; i<n; i++) {
    y += exp(xarray[i] - m);
  }

  /* return max + log(sum(exp((x-max(x))))) */
  return m + log(y);
}

static PyObject *xfactorialposterior(PyObject *self, PyObject *args) {
  PyArrayObject *logphi_array,*logpsi_array,*logpi_array,*data_array,*x_array,*Rphi_array,*Rpsi_array,*S_array;
  int Nj,F,K,i,k,v,g,f;
  double rand_x,s,z;
  int D,J;

  int lastv = INT_MAX, lastf = INT_MAX, lastg = INT_MAX;

  if (!PyArg_ParseTuple(args, "O!O!O!O!iiiii",
    &PyArray_Type, &logphi_array,
    &PyArray_Type, &logpsi_array,
    &PyArray_Type, &logpi_array,
    &PyArray_Type, &data_array,
    &Nj,
    &D,
    &F,
    &J,
    &K)) {
      return NULL;
  }

  int dims_Rphi[2] = {K, D};
  int dims_Rpsi[2] = {K, F};
  int dims_S[2] = {J, K};

  double Z = 0;

  Rphi_array =(PyArrayObject *) PyArray_FromDims(2,dims_Rphi,NPY_INT);
  Rpsi_array =(PyArrayObject *) PyArray_FromDims(2,dims_Rpsi,NPY_INT);
  S_array =(PyArrayObject *) PyArray_FromDims(2,dims_S,NPY_INT);

  /* seed the random number generator*/
  gsl_rng_set(random_number_generator,time(NULL));

  double f_array[K];
  double sz_array[K];

  for (i=0; i<Nj; i++) {
    // vocab item
    v=*((int *)(data_array->data + 1*data_array->strides[0] + i*data_array->strides[1]));
    // feature item
    f=*((int *)(data_array->data + 2*data_array->strides[0] + i*data_array->strides[1]));
    // docid
    g=*((int *)(data_array->data + 0*data_array->strides[0] + i*data_array->strides[1]));

    // if the last v, g and f are all the same, then this
    // next calculation will be the same, so let's reuse it.
    if (!(v == lastv && g == lastg && f == lastf)) {
      // got here, so we couldn't reuse it.
      for (k=0; k<K; k++) {
        f_array[k] =
            *((double *)(logphi_array->data + k*logphi_array->strides[0] + v*logphi_array->strides[1])) +
            *((double *)(logpsi_array->data + k*logpsi_array->strides[0] + f*logpsi_array->strides[1])) +
            *((double *)(logpi_array->data +  g*logpi_array->strides[0] +  k*logpi_array->strides[1]));
      }

      lastv = v;
      lastg = g;
      lastf = f;

      z = lnsumexp(f_array, K);
      s = 0;
      for (k = 0; k<K; k++) {
        s += exp(f_array[k] - z);
        sz_array[k] = s;
      }
    }

    Z += z;

    rand_x = gsl_rng_uniform(random_number_generator);
    /* sample from exp(f_array[0]-z) */
    for (k=0; k < K && rand_x >= sz_array[k]; k++);

    *((int *)(Rphi_array->data + k*Rphi_array->strides[0] + v*Rphi_array->strides[1] ))+=1;
    *((int *)(Rpsi_array->data + k*Rpsi_array->strides[0] + f*Rpsi_array->strides[1] ))+=1;
    *((int *)(S_array->data + g*S_array->strides[0] + k*S_array->strides[1] ))+=1;

  }

  return Py_BuildValue("(NNNd)", Rphi_array,Rpsi_array,S_array,Z);

}


static PyMethodDef xmod_methods[] = {
  {"a_update", a_update, METH_VARARGS},
  {"xfactorialposterior", xfactorialposterior, METH_VARARGS},
  {NULL, NULL} // required ending of the method table
};


PyMODINIT_FUNC initxmod() {
  PyObject *mod = Py_InitModule("xmod", xmod_methods);
  // initialize the RNG
  random_number_generator = gsl_rng_alloc (gsl_rng_mt19937);

  // required NumPy initialization */
  import_array();
}



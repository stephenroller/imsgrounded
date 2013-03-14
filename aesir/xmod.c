#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_rng.h>

double max(PyArrayObject *array) {
  int m,x;
  int i,n;

  n=array->dimensions[0];
  m = *(((int *)array->data));
  for (i = 1; i < n; i++) {
    x= *((int *)(array->data + i*array->strides[0] ));
    if (x > m) { m=x; }
  }
  return m;
}

double lnsumexp(double xarray[], int n){
  int i;
  double x,m,y,z;

  m=xarray[0];
  for (i=1;i<n;i++) {
    if (xarray[i]>m) {
      m = xarray[i];
    }
  }

  /* add up the exp of every element minux max */
  y=0;
  for (i=0;i<n;i++) {
    y += exp(xarray[i]-m);
  }

  /* return max + log(sum(exp((x-max(x))))) */
  z= m + log(y);

  return z;
}


double logsumexp(PyArrayObject *xarray) {
  int i,n;
  double x,m,y,z;

  /* get the dimensionality of x */
  n=xarray->dimensions[0];

  /*get the max of x-array */
  m=*(double *)(xarray->data);
  for (i=1; i<n; i++) {
    x= *((double *)(xarray->data + i*xarray->strides[0] ));
    if (x>m) {
      m = x;
    }
  }
  /* add up the exp of every element minux max */
  for (i=0;i<n;i++) {
    y+=exp(*((double *)(xarray->data + i*xarray->strides[0] ))-m);
  }

  /* return max + log(sum(exp((x-max(x))))) */
  z = m + log(y);

  return z;
}


static PyObject *xfactorialposterior(PyObject *self, PyObject *args) {
  PyArrayObject *phi_array,*psi_array,*pi_array,*data_array,*x_array,*Rphi_array,*Rpsi_array,*S_array;
  int Nj,F,K,i,k,dims_data[1],dims_kslice[2],v,g,f;
  double rand_x,s,z;
  int D,J;

  if (!PyArg_ParseTuple(args, "O!O!O!O!iiiii",
    &PyArray_Type, &phi_array,
    &PyArray_Type, &psi_array,
    &PyArray_Type, &pi_array,
    &PyArray_Type, &data_array,
    &Nj,
    &D,
    &F,
    &J,
    &K)) {
      return NULL;
  }

  dims_data[0]=Nj;
  dims_kslice[0]=K;
  dims_kslice[1]=Nj;

  int dims_Rphi[2];
  int dims_Rpsi[2];
  int dims_S[2];

  dims_Rphi[0]=K;
  dims_Rphi[1]=D;

  dims_Rpsi[0]=K;
  dims_Rpsi[1]=F;

  dims_S[0]=J;
  dims_S[1]=K;

  double Z=0;

  Rphi_array =(PyArrayObject *) PyArray_FromDims(2,dims_Rphi,NPY_INT);
  Rpsi_array =(PyArrayObject *) PyArray_FromDims(2,dims_Rpsi,NPY_INT);
  S_array =(PyArrayObject *) PyArray_FromDims(2,dims_S,NPY_INT);

  /* create a random number generator object */
  gsl_rng *r = gsl_rng_alloc (gsl_rng_mt19937);

  /* seed the random number generator*/
  gsl_rng_set(r,time(NULL));

  double f_array[K];
  double p_array[K];

  for (i=0;i<Nj;i++) {
    v=*((int *)(data_array->data + 2*data_array->strides[0] + i*data_array->strides[1]  ));
    f=*((int *)(data_array->data + 3*data_array->strides[0] + i*data_array->strides[1]  ));
    g=*((int *)(data_array->data + 0*data_array->strides[0] + i*data_array->strides[1]  ));

    for (k=0;k<K;k++) {
      f_array[k] =
          log(*((double *)(phi_array->data + k*phi_array->strides[0] + v*phi_array->strides[1])))
          + log(*((double *)(psi_array->data + k*psi_array->strides[0] + f*psi_array->strides[1])))
          + log(*((double *)(pi_array->data + g*pi_array->strides[0] + k*pi_array->strides[1])));
    }

    z=lnsumexp(f_array,K);
    Z+=z;

    rand_x=gsl_rng_uniform(r);
    s=exp(f_array[0]-z);

    k = 0;
    /* sample from exp(f_array[0]-z) */
    while ((rand_x>=s) && (k<K)) {
      k++;
      s+=exp(f_array[k]-z);
    }

    *((int *)(Rphi_array->data + k*Rphi_array->strides[0]  + v*Rphi_array->strides[1] ))+=1;
    *((int *)(Rpsi_array->data + k*Rpsi_array->strides[0]  + f*Rpsi_array->strides[1] ))+=1;
    *((int *)(S_array->data + g*S_array->strides[0]  + k*S_array->strides[1] ))+=1;

  }

  return Py_BuildValue("(NNNd)", Rphi_array,Rpsi_array,S_array,Z);

  gsl_rng_free(r);

}


static PyObject *indsum(PyObject *self, PyObject *args) {
  PyArrayObject *parray, *iarray, *sarray;
  int i,n,N,dims[2];
  double xi;
  int ii,j,k;

  if (!PyArg_ParseTuple(args, "iO!O!:indsum",
    &N,
    &PyArray_Type, &iarray,
    &PyArray_Type, &parray)) {
      return NULL;
  }

  if (parray->nd > 2) {
    PyErr_SetString(PyExc_ValueError,"parray should be no more than 2d");
    return NULL;
  }

  if (parray->nd ==1)
    k=1;
  else
    k=parray->dimensions[1];

  if (iarray->nd !=1 ) {
    PyErr_SetString(PyExc_ValueError,"iarray needs to be 1d");
    return NULL;
  }

  if (parray->dimensions[0] != iarray->dimensions[0]) {
    PyErr_SetString(PyExc_ValueError,"parray and iarray need to be of same length");
    return NULL;
  }

  if (parray->descr->type_num != NPY_DOUBLE) {
    PyErr_SetString(PyExc_ValueError,"parray should be floats");
    return NULL;
  }

  if (iarray->descr->type_num != NPY_LONG && iarray->descr->type_num != NPY_INT) {
    PyErr_SetString(PyExc_ValueError,"iarray should be integers");
    return NULL;
  }

  dims[0]=N;
  dims[1]=k;

  sarray =(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);

  for (i=0; i<N; i++) {
    for (j=0; j<k; j++) {
      *((double *)(sarray->data + i*sarray->strides[0] + j*sarray->strides[1])) = 0.0;
    }
  }

  n=parray->dimensions[0];

  for (i=0; i<n; i++) {
    ii=*((int *)(iarray->data + i*iarray->strides[0]));
    for (j=0; j<k; j++) {
      xi=*((double *)(parray->data + i*parray->strides[0] + j*parray->strides[1] ));
      *((double *)(sarray->data + ii*sarray->strides[0]  + j*sarray->strides[1] ))+=xi;
    }
  }

  return PyArray_Return(sarray);
}



static PyMethodDef xmod_methods[] = {
  {"indsum",indsum, METH_VARARGS},
  {"xfactorialposterior",xfactorialposterior,METH_VARARGS},
  {NULL, NULL}     /* required ending of the method table */
};


PyMODINIT_FUNC initxmod() {
  Py_InitModule("xmod", xmod_methods);
  import_array();   /* required NumPy initialization */
}



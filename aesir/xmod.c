#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_rng.h>


/* create a random number generator object */
gsl_rng *random_number_generator;

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

  /* seed the random number generator*/
  gsl_rng_set(random_number_generator,time(NULL));

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

    rand_x = gsl_rng_uniform(random_number_generator);
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

}


static PyMethodDef xmod_methods[] = {
  {"xfactorialposterior",xfactorialposterior,METH_VARARGS},
  {NULL, NULL}     /* required ending of the method table */
};


PyMODINIT_FUNC initxmod() {
  Py_InitModule("xmod", xmod_methods);
  // required NumPy initialization */
  import_array();
  // initialize the RNG
  random_number_generator = gsl_rng_alloc (gsl_rng_mt19937);
}



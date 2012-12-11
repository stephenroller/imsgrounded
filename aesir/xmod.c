#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_rng.h>




double max(PyArrayObject *array)
{
int m,x;
int i,n;

	n=array->dimensions[0];
	m = *(((int *)array->data));
	
	for (i = 1; i < n; i++){
		x= *((int *)(array->data + i*array->strides[0] ));
			if (x > m){m=x;}
		}
	
	return m;
}

double digamma(double x)
{
	double p;
	x=x+6;
	p=1/(x*x);
	p=(((0.004166666666667*p-0.003968253986254)*p+
	0.008333333333333)*p-0.083333333333333)*p;
	p=p+log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
	return p;
}

double trigamma(double x)
{
    double p;
    int i;

    x=x+6;
    p=1/(x*x);
    p=(((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
         *p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
    for (i=0; i<6 ;i++)
    {
        x=x-1;
        p=1/(x*x)+p;
    }
    return(p);
}


double lnsumexp(double xarray[], int n){
	int i;
	double x,m,y,z;

		/* get the dimensionality of x 
		n=xarray->dimensions[0];*/

		
		/*get the max of x-array 
		m=*(double *)(xarray->data);*/
		m=xarray[0];
		for (i=1;i<n;i++){
				if (xarray[i]>m){m=xarray[i];}
			}
		/* add up the exp of every element minux max */
		y=0;
		for (i=0;i<n;i++){
			y+=exp(xarray[i]-m);
			}
		    
		/* return max + log(sum(exp((x-max(x))))) */
		z=m+log(y);

		return z;
}


double logsumexp(PyArrayObject *xarray){
	int i,n;
	double x,m,y,z;

		/* get the dimensionality of x */
		n=xarray->dimensions[0];

		
		/*get the max of x-array */
		m=*(double *)(xarray->data);
		for (i=1;i<n;i++){
			x= *((double *)(xarray->data + i*xarray->strides[0] ));
				if (x>m){m=x;}
			}
		/* add up the exp of every element minux max */
		for (i=0;i<n;i++){
			y+=exp(*((double *)(xarray->data + i*xarray->strides[0] ))-m);
			}
		    
		/* return max + log(sum(exp((x-max(x))))) */
		z=m+log(y);

		return z;
}


static PyObject *psi(PyObject *self, PyObject *args){
	PyArrayObject *xarray,*yarray;
	int n,m,i,j;
	double x,y;

	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &xarray)) 
		{return NULL;}
		

		if (xarray->nd == 1) {
			int dims[1];
			n=dims[0]=xarray->dimensions[0];
	
			/* Make a new array here, same size as input array */
			yarray =(PyArrayObject *) PyArray_FromDims(1,dims,NPY_DOUBLE);
		
	
			for (i=0;i<n;i++) {
			        x= *((double *)(xarray->data + i*xarray->strides[0] ));
				y=digamma(x);
				*((double *)(yarray->data + i*yarray->strides[0]))=y;

			}

		}		
		
		if (xarray->nd == 2) {
			int dims[2];

			n=dims[0]=xarray->dimensions[0];
			m=dims[1]=xarray->dimensions[1];
	
			/* Make a new array here, same size as input array */
			yarray =(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
	
			for (i=0;i<n;i++) {
				for (j=0;j<m;j++) {
				x=*((double *)(xarray->data + i*xarray->strides[0]  + j*xarray->strides[1] ));
				y=digamma(x);
				*((double *)(yarray->data + i*yarray->strides[0]  + j*yarray->strides[1] ))=y;
				}
			}
		}

		return PyArray_Return(yarray);
}
	
static PyObject *tripsi(PyObject *self, PyObject *args){
	PyArrayObject *xarray,*yarray;
	int n,m,i,j;
	double x,y;

	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &xarray)) 
		{return NULL;}
		

		if (xarray->nd == 1) {
			int dims[1];
			n=dims[0]=xarray->dimensions[0];
	
			/* Make a new array here, same size as input array */
			yarray =(PyArrayObject *) PyArray_FromDims(1,dims,NPY_DOUBLE);
		
	
			for (i=0;i<n;i++) {
			        x= *((double *)(xarray->data + i*xarray->strides[0] ));
				y=trigamma(x);
				*((double *)(yarray->data + i*yarray->strides[0]))=y;

			}

		}		
		
		if (xarray->nd == 2) {
			int dims[2];

			n=dims[0]=xarray->dimensions[0];
			m=dims[1]=xarray->dimensions[1];
	
			/* Make a new array here, same size as input array */
			yarray =(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
	
			for (i=0;i<n;i++) {
				for (j=0;j<m;j++) {
				x=*((double *)(xarray->data + i*xarray->strides[0]  + j*xarray->strides[1] ));
				y=trigamma(x);
				*((double *)(yarray->data + i*yarray->strides[0]  + j*yarray->strides[1] ))=y;
				}
			}
		}

		return PyArray_Return(yarray);
}
	
	


static PyObject *lugsumexp(PyObject *self, PyObject *args){
	PyArrayObject *xarray,*yarray;
	int n,i,dims[1];
	double x,m,y,z;

	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &xarray)) 
		{return NULL;}
		

		if (xarray->nd != 1) {
		PyErr_SetString(PyExc_ValueError,"array should be 1d");
		return NULL;
		}
		
		/* get the dimensionality of x */
		n=dims[0]=xarray->dimensions[0];

		/* Make a new array here, same size as input array */
		yarray =(PyArrayObject *) PyArray_FromDims(1,dims,NPY_DOUBLE);
		
		/*get the max of x-array */
		m=*(double *)(xarray->data);
		for (i=1;i<n;i++){
			x= *((double *)(xarray->data + i*xarray->strides[0] ));
				if (x>m){m=x;}
			}

		for (i=0;i<n;i++){
			/* *((double *)(yarray->data + i*yarray->strides[0] ))=*((double *)(xarray->data + i*xarray->strides[0] ))-m; */
			y+=exp(*((double *)(xarray->data + i*xarray->strides[0] ))-m);
			}
		    
		z=m+log(y);

		return Py_BuildValue("d", z);


		/*return PyArray_Return(y);*/
}

static PyObject *xptest(PyObject *self, PyObject *args){

	PyArrayObject *phi_array,*pi_array,*data_array,*x_array,*r_array,*s_array;
	int Nj,K,i,k,dims_data[1],dims_kslice[2],v,g;
	double rand_x,s,z;
	int D,J;

	if (!PyArg_ParseTuple(args, "O!O!O!iiii", 
		&PyArray_Type, &phi_array,
		&PyArray_Type, &pi_array,
		&PyArray_Type, &data_array,
		&Nj,
		&D,
		&J,
		&K)) 
		{return NULL;}

	dims_data[0]=Nj;
	dims_kslice[0]=K;
	dims_kslice[1]=Nj;

	int dims_R[2];
	int dims_S[2];


	dims_R[0]=K;
	dims_R[1]=D;
	
	dims_S[0]=J;
	dims_S[1]=K;

	double Z=0;

	r_array =(PyArrayObject *) PyArray_FromDims(2,dims_R,NPY_INT);
	s_array =(PyArrayObject *) PyArray_FromDims(2,dims_S,NPY_INT);
	
	/* create a random number generator object */
	gsl_rng *r = gsl_rng_alloc (gsl_rng_mt19937);

	/* seed the random number generator*/ 
	gsl_rng_set(r,time(NULL));
	
	double f_array[K];
	double p_array[K];
	/*x_array =(PyArrayObject *) PyArray_FromDims(1,dims_data,NPY_INT);*/
/*	out_array =(PyArrayObject *) PyArray_FromDims(2,dims_kslice,NPY_DOUBLE);*/
	
	for (i=0;i<Nj;i++) {
		v=*((int *)(data_array->data + 2*data_array->strides[0] + i*data_array->strides[1]  ));
		g=*((int *)(data_array->data + 0*data_array->strides[0] + i*data_array->strides[1]  ));
	
	for (k=0;k<K;k++) {
				/*f_array[k] = *((double *)(phi_array->data +k*phi_array->strides[0] + v*phi_array->strides[1]  )) ;*/
				f_array[k] = log(*((double *)(phi_array->data + k*phi_array->strides[0] + v*phi_array->strides[1])))
				+ log(*((double *)(pi_array->data + g*pi_array->strides[0] + k*pi_array->strides[1])));

			}
	z=lnsumexp(f_array,K);
	Z+=z;
/*
	for (k=0;k<K;k++) {
			p_array[k]=exp(f_array[k]-z);
		}
*/

		rand_x=gsl_rng_uniform(r);
		/*s=p_array[0];*/
		s=exp(f_array[0]-z);
		
		k=0;
		while ((rand_x>=s) && (k<K)) {
			k++;
			/*s+=p_array[k];*/
			s+=exp(f_array[k]-z);
		}
		/*	*((int *)(x_array->data + i*x_array->strides[0] )) = k;*/

		
		*((int *)(r_array->data + k*r_array->strides[0]  + v*r_array->strides[1] ))+=1;
		*((int *)(s_array->data + g*s_array->strides[0]  + k*s_array->strides[1] ))+=1;
/*
		for (k=0;k<K;k++) {
			*((double *)(out_array->data + k*out_array->strides[0] + i*out_array->strides[1])) = p_array[k] ;
		}
*/
}

	/*return Py_BuildValue("d", *((double *)(phi_array->data)));*/
	/*return PyArray_Return(s_array);*/
	/*return Py_BuildValue("i",0);*/
	return Py_BuildValue("(NNd)", r_array,s_array,Z) ;
/*	Py_DECREF(r_array);*/
	/*Py_DECREF(s_array);*/

	gsl_rng_free(r);
}


static PyObject *xfactorialposterior(PyObject *self, PyObject *args){

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
		&K)) 
		{return NULL;}

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
			f_array[k] = log(*((double *)(phi_array->data + k*phi_array->strides[0] + v*phi_array->strides[1])))
					+ log(*((double *)(psi_array->data + k*psi_array->strides[0] + f*psi_array->strides[1])))
					+ log(*((double *)(pi_array->data + g*pi_array->strides[0] + k*pi_array->strides[1])));
				}
	z=lnsumexp(f_array,K);
	Z+=z;

		rand_x=gsl_rng_uniform(r);
		s=exp(f_array[0]-z);
		
		k=0;
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


static PyObject *xposterior(PyObject *self, PyObject *args){
	
	PyArrayObject *phi_array,*pi_array,*data_array,*f_array,*p_array,*x_array;
	int Nj,K,i,k,dims[1],dims_data[1],dims_kslice[1],v,g;
	double rand_x,s,z;

	if (!PyArg_ParseTuple(args, "O!O!O!ii", 
		&PyArray_Type, &phi_array,
		&PyArray_Type, &pi_array,
		&PyArray_Type, &data_array,
		&Nj,
		&K)) 
		{return NULL;}
	
	/* create a random number generator object */
	gsl_rng *r = gsl_rng_alloc (gsl_rng_mt19937);

	/* seed the random number generator*/ 
	gsl_rng_set(r,time(NULL));
	
	dims_data[0]=Nj;
	dims_kslice[0]=K;

	x_array =(PyArrayObject *) PyArray_FromDims(1,dims_data,NPY_INT);
	f_array =(PyArrayObject *) PyArray_FromDims(1,dims_kslice,NPY_DOUBLE);
	p_array =(PyArrayObject *) PyArray_FromDims(1,dims_kslice,NPY_DOUBLE);

	for (i=0;i<Nj;i++) {
			
		v=*((int *)(data_array->data + 2*data_array->strides[0] + i*data_array->strides[1]  ));
		g=*((int *)(data_array->data + 0*data_array->strides[0] + i*data_array->strides[1]  ));
			for (k=0;k<K;k++) {
				*((double *)(f_array->data + k*f_array->strides[0])) = 
				log( *((double *)(phi_array->data + k*phi_array->strides[0] + v*phi_array->strides[1])));
				+log( *((double *)(pi_array->data + g*pi_array->strides[0] + k*pi_array->strides[1])));

			}
			
	
		z=logsumexp(f_array);

			for (k=0;k<K;k++) {
				*((double *)(p_array->data + k*p_array->strides[0])) 
				= exp(*((double *)(f_array->data + k*f_array->strides[0])) - z);
			}


		rand_x=gsl_rng_uniform(r);
		s=*(double *)(p_array->data);
		
		k=0;
		while ((rand_x>=s) && (k<K)) {
			k++;
			s+=*(double *)(p_array->data + k*p_array->strides[0]);
		}
			*((int *)(x_array->data + i*x_array->strides[0] )) = k;

	}
	

	return PyArray_Return(p_array);
	
/*	return Py_BuildValue("i", K);*/
	/* free up the object */
	gsl_rng_free(r);
}

static PyObject *discretep(PyObject *self, PyObject *args){
	PyArrayObject *xarray,*yarray;
	double x,s;
	int N,j,i,n,dims[1];

	if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &xarray,&N)) 
		{return NULL;}
	
	if (xarray->nd != 1) {
		PyErr_SetString(PyExc_ValueError,"array should be 1d");
		return NULL;
				}
	
	gsl_rng *r = gsl_rng_alloc (gsl_rng_mt19937);

	/* seed the random number generator */
	gsl_rng_set(r,time(NULL));
	
	/* what is the length of x */
	n=xarray->dimensions[0];
	dims[0]=N;
	
	yarray =(PyArrayObject *) PyArray_FromDims(1,dims,NPY_INT);
	
		
	for (j=0;j<N;j++){
		i=0;

			x=gsl_rng_uniform(r);
			s=*(double *)(xarray->data);

		while ((x>=s) && (i<=n)) {
			i++;
			s+=*(double *)(xarray->data + i*xarray->strides[0]);
		}
			*((int *)(yarray->data + j*yarray->strides[0] )) = i;
	
	}
	/*
	for (i=0;i<n;i++){
			if (x
			*((double *)(yarray->data + i*yarray->strides[0] )) = gsl_rng_uniform(r); 
			}
	*/

	/*return Py_BuildValue("i", i);*/
	return PyArray_Return(yarray);
	
	/* free up the object */
	gsl_rng_free(r);
}

static PyObject *indsum(PyObject *self, PyObject *args){

	PyArrayObject *parray, *iarray, *sarray; 
	int i,n,N,dims[2];
	double xi;
	int ii,j,k;

	if (!PyArg_ParseTuple(args, "iO!O!:indsum", 
		&N,
		&PyArray_Type, &iarray,
		&PyArray_Type, &parray)
		) 
		{return NULL;}
		
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

		
		/*N=max(iarray)+1;*/

		dims[0]=N;
		dims[1]=k;

		sarray =(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);

		for (i=0;i<N;i++){
			for (j=0;j<k;j++) {
			*((double *)(sarray->data + i*sarray->strides[0] + j*sarray->strides[1]))=0.0;
			}
		}

		n=parray->dimensions[0];

		for (i=0;i<n;i++) {
		
			ii=*((int *)(iarray->data + i*iarray->strides[0]));
				for (j=0;j<k;j++){
					xi=*((double *)(parray->data + i*parray->strides[0] + j*parray->strides[1] ));
					*((double *)(sarray->data + ii*sarray->strides[0]  + j*sarray->strides[1] ))+=xi;
				}
			}
		
			return PyArray_Return(sarray);
	
	}

static PyObject *bigram(PyObject *self, PyObject *args){

PyArrayObject *xarray, *yarray, *sarray;
int N,M,n,dims[2];
int xi,i,yi,j;

if (!PyArg_ParseTuple(args, "iiO!O!:bigram", 
		&N,
		&M,
		&PyArray_Type, &xarray,
		&PyArray_Type, &yarray)
		) 
		{return NULL;}
		

	if (xarray->nd != 1 ) {
		PyErr_SetString(PyExc_ValueError,"x should be 1d array ");
		return NULL;
		}
	
	if (yarray->nd != 1) {
		PyErr_SetString(PyExc_ValueError,"y should be 1d array");
		return NULL;
		}
	if (xarray->dimensions[0] != yarray->dimensions[0]) {
		PyErr_SetString(PyExc_ValueError,"xarray and yarray need to be of same length");
		return NULL;
		}
				
	if (xarray->descr->type_num != NPY_LONG && xarray->descr->type_num != NPY_INT  ) {
	PyErr_Format(PyExc_ValueError,"x array should be integers not type %d", xarray->descr->type_num);
		return NULL;
		}	
	if (yarray->descr->type_num != NPY_LONG && yarray->descr->type_num != NPY_INT  ) {
	PyErr_Format(PyExc_ValueError,"y array should be a integers, not type %d", yarray->descr->type_num);
		return NULL;
		}	

	/*N=max(xarray)+1;
	M=max(yarray)+1;*/

	dims[0]=N;
	dims[1]=M;

	
	sarray =(PyArrayObject *) PyArray_FromDims(2,dims,NPY_LONG);

	/* Initialize as Zero Array */
	for (i=0;i<N;i++){
		for (j=0;j<M;j++) {
		*((int *)(sarray->data + i*sarray->strides[0] + j*sarray->strides[1]))=0;
		}
	}
	

	n=xarray->dimensions[0];

	for (i=0;i<n;i++) {
		xi=*((int *)(xarray->data + i*xarray->strides[0]));
		yi=*((int *)(yarray->data + i*yarray->strides[0]));
		*((int *)(sarray->data + xi*sarray->strides[0]  + yi*sarray->strides[1] ))+=1;
		}

	return PyArray_Return(sarray);

}

static PyObject *unigram(PyObject *self, PyObject *args){

PyArrayObject *xarray, *sarray;
int N,M,n,dims[1];
int xi,yi,i,j;
int x;

if (!PyArg_ParseTuple(args, "iO!", &M,&PyArray_Type, &xarray)) 
		{return NULL;}
		

	if (xarray->nd != 1) {
		PyErr_SetString(PyExc_ValueError,"array should be 1d");
		return NULL;
		}
		
	if (xarray->descr->type_num != NPY_LONG && xarray->descr->type_num != NPY_INT  ) {
	PyErr_Format(PyExc_ValueError,"a array should be a long-integer, not type %d", xarray->descr->type_num);
		return NULL;
		}

	n=xarray->dimensions[0];
/*

	M = *(((int *)xarray->data));

	for (i = 1; i < n; i++){
		x= *((int *)(xarray->data + i*xarray->strides[0] ));
			if (x > M){M=x;}
		}
		M++;
*/
	

	/*printf("This is the max of the input array: %d\n",M);*/

	/* get max 
	dims[0]=(int)N;

	*/
	/*M=max(xarray)+1;*/
	dims[0]=M;

	sarray =(PyArrayObject *) PyArray_FromDims(1,dims,NPY_LONG);
	

	for (i=0;i<M;i++){
		*((int *)(sarray->data + i*sarray->strides[0] ))=0;
		}
		
	for (i=0;i<n;i++) {
		xi=*((int *)(xarray->data + i*xarray->strides[0]));
		*((int *)(sarray->data + xi*sarray->strides[0]  ))+=1;
		}
/*	
*/


return PyArray_Return(sarray);
/*return Py_BuildValue("i", N) ;*/
}

static PyMethodDef xmod_methods[] = {
  	{"indsum",indsum, METH_VARARGS}, 
	{"bigram",bigram, METH_VARARGS}, 
	{"unigram",unigram, METH_VARARGS}, 
	{"xposterior",xposterior,METH_VARARGS},
	{"xptest",xptest,METH_VARARGS},
	{"discretep",discretep,METH_VARARGS},
	{"psi",psi,METH_VARARGS},
	{"xfactorialposterior",xfactorialposterior,METH_VARARGS},
	{"tripsi",tripsi,METH_VARARGS},
	{NULL, NULL}     /* required ending of the method table */
};


PyMODINIT_FUNC initxmod()
{
  	Py_InitModule("xmod", xmod_methods);
	import_array();   /* required NumPy initialization */
}



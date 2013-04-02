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
#include <pthread.h>
#include "fastapprox.h"

#define MAGIC_GAMMA_CONSTANT FLT_MAX
#define GAMMA_THRESH (1e-9)

#define QUEUE_INCREMENT 100

// thread trackers
pthread_t* thread_ids;
// and mutex locks, one per topic (K)
pthread_mutex_t* locks;
// and the queue mutex

pthread_mutex_t queue_lock;
int queue;

// we'll need to pass thread parameters around
typedef struct {
  int tid;
  int Nj;
  double* Z_array;
  PyArrayObject *logphi;
  PyArrayObject *logpsi;
  PyArrayObject *logpi;
  PyArrayObject *data;
  PyArrayObject *Rphi;
  PyArrayObject *Rpsi;
  PyArrayObject *S;
} thread_params;

// need these for parallelization
int NUM_CORES;
int NUM_TOPICS;

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
  if (isinf(p) && signbit(p)) {
    p = -1e33;
  } else if (isinf(p)) {
    p = 1e33;
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
    if (a < FLT_MIN) {
      a = FLT_MIN;
    }
    old_a = a;
    am = a * m;
    d1 = J * fastdigamma(a) - m * fastdigamma(am) + m * sum_logdatamean;
    d2 = J * fasttrigamma(a) - (m * m * fasttrigamma(am));

    inva =((1.0 / a) + (d1 / d2) * a * a);
    a = 1.0/inva;
    if (a < FLT_MIN || isnan(a)) {
      a = old_a;
    }
    delta = abs(old_a - a);
  }

  return PyFloat_FromDouble(a);
}


static inline double lnsumexp(double xarray[], int n) {
  int i;
  double x,m,y;
  // we're given an array of log probabilities, and we
  // want to return the log of the sum of those probs.
  // finding the max and subtracting it, and adding it
  // back in later is a really clever way of doing this
  // while avoiding underflows.
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

inline void* index_pyarray(PyArrayObject *array, int i, int j) {
  return (void*)(array->data + i*array->strides[0] + j*array->strides[1]);
}

inline static int find_index(double array[], int n, double value) {
  // binary search for a desired value.
  // opposite rules of the price is right: we want closest
  // while going over.
  // parameters are the array, the size of the array,
  // and the value we wish to locate.

  // for simplicity of the algorithm, we're not going to
  // do a full binary search. we're going play price is right
  // rules, and then look one to the right.

  int index = (n - 1) / 2;
  int search_width = n / 4;
  if (search_width == 0) index = 0;

  while (search_width > 0) {
    if (array[index] <= value) {
      index += search_width;
    } else if (array[index] > value) {
      index -= search_width;
    }
    search_width /= 2;
    if (index > 0 && search_width == 0 && array[index] > value) {
      search_width = 1;
    }
  }
  while (index < n-1 && value >= array[index]) {
    index += 1;
  }
  return index;
}

void* threaded_posterier_chunk(void* args) {
  thread_params* tp = (thread_params*)args;

  /* seed the random number generator*/
  gsl_rng *random_number_generator;
  random_number_generator = gsl_rng_alloc (gsl_rng_mt19937);
  gsl_rng_set(random_number_generator,time(NULL));

  double f_array[NUM_TOPICS];
  double sz_array[NUM_TOPICS];

  double z, s, rand_x;
  int i, v, f, g, c, k, ci, i_big, i_small;

  // have to manually initialize this array to 0's.
  unsigned long topic_hits[NUM_TOPICS];
  for (k=0; k<NUM_TOPICS; k++) topic_hits[k] = 0;


  i = i_big = i_small = 0;
  // okay, core algorithm
  while (i_big < tp->Nj) {
    // first get the index from the queue
    pthread_mutex_lock(&queue_lock);
    i_big = queue;
    queue += QUEUE_INCREMENT;
    pthread_mutex_unlock(&queue_lock);

    if (i_big >= tp->Nj)
        break;

    for (i_small = 0; i_small < QUEUE_INCREMENT; i_small++) {
      i = i_big + i_small;
      if (i >= tp->Nj)
        break;

      // first fetch the data
      // vocab item
      v=*((int *)index_pyarray(tp->data, 1, i));
      // feature item
      f=*((int *)index_pyarray(tp->data, 2, i));
      // docid
      g=*((int *)index_pyarray(tp->data, 0, i));
      // count
      c=*((int *)index_pyarray(tp->data, 3, i));

      // calculate the log likelihood
      for (k=0; k<NUM_TOPICS; k++) {
        f_array[k] =
            *((double *)index_pyarray(tp->logphi, k, v)) +
            *((double *)index_pyarray(tp->logpsi, k, f)) +
            *((double *)index_pyarray(tp->logpi, g, k));
      }

      z = lnsumexp(f_array, NUM_TOPICS);
      s = 0;

      // what we're building here is the conditional CDF, where
      // sz_array[i] = p(x <= i)
      // so sz_array (sum of z) is monotonically increasing, and
      // at it's max should be close to 1.
      // z is the log sum of the absolute probabilities, so
      // we need to divide (sub in log space) total prob.
      for (k = 0; k<NUM_TOPICS; k++) {
        s += exp(f_array[k] - z);
        sz_array[k] = s;
      }


      // alright, now let's count up all our topics.
      for (ci=0; ci<c; ci++) {
        tp->Z_array[tp->tid] += z;

        rand_x = gsl_rng_uniform(random_number_generator);
        /* sample from exp(f_array[0]-z) */
        k = find_index(sz_array, NUM_TOPICS, rand_x);
        //for (k=0; k < NUM_TOPICS && rand_x >= sz_array[k]; k++);

        topic_hits[k] += 1;
      }

      // finally we need to syncronize these hits with the other threads
      for (k=0; k<NUM_TOPICS; k++) {
        // don't bother syncing if there aren't updates!
        if (topic_hits[k] == 0)
          continue;

        // grab the mutex to make sure we don't have race conditions
        pthread_mutex_lock(&locks[k]);

        *((int *)index_pyarray(tp->Rphi, k, v)) += topic_hits[k];
        *((int *)index_pyarray(tp->Rpsi, k, f)) += topic_hits[k];
        *((int *)index_pyarray(tp->S, g, k)) += topic_hits[k];

        // make sure we reset the counter for next iteration.
        topic_hits[k] = 0;

        // and unlock
        pthread_mutex_unlock(&locks[k]);
      }
    }
  }

  // clean up
  gsl_rng_free(random_number_generator);
}

static PyObject *xfactorialposterior(PyObject *self, PyObject *args) {
  PyArrayObject *logphi,*logpsi,*logpi,*data,*Rphi,*Rpsi,*S;
  int Nj,F,D,J,i,p,err;

  if (!PyArg_ParseTuple(args, "O!O!O!O!iiii",
    &PyArray_Type, &logphi,
    &PyArray_Type, &logpsi,
    &PyArray_Type, &logpi,
    &PyArray_Type, &data,
    &Nj,
    &D,
    &F,
    &J)) {
      return NULL;
  }

  int dims_Rphi[2] = {NUM_TOPICS, D};
  int dims_Rpsi[2] = {NUM_TOPICS, F};
  int dims_S[2] = {J, NUM_TOPICS};

  // we're going to need to sum all of these up in the end.
  double Z_array[NUM_CORES];
  double Z = 0;

  for (p=0; p<NUM_CORES; p++) Z_array[p] = 0;

  Rphi = (PyArrayObject *)PyArray_FromDims(2,dims_Rphi,NPY_INT);
  Rpsi = (PyArrayObject *)PyArray_FromDims(2,dims_Rpsi,NPY_INT);
  S = (PyArrayObject *)PyArray_FromDims(2,dims_S,NPY_INT);

  // reset the queue
  pthread_mutex_lock(&queue_lock);
  queue = 0;
  pthread_mutex_unlock(&queue_lock);

  // k, we've initialized out output arrays
  // let's start the threads.
  thread_params* parameters = (thread_params*)malloc(NUM_CORES * sizeof(thread_params));
  for (p=0; p<NUM_CORES; p++) {
    thread_params* tp = &(parameters[p]);
    tp->tid = p;
    tp->Nj = Nj;
    tp->Z_array = Z_array;
    tp->logphi = logphi;
    tp->logpsi = logpsi;
    tp->logpi = logpi;
    tp->data = data;
    tp->Rphi = Rphi;
    tp->Rpsi = Rpsi;
    tp->S = S;

    err = pthread_create(&(thread_ids[p]), NULL, &threaded_posterier_chunk, (void*)tp);
  }

  // sync the threads.
  for (p=0; p<NUM_CORES; p++) {
    pthread_join(thread_ids[p], NULL);
  }
  free(parameters);

  // okay, all the threads are done. we need to accumulate total probability
  for (p=0; p<NUM_CORES; p++) {
    Z += Z_array[p];
  }

  return Py_BuildValue("(NNNd)", Rphi,Rpsi,S,Z);

}

static PyObject *initialize(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, "ii", &NUM_CORES, &NUM_TOPICS)) {
    return NULL;
  }

  // allocate the locks
  locks = (pthread_mutex_t*) malloc(NUM_TOPICS * sizeof(pthread_mutex_t));
  int k;
  for (k=0; k<NUM_TOPICS; k++) {
    if (pthread_mutex_init(&(locks[k]), NULL) != 0) {
      printf("Lock init failed.\n");
    }
  }

  pthread_mutex_init(&queue_lock, NULL);

  thread_ids = (pthread_t*) malloc(NUM_CORES * sizeof(pthread_t));

  return Py_BuildValue("z", NULL);
}

static PyObject *finalize(PyObject *self, PyObject *args) {
  free(thread_ids);
  int k;
  for (k=0; k<NUM_TOPICS; k++) {
    pthread_mutex_destroy(&(locks[k]));
  }
  pthread_mutex_destroy(&queue_lock);
  free(locks);
  return Py_BuildValue("z", NULL);
}

static PyMethodDef xmod_methods[] = {
  {"a_update", a_update, METH_VARARGS},
  {"xfactorialposterior", xfactorialposterior, METH_VARARGS},
  {"initialize", initialize, METH_VARARGS},
  {"finalize", finalize, METH_NOARGS},
  {NULL, NULL} // required ending of the method table
};


PyMODINIT_FUNC initxmod() {
  PyObject *mod = Py_InitModule("xmod", xmod_methods);
  // initialize the RNG

  // required NumPy initialization */
  import_array();
}



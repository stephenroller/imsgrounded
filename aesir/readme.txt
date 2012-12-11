This code (called aesir) is a python module that implements a type of LDA
model whose latent variables simultaneously define two distributions over
discrete data. This model is described in the following article, where it is
referred to as the "combined model". 

	Andrews, M., Vigliocco, G. & Vinson, D. (2009). Integrating Experiential and
	Distributional Data to Learn Semantic Representations.  Psychological Review,
	Vol. 116(3), pp 463-498.

The module also implements the standard LDA model. 

Installation
============

The python module relies on an external C module, which must be compiled
beforehand. This requires the gnu scientific library. After that, all that is
required is the core python interpreter itself and the python modules numpy and
scipy. 

The following setup is what I use on an Intel-64 machines running ubuntu linux. 

I get the python tools with
	apt-get install python2.5 python-scipy python-numpy python2.5-dev 
and the C compiler with 
	apt-get install gcc
and the gsl libraries with 
	apt-get install gsl-bin libgsl0-dbg libgsl0-dev libgsl0ldbl	

then I do  the following to compile the aesir-utilities.c 

gcc -fPIC -I/usr/include/python2.5 -I/usr/lib/python2.5/site-packages/numpy/core/include  -c aesir-utilities.c -o aesir-utilities.o 
gcc -shared -o xmod.so aesir-utilities.o -lgsl -lgslcblas -lm

Usage
=====

If the above was successful, the module is easiest to use in a python shell such as ipython.
It is imported with 
	import aesir 

You now can use this to load up a training-data file. For a combined data corpus, this data should have a
standard format that is based on the common format used for LDA models. An
example of the latter is as follows:

12:1 45:6 8888:1 12345:2
67:2 998:2 18987:1
23:5 457:18

This represents a corpus of 3 texts. The first text contains words identified as
12, 45, 8888, 12345. The second contains 67, 998 etc. Their frequencies are
indicated by the integer after in the colon. E.g. in text 1, word 8888 occurs
once, word 12345 occurs twice, etc. 

To represent combined data, we use a version of the above format such as 

12:1 45,123:6 8888:1 12345,123:2
67:2 998:2 18987:1
23:5 457,17:18

This says that, e.g. in text 1, word 45 was observed 6 times with feature 123, word 12345 was observed twice with feature 123, etc. 

If you have a data of this nature, you can load it into a python numerical array for use with the module by running 

	data = aesir.dataread("corpus.dat") # assuming the file is called corpus.dat 

Now, we create a model as follows 
	model=aesir.freyr(data,K=10); # where K is the number of latent variables 

You run the gibbs sampler using 
	model.mcmc() ; 

when this ends, you can look at the results by first loading labels for the words and features
	model.getfeaturelabels("feature.labels.dat")
	model.getvocablabels("word.labels.dat")

and then running
	model.printlatentlabels()

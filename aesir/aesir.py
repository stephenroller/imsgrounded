import scipy
import sys
import scipy.special as Sp
import numpy as np
import xmod
import time
import logging
import tempfile
import datetime

log = np.log

class freyr:
    def __init__(self,data,K=100):
        self.data=data
        self.V=self.data[1].max()+1
        """We augment the feature indices in data[2] by one, reserving 0 for the absence of a feature"""
        self.F=self.data[2].max()+1-1
        self.J=self.data[0].max()+1
        self.nj=doccounts(data[0])
        self.Nj=int(self.nj.sum())
        self.K=K

        self.theta=np.ones(self.K)/self.K
        self.beta=np.ones(self.V)/self.V
        self.gamma=np.ones(self.F)/self.F

        self.phi=clip(dirichletrnd(self.beta,self.K))
        self.psi=clip(dirichletrnd(self.gamma,self.K))
        self.pi=clip(dirichletrnd(self.theta,self.J))

        self.phiprior = dirichlet()
        self.phiprior.m = 1.0/self.V
        self.psiprior = dirichlet()
        self.psiprior.m = 1.0/self.F
        self.piprior = dirichlet()
        self.piprior.m=1.0/self.K

        self.mcmc_iteration_max=1e3
        self.verbose=1

    def mcmc(self, cores=8):
        # need to set up for parallelization
        xmod.initialize(cores, self.K)

        for iteration in xrange(int(self.mcmc_iteration_max)):
            last_time = datetime.datetime.now()
            self.fast_posterior()
            self.gamma_a_mle()
            self.theta_a_mle()
            self.beta_a_mle()

            if self.verbose:
                timediff = datetime.datetime.now() - last_time
                logging.warning("LL[%4d] = %f, took %s" % (iteration, self.pseudologlikelihood, timediff))

        # have to clean up memory
        xmod.finalize()

    def fast_posterior(self):
        logvpsi=np.hstack(( np.zeros((self.K,1)),log(self.psi)))
        logphi = log(self.phi)
        logpi = log(self.pi)
        self.Rphi,self.Rpsi,self.S,Z=xmod.xfactorialposterior(logphi,logvpsi,logpi,self.data,self.Nj,self.V,self.F+1,self.J)
        phi=clip(dirichletrnd_array(self.Rphi+self.beta))
        psi=clip(dirichletrnd_array(self.Rpsi[:,1:]+self.gamma))
        vpi=clip(dirichletrnd_array(self.S+self.theta))

        self.phi=row_norm(phi)
        self.psi=row_norm(psi)
        self.pi=row_norm(vpi)

        self.pseudologlikelihood=Z

    def beta_a_mle(self):
        self.phiprior.observation(self.phi)
        self.phiprior.a=self.beta.sum()
        self.phiprior.a_update()
        self.beta=self.phiprior.a*self.phiprior.m

    def theta_a_mle(self):
        self.piprior.observation(self.pi)
        self.piprior.a=np.sum(self.theta)
        self.piprior.a_update()
        self.theta=self.piprior.a*self.piprior.m

    def gamma_a_mle(self):
        self.psiprior.observation(self.psi)
        self.psiprior.a = self.gamma.sum()
        self.psiprior.a_update()
        self.gamma=self.psiprior.a*self.psiprior.m

    def getlatentlabels(self,k=10):
        self.latent_labels=[]
        for j in np.arange(self.K):
            Lphi=[]
            Lpsi=[]
            for i in np.flipud(np.argsort(self.phi[j])[-k:]):
                Lphi.append((self.vocab_labels[i],self.phi[j,i])),
            for i in np.flipud(np.argsort(self.psi[j])[-k:]):
                continue
                Lpsi.append((self.feature_labels[i],self.psi[j,i])),
            self.latent_labels.append((Lphi,Lpsi))

    def printlatentlabels(self,k=10):
        self.getlatentlabels(k)
        k=0
        for l in self.latent_labels:
            k+=1
            print str(k)+': ',
            for i in l[0]:
                print '%s(%2.4f)' % (i[0],i[1]),
            print "\n",
            for j in l[1]:
                print '%s(%2.4f)' % (j[0],j[1]),
            print "\n\n",

    def getfeaturelabels(self,file):
        self.feature_labels=open(file).read().split()

    def getvocablabels(self,file):
        self.vocab_labels=open(file).read().split()


class dirichlet:
    def __init__(self,K=10):
        self.K=K
        self.iteration_eps=1e-4
        self.iteration_max=5

    def observation(self,data):
        self.data=clip(data)
        self.J=data.shape[0]
        self.K=data.shape[1]
        self.logdatamean=np.log(self.data).mean(axis=0)

    def initialize(self):
        self.a, self.m = moment_match(self.data)

    def a_update(self):
        self.a = xmod.a_update(self.a, self.m, np.sum(self.logdatamean), self.J, self.iteration_max, self.iteration_eps)


def doccounts(docidcol):
    counts = []
    lastdoc = -1
    count = 0
    for docid in docidcol:
        if docid != lastdoc and lastdoc != -1:
            counts.append(count)
            count = 0
        count += 1
        lastdoc = docid
    counts.append(count)
    return np.array(counts, float)

# some random number generators
def dirichletrnd(a,J):
    g=np.random.gamma(a,size=(J,np.shape(a)[0]))
    return row_norm(g)

def dirichletrnd_array(a):
    g = np.random.gamma(a + 1e-9) + 1e-10
    return row_norm(g)

def row_norm(a):
    row_sums = a.sum(axis=1)
    a /= row_sums[:, np.newaxis]
    return a

# IO Stuff
def itersplit(s, sub):
    pos = 0
    while True:
        i = s.find(sub, pos)
        if i == -1:
            yield s[pos:]
            break
        else:
            yield s[pos:i]
            pos = i + len(sub)

def int2bytes(i8):
    s = ""
    for bi in xrange(8):
        b = i8 & 0xFF
        s += chr(b)
        i8 >>= 8
    return s

def parse_item(item):
    item = item.strip()
    if "," in item:
        left, tworight = item.split(",")
        retval = [left] + tworight.split(":")
    else:
        retval = item.split(":")
    return map(int, retval)

def clip(arr):
  return np.clip(arr, 1e-10, 1 - 1e-10)

def dataread(file):
    return np.load(file).T

    tmpfile = open("binary.dat", "wb")

    #tmpfile = tempfile.NamedTemporaryFile(delete=False)
    #print tmpfile.name

    tmpfile.write("\x93\x4e\x55\x4d\x50\x59\x01\x00\x46\x00")

    data_file=open(file)
    row_count = 0
    logging.warning("Starting to read data (pass 1)...")
    for doc_id, doc in enumerate(data_file):
        for item in itersplit(doc, " "):
            row_count += 1
    data_file.close()

    # okay let's go through this again
    # start writing the header.
    header = "{'descr': '<i8', 'fortran_order': False, 'shape': (%d, %d), }" % (row_count, 4)
    header = ("%-69s\x0a" % header)
    tmpfile.write(header)

    # so far recognizes two data-types, lda and combinatorial lda
    data_file=open(file)
    group_j=0

    dimensions = 1
    logging.warning("Starting to read data (pass 2)...")
    for doc_id, doc in enumerate(data_file):
        for item in itersplit(doc, " "):
            splitted = parse_item(item)
            if len(splitted) == 2:
                feat_id = 0
                word_id, count = splitted
            elif len(splitted) == 3:
                word_id, feat_id, count = splitted

            a = [doc_id, word_id, feat_id, count]
            tmpfile.write("".join(map(int2bytes, a)))

    tmpfile.close()
    data = np.load(tmpfile.name)
    del tmpfile

    return data.T


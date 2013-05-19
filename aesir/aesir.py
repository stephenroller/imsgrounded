import scipy
import sys
import scipy.special as Sp
import numpy as np
import xmod
import time
import logging
import tempfile
import datetime
import struct
import os.path
import zipfile

log = np.log
now = datetime.datetime.now
ONE_HOUR = datetime.timedelta(hours=1)

def safe_pi_read(filename):
    zipf = zipfile.ZipFile(filename)
    buf_s = zipf.read("pi.npy")
    header = buf_s[:80]
    dims = map(int, header[header.index("(")+1:header.rindex(")")].split(", "))
    pi = np.fromstring(buf_s[80:]).reshape(dims)
    del buf_s, header, dims
    zipf.close()
    return pi


class freyr:
    def __init__(self, data, K=100, model_out=None):
        self.data=data
        self.J=self.data[0].max() + 1
        self.V=self.data[1].max() + 1
        self.F=self.data[2].max() + 1
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

        self.model_out = model_out

        self.burnin = 100
        self.mcmc_iterations_max = 1000
        self.max_iteration = 0
        self.loglikelihoods = []
        self.timediffs = []
        self.pseudologlikelihood = 0

    def mcmc(self, cores=8):
        # need to set up for parallelization
        logging.info("Initializing xmod...")
        xmod.initialize(cores, self.K)
        logging.info("xmod initialized.")

        try:
            last_save_time = datetime.datetime.now()

            for iteration in xrange(self.max_iteration + 1, int(self.mcmc_iterations_max) + 1):
                last_time = now()
                self.fast_posterior()
                self.gamma_a_mle()
                self.theta_a_mle()
                self.beta_a_mle()

                timediff = datetime.datetime.now() - last_time
                self.loglikelihoods.append(self.pseudologlikelihood)
                self.timediffs.append(timediff.total_seconds())
                self.max_iteration = iteration
                logging.debug("LL(%4d) = %f, took %s" % (iteration, self.pseudologlikelihood, timediff))

                if now() - last_save_time >= ONE_HOUR and self.model_out:
                    self.save_model(self.model_out)
                    logging.debug("%s passed. Saved progress to %s." % (now() - last_save_time, self.model_out))
                    last_save_time = now()


        except KeyboardInterrupt:
            logging.info("Terminated early. Cleaning up.")

        # have to clean up memory
        logging.debug("Finalizing xmod...")
        xmod.finalize()
        logging.debug("xmod finalized.")

    def fast_posterior(self):
        logvpsi=np.hstack(( np.zeros((self.K,1)),log(self.psi)))
        logphi = log(self.phi)
        logpi = log(self.pi)

        if abs(self.pseudologlikelihood) > 1e20 or np.any(np.isnan(logphi)) or np.any(np.isnan(logvpsi)) or np.any(np.isnan(logpi)) or \
                np.any(logphi > 0) or np.any(logvpsi > 0) or np.any(logpi > 0):
            logging.error("Numerical instability detected. Dropping to debugger. (before loop)")
            import pdb
            pdb.set_trace()

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
        if np.isnan(self.phiprior.a) or np.isnan(self.phiprior.m):
            logging.error("I got some numerical instability. Let's go.")
            import pdb
            pdb.set_trace()

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

    def save_model(self, filename):
        if np.isnan(self.pseudologlikelihood):
            logging.error("Numerical instability detected. Cowardly refusing to save and dying hard...")
            sys.exit(2)

        np.savez_compressed(
                filename + ".tmp",
                psi=self.psi,
                phi=self.phi,
                pi=self.pi,
                k=self.K,
                max_iteration=self.max_iteration,
                loglikelihoods=self.loglikelihoods,
                timediffs=self.timediffs)

        os.rename(filename + ".tmp", filename)

    def load_model(self, filename):
        model = np.load(filename)
        self.psi = model['psi']
        self.phi = model['phi']
        self.K = model['k']
        self.pi = safe_pi_read(filename) # hack b/c np doesn't buffer things properly
        self.max_iteration = model['max_iteration']
        self.loglikelihoods = list(model['loglikelihoods'])
        self.timediffs = list(model['timediffs'])

    def getfeaturelabels(self,file):
        self.feature_labels=open(file).read().split()

    def getvocablabels(self,file):
        self.vocab_labels=open(file).read().split()


class dirichlet:
    def __init__(self,K=10):
        self.K=K
        self.iteration_eps=1e-4
        self.iteration_max=5
        self.a = 0

    def observation(self,data):
        self.data=clip(data)
        self.J=data.shape[0]
        self.K=data.shape[1]
        self.logdatamean=np.log(self.data).mean(axis=0)

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
    try:
        if os.path.getmtime(file + ".npy") < os.path.getmtime(file):
            logging.info("The corpus file is newer than the binary file. Recreating it...")
        else:
            return np.load(file + ".npy").T
    except (IOError, OSError):
        logging.info("Binary file has not been created; creating it.")
        pass

    tmpfile = open(file + ".npy", "wb")

    tmpfile.write("\x93\x4e\x55\x4d\x50\x59\x01\x00\x46\x00")

    data_file=open(file)
    row_count = 0
    logging.warning("Starting to read data (pass 1)...")
    for doc_id, doc in enumerate(data_file):
        for item in itersplit(doc, " "):
            if item.strip():
                row_count += 1
    data_file.close()
    num_docs = doc_id + 1

    # okay let's go through this again
    # start writing the header.
    header = "{'descr': '<i8', 'fortran_order': False, 'shape': (%d, %d), }" % (row_count, 4)
    header = ("%-69s\x0a" % header)
    tmpfile.write(header)

    # so far recognizes two data-types, lda and combinatorial lda
    data_file=open(file)

    logging.warning("Starting to read data (pass 2)...")
    rows_written = 0
    for doc_id, doc in enumerate(data_file):
        if (doc_id + 1) % 1000 == 0:
            logging.info("Processing doc %d/%d (%4.2f%%)" % (doc_id + 1, num_docs, 100 * float(rows_written) / row_count))
        for item in itersplit(doc, " "):
            if not item.strip():
                continue
            splitted = parse_item(item)
            if len(splitted) == 2:
                feat_id = 0
                word_id, count = splitted
            elif len(splitted) == 3:
                word_id, feat_id, count = splitted

            outs = struct.pack("<QQQQ", doc_id, word_id, feat_id, count)
            rows_written += 1
            tmpfile.write(outs)

    tmpfile.close()
    data = np.load(tmpfile.name)
    del tmpfile

    return data.T


#!/usr/bin/env python

# onlineldavb.py: Package of functions for fitting Latent Dirichlet
# Allocation (LDA) with online variational Bayes (VB).
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Extended and modified 2013 by Stephen Roller <roller@cs.utexas.edu>

import datetime
import sys, re, time, string
import numpy as n
import logging
import argparse
import os
from collections import Counter
from xmod import vdigamma as psi, vlngamma as gammaln
from random import sample, seed
from aesir import itersplit, row_norm, ONE_HOUR

logging.basicConfig(
    format="[ %(levelname)-10s %(module)-8s %(asctime)s  %(relativeCreated)-10d ]  %(message)s",
    datefmt="%H:%M:%S:%m",
    level=logging.DEBUG)

MEAN_CHANGE_THRESH = 0.001
DEBUG = False

def parse_doc_list(docs):
    return docs
    wordids = [[wid for wid, cnt in d] for d in docs]
    wordcts = [n.array([cnt for wid, cnt in d]) for d in docs]
    return((wordids, wordcts))

def dirichlet_expectation_2(alpha):
    """
    Computes multiple dirichlet expectations simultaneously.
    """
    return (psi(alpha.T) - psi(n.sum(alpha, 1))).T

def dirichlet_expectation_1(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    return (psi(alpha) - psi(n.sum(alpha)))


def gamma_estimation_step(alpha, expElogthetad, cts, phinorm, expElogbetad):
    gammad = alpha + expElogthetad * n.dot(n.divide(cts, phinorm), expElogbetad.T)
    Elogthetad = dirichlet_expectation_1(gammad)
    expElogthetad = n.exp(Elogthetad)
    phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
    return (gammad, Elogthetad, expElogthetad, phinorm)


class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, vocab, K, D, alpha, eta, tau0, kappa):
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. When analyzing documents, any word
           not in this set will be ignored.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """
        #self._vocab = dict()
        #for word in vocab:
        #    word = word.lower()
        #    word = re.sub(r'[^a-z]', '', word)
        #    self._vocab[word] = len(self._vocab)
        self._vocab = vocab
        self._K = K
        self._W = max(vocab) + 1
        self._D = D
        self._alpha = alpha
        self._eta = eta
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0
        self._rhot = 1.0

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = 1*n.random.gamma(100., 1./100., (self._K, self._W))
        self._Elogbeta = dirichlet_expectation_2(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

        # bookkeeping stuff
        self.timediffs = []
        self.perwordbounds = []
        self.max_iteration = 0
        self.times_doc_seen = n.zeros(D)

    def do_e_step(self, docs):
        """
        Given a mini-batch of documents, estimates the parameters
        gamma controlling the variational distribution over the topic
        weights for each document in the mini-batch.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns a tuple containing the estimated values of gamma,
        as well as sufficient statistics needed to update lambda.
        """
        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        (wordids, wordcts) = parse_doc_list(docs)
        batchD = len(wordids)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1 * n.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation_2(gamma)
        expElogtheta = n.exp(Elogtheta)

        sstats = n.zeros(self._lambda.shape).T
        # Now, for each document d update that document's gamma and phi
        for d in xrange(batchD):
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d]
            Elogthetad = Elogtheta[d]
            expElogthetad = expElogtheta[d]
            expElogbetad = n.take(self._expElogbeta, ids, axis=1)
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad, Elogthetad, expElogthetad, phinorm = gamma_estimation_step(self._alpha, expElogthetad, cts, phinorm, expElogbetad)
                # If gamma hasn't changed much, we're done.
                meanchange = n.sum(n.abs(gammad - lastgamma))
                if (meanchange < self._K * MEAN_CHANGE_THRESH):
                    break
            gamma[d] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[ids,:] += n.outer(cts/phinorm, expElogthetad)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats.T * self._expElogbeta

        return((gamma, sstats))

    def update_lambda(self, docs):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.

        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma, sstats) = self.do_e_step(docs)
        # Estimate held-out likelihood for current values of lambda.
        bound = self.approx_bound(docs, gamma)
        # Update lambda based on documents.
        self._lambda = self._lambda * (1-rhot) + \
            rhot * (self._eta + self._D * sstats / len(docs[0]))
        self._Elogbeta = dirichlet_expectation_2(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._updatect += 1

        return(gamma, bound)

    def approx_bound(self, docs, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """

        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if (type(docs).__name__ == 'string'):
            temp = list()
            temp.append(docs)
            docs = temp

        (wordids, wordcts) = parse_doc_list(docs)
        batchD = len(wordids)

        score = 0
        Elogtheta = dirichlet_expectation_2(gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in xrange(batchD):
            gammad = gamma[d]
            ids = wordids[d]
            cts = n.array(wordcts[d])
            right = n.take(self._Elogbeta, ids, axis=1).T
            temp = Elogtheta[d] + right[:,:]
            tmax_v = n.max(temp, axis=1)
            phinorm = n.log(n.sum(n.exp(temp[:,:].T - tmax_v), axis=0)) + tmax_v
            score += n.dot(cts, phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma)*Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += n.sum(gammaln(self._alpha*self._K) - gammaln(n.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / batchD

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta-self._lambda)*self._Elogbeta)
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + n.sum(gammaln(self._eta*self._W) -
                              gammaln(n.sum(self._lambda, 1)))

        return(score)

    def save_model(self, filename):
        n.savez_compressed(filename + ".tmp.npz",
                #phi=row_norm(self._lambda),
                phi=self._lambda,
                psi=n.ones((self._K, 1))/self._K,
                max_iteration=self._updatect,
                k=self._K,
                timediffs=self.timediffs,
                perwordbounds=self.perwordbounds,
                eta = self._eta,
                tau0 = self._tau0,
                alpha = self._alpha,
                expElogbeta = self._expElogbeta,
                times_doc_seen = self.times_doc_seen,
                )
        os.rename(filename + ".tmp.npz", filename)

    def load_model(self, filename):
        m = n.load(filename)
        self._lambda = m['phi']
        self._K = m['k']
        self._updatect = m['max_iteration']
        self._eta = m['eta']
        self._tau0 = m['tau0']
        self._alpha = m['alpha']
        self.timediffs = list(m['timediffs'])
        self.perwordbounds = list(m['perwordbounds'])
        self._expElogbeta = m['expElogbeta']
        self.times_doc_seen = m['times_doc_seen']

        # reinitialize the variational distribution q(beta|lambda)
        self._Elogbeta = dirichlet_expectation_2(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

    def inference(self, corpus, batchsize, max_iterations, model_file):
        D = self._D
        batchsize = min(batchsize, D)
        # keep track of docs seen
        bigtic = datetime.datetime.now()
        try:
            save_tic = datetime.datetime.now()
            for iteration in xrange(self._updatect + 1, max_iterations + 1):
                tic = datetime.datetime.now()
                docset_ids = sample(xrange(D), batchsize)
                docset = (corpus[0][docset_ids], corpus[1][docset_ids])
                (gamma, bound) = self.update_lambda(docset)
                (wordids, wordcts) = parse_doc_list(docset)
                perwordbound = bound * len(docset_ids) / (D * sum(n.sum(doc) for doc in wordcts))
                toc = datetime.datetime.now()
                self.times_doc_seen[docset_ids] += 1
                logging.info('(%4d) %4d [%15s/%15s]:  rho_t = %1.5f,  perp est = (%8f) [seen = %d/%d]' %
                    (self._K, iteration, toc - tic, toc - bigtic, self._rhot, n.exp(-perwordbound), n.sum(self.times_doc_seen > 0), D))
                self.perwordbounds.append(n.exp(-perwordbound))
                self.timediffs.append((toc - tic).total_seconds())
                if toc - save_tic >= ONE_HOUR:
                    save_tic = toc
                    logging.info("Processed for one hour. Saving model to %s..." % model_file)
                    self.save_model(model_file)
        except KeyboardInterrupt:
            logging.info("Terminated early...")
            pass

        bigtoc = datetime.datetime.now()
        logging.info("Total learning runtime: %s" % (bigtoc - bigtic))
        logging.info("Done with training. Saving model...")
        self.save_model(model_file)





# Reads in the corpus
def read_andrews(filename):
    tic = datetime.datetime.now()
    logging.info("reading corpus...")
    wordids = []
    wordcts = []
    with open(filename) as f:
        for d_s in f.readlines():
            dids = []
            dcts = []
            for w in d_s.strip().split():
                colon = w.rindex(":")
                i, c = (int(w[:colon]), int(w[colon+1:]))
                dids.append(i)
                dcts.append(c)

            wordids.append(n.array(dids))
            wordcts.append(n.array(dcts))
    wordids = n.array(wordids)
    wordcts = n.array(wordcts)
    toc = datetime.datetime.now()
    logging.info("finished reading corpus. (took %s)" % (toc - tic))
    return (wordids, wordcts)

def main():

    parser = argparse.ArgumentParser(
                description='Variational inference for Andrews model.')
    parser.add_argument('--input', '-i', metavar='FILE',
                        help='Load a precreated Numpy Array as the data matrix.')
    parser.add_argument('--output', '-o', metavar='FILE', help='Save the model.')
    parser.add_argument('--topics', '-k', metavar='INT', default=100, type=int,
                        help='The number of topics to load.')
    parser.add_argument('--iterations', '-I', metavar='INT', default=1000, type=int,
                        help='Number of iterations.')
    parser.add_argument('--tau0', '-T', metavar='INT', default=32, type=int,
                        help='Sets the tau0 hyperparameter.')
    parser.add_argument('--kappa', '-K', metavar='FLOAT', type=float, default=0.7,
                        help='Sets the kappa hyperparameter.')
    parser.add_argument('--batchsize', '-S', metavar='INT', type=int, default=1024,
                        help='The size of the mini-batches.')
    parser.add_argument('--continue', '-c', action='store_true', dest='kontinue',
                        help='Continue computing from an existing model.')
    parser.add_argument('--eta', metavar='FLOAT', type=float,
                        help='Hyperparamater eta. (Default 1/k)')
    parser.add_argument('--alpha', metavar='FLOAT', type=float,
                        help='Hyperparameter alpha. (Default 1/k)')
    parser.add_argument('--randomseed', metavar='INT', type=int,
                        help='Supply the seed for the random number generator.')
    args = parser.parse_args()

    if args.randomseed:
        seed(args.randomseed)
        n.random.seed(args.randomseed)

    logging.info("Online Variational Bayes inference.")
    logging.info("Calling read_andrews")
    corpus = read_andrews(args.input)
    logging.info("Finished reading corpus.")

    D = len(corpus[0])
    vocab = range(max(wid for doc in corpus[0] for wid in doc)+1)

    k = args.topics
    batchsize = args.batchsize
    tau0 = args.tau0
    kappa = args.kappa
    numiterations = args.iterations
    eta = args.eta and args.eta or 1./k
    alpha = args.alpha and args.alpha or 1./k

    logging.info("Initializing OnlineLDA object.")
    olda = OnlineLDA(vocab, k, D, alpha, eta, tau0, kappa)
    if args.kontinue:
        logging.info("Attemping to resume from %s." % args.output)
        try:
            olda.load_model(args.output)
        except IOError:
            logging.warning("IOError when loading old model. Starting from the beginning.")

    logging.info("Starting inference.")
    olda.inference(corpus, batchsize, numiterations, args.output)
    logging.info("Finished with inference.")

if __name__ == '__main__':
    main()


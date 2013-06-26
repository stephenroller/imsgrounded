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
#from xmod import vdigamma as psi, vlngamma as gammaln
from scipy.special import gammaln, psi
from random import sample, seed
from aesir import itersplit, row_norm, ONE_HOUR, QUARTER_HOUR

SAVE_FREQUENCY = ONE_HOUR

logging.basicConfig(
    format="[ %(levelname)-10s %(module)-8s %(asctime)s  %(relativeCreated)-10d ]  %(message)s",
    datefmt="%H:%M:%S:%m",
    level=logging.DEBUG)

MEAN_CHANGE_THRESH = 0.001
DEBUG = False

def parse_doc_list(docs):
    return docs

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


def remove_redundancies(ids, update_matrix):
    # in our E step, we'll need to update the feature and word ids with
    # the updates. However, we can't just simply use fancy indexing,
    # because redundancies in the matrix cause the second update of
    # the same ID to get ignored. Our goal here is to reduce the size
    # of this matrix so we can use the fancy indexing.
    uniq_ids = list(set(ids))
    uniq_ids_d = { id_: new_i for new_i, id_ in enumerate(uniq_ids) }
    update_matrix_uniq = n.zeros(len(uniq_ids))
    for old_i, id_ in enumerate(ids):
        col = update_matrix[old_i]
        new_id = uniq_ids_d[id_]
        update_matrix_uniq[new_id] += col
    return uniq_ids, update_matrix_uniq


class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, input_filename, vocab, feats, K, D, alpha, eta, mu, tau0, kappa):
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
        self.input_filename = input_filename

        self._vocab = vocab
        self._feats = feats
        self._W = max(vocab) + 1
        self._F = max(feats) + 1

        self._K = K
        self._D = D

        self._alpha = alpha
        self._eta = eta
        self._mu = mu
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0
        self._rhot = 1.0

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = 1*n.random.gamma(100., 1./100., (self._K, self._W))
        self._Elogbeta = dirichlet_expectation_2(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

        # ... and var distr q(pi|omega)
        self._omega = 1 * n.random.gamma(100., 1./100., (self._K, self._F))
        self._Elogpi = dirichlet_expectation_2(self._omega)
        self._expElogpi = n.exp(self._Elogpi)

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
        (wordcts, wordids, featids) = parse_doc_list(docs)
        batchD = len(wordids)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1 * n.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation_2(gamma)
        expElogtheta = n.exp(Elogtheta)

        # book keeping for updating lambda later.
        wstats = n.zeros(self._lambda.shape)
        # and updating pi later
        fstats = n.zeros(self._omega.shape)

        # Now, for each document d update that document's gamma and phi
        for d in xrange(batchD):
            # These are mostly just shorthand (but might help cache locality)
            wids = wordids[d]
            fids = featids[d]
            cts = wordcts[d]

            # k sized vectors, distr of topics over doc
            gammad = gamma[d]
            Elogthetad = Elogtheta[d]
            expElogthetad = expElogtheta[d]

            Elogbetad = n.take(self._Elogbeta, wids, axis=1)
            expElogbetad = n.take(self._expElogbeta, wids, axis=1)
            Elogpid = n.take(self._Elogpi, fids, axis=1)
            expElogpid = n.take(self._expElogpi, fids, axis=1)

            # The optimal phi_{dwk} is proportional to
            #    expElogthetad_k * expElogbetad_w * expElogpid_f = exp { Elogthetad_k + Elogbetad_w  + Elogpid_f }
            # phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, n.multiply(expElogbetad, expElogpid)) + 1e-100

            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                # keep track of convergence
                lastgamma = gammad

                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad * n.dot(cts / phinorm, expElogbetad.T * expElogpid.T)
                Elogthetad = dirichlet_expectation_1(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, n.multiply(expElogbetad, expElogpid)) + 1e-100

                # If gamma hasn't changed much, we're done.
                meanchange = n.sum(n.abs(gammad - lastgamma))
                if (meanchange < self._K * MEAN_CHANGE_THRESH):
                    break

            # update our global copy of gamma
            gamma[d] = gammad

            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            uniq_wids, w_cts = remove_redundancies(wids, cts)
            uniq_fids, f_cts = remove_redundancies(fids, cts)
            update_stats_wids = n.outer(expElogthetad.T, w_cts / n.dot(expElogthetad, n.take(self._expElogbeta, uniq_wids, axis=1)))
            update_stats_fids = n.outer(expElogthetad.T, f_cts / n.dot(expElogthetad, n.take(self._expElogpi, uniq_fids, axis=1)))

            wstats[:,uniq_wids] += update_stats_wids
            fstats[:,uniq_fids] += update_stats_fids

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # wstats[k, w] = \sum_d n_{dw} * phi_{dwk}
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        wstats = wstats * self._expElogbeta
        fstats = fstats * self._expElogpi

        return (gamma, wstats, fstats)

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
        (gamma, wstats, fstats) = self.do_e_step(docs)
        # Estimate held-out likelihood for current values of lambda.
        bound = self.approx_bound(docs, gamma)

        # Update lambda based on documents.
        self._lambda = self._lambda * (1-rhot) + rhot * (self._eta + self._D * wstats / len(docs[0]))
        # hardcode equal probability for each of the zero words. technically this isn't necessary.
        if self._W > 1:
            self._lambda[:,0] = self._lambda[:,1:].mean()
        else:
            self._lambda[:,0] = self._lambda.mean()
        # update expectations
        self._Elogbeta = dirichlet_expectation_2(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

        # update pi based on documents
        self._omega = self._omega * (1 - rhot) + rhot * (self._mu + self._D * fstats / len(docs[0]))
        # hardcode equal probability for each the features
        if self._F > 1:
            self._omega[:,0] = self._omega[:,1:].mean()
        else:
            self._omega[:,0] = self._omega.mean()
        # update expectations
        self._Elogpi = dirichlet_expectation_2(self._omega)
        self._expElogpi = n.exp(self._Elogpi)

        # mark that we completed this iteration
        self._updatect += 1

        return gamma, bound

    def approx_bound(self, docs, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """
        (wordcts, wordids, featids) = parse_doc_list(docs)
        batchD = len(wordids)

        score = 0
        Elogtheta = dirichlet_expectation_2(gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(docs | theta, beta, pi)]
        for d in xrange(batchD):
            gammad = gamma[d]
            wids = wordids[d]
            fids = featids[d]
            cts = n.array(wordcts[d])
            right = n.take(self._Elogbeta, wids, axis=1).T
            right2 = n.take(self._Elogpi, fids, axis=1).T
            temp = Elogtheta[d] + (right + right2)[:,:]
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
        score += n.sum((self._eta-self._lambda)*self._Elogbeta)
        score += n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score += n.sum(gammaln(self._eta*self._W) -
                       gammaln(n.sum(self._lambda, 1)))

        # E[log p(pi | mu) - log q (pi | omega)]
        score += n.sum((self._mu-self._omega)*self._Elogpi)
        score += n.sum(gammaln(self._omega) - gammaln(self._mu))
        score += n.sum(gammaln(self._mu*self._F) -
                       gammaln(n.sum(self._omega, 1)))
        return score

    def save_model(self, filename):
        n.savez_compressed(filename + ".tmp.npz",
                phi=self._lambda,
                psi=self._omega,
                max_iteration=self._updatect,
                k=self._K,
                timediffs=self.timediffs,
                perwordbounds=self.perwordbounds,
                eta = self._eta,
                mu = self._mu,
                tau0 = self._tau0,
                alpha = self._alpha,
                times_doc_seen = self.times_doc_seen,
                input_filename = self.input_filename,
                )
        modified_filename = filename.endswith(".npz") and filename[:-4] or filename
        versioned_filename = "%s.%d.npz" % (modified_filename, self._updatect)
        os.rename(filename + ".tmp.npz", versioned_filename)
        if os.path.exists(filename):
            os.remove(filename)
        os.symlink(os.path.abspath(versioned_filename), filename)

    def load_model(self, filename):
        m = n.load(filename)
        self._lambda = m['phi']
        self._omega = m['psi']
        self._K = m['k']
        self._updatect = m['max_iteration']
        self._eta = m['eta']
        self._mu = m['mu']
        self._tau0 = m['tau0']
        self._alpha = m['alpha']
        self.timediffs = list(m['timediffs'])
        self.perwordbounds = list(m['perwordbounds'])
        self.times_doc_seen = m['times_doc_seen']
        self.input_filename = str(m['input_filename'])

        # reinitialize the variational distribution q(beta|lambda)
        self._Elogbeta = dirichlet_expectation_2(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        # and q(pi|omega)
        self._Elogpi = dirichlet_expectation_2(self._omega)
        self._expElogpi = n.exp(self._Elogpi)

    def inference(self, corpus, batchsize, max_iterations, model_file):
        D = self._D
        batchsize = min(batchsize, D)
        # keep track of docs seen
        bigtic = datetime.datetime.now()
        shouldsave = True
        try:
            save_tic = datetime.datetime.now()
            for iteration in xrange(self._updatect + 1, max_iterations + 1):
                tic = datetime.datetime.now()
                docset_ids = sample(xrange(D), batchsize)
                docset = (corpus[0][docset_ids], corpus[1][docset_ids], corpus[2][docset_ids])
                (gamma, bound) = self.update_lambda(docset)
                (wordcts, wordids, featids) = parse_doc_list(docset)
                perwordbound = bound * len(docset_ids) / (D * sum(n.sum(doc) for doc in wordcts))
                if n.isnan(perwordbound):
                    logging.error("perwordbound is nan. Cleaning up without saving.")
                    shouldsave = False
                    break
                toc = datetime.datetime.now()
                self.times_doc_seen[docset_ids] += 1
                logging.info('(%4d) %4d [%15s/%15s]:  rho_t = %1.5f,  perwordbound = (%8f) [seen = %d/%d]' %
                    (self._K, iteration, toc - tic, toc - bigtic, self._rhot, perwordbound, n.sum(self.times_doc_seen > 0), D))
                self.perwordbounds.append(perwordbound)
                self.timediffs.append((toc - tic).total_seconds())
                if toc - save_tic >= SAVE_FREQUENCY:
                    logging.info("Processed for %s > %s. Saving model to %s..." % (toc - save_tic, SAVE_FREQUENCY, model_file))
                    save_tic = toc
                    self.save_model(model_file)
        except KeyboardInterrupt:
            logging.info("Terminated early...")
            pass

        bigtoc = datetime.datetime.now()
        logging.info("Total learning runtime: %s" % (bigtoc - bigtic))
        if shouldsave:
            logging.info("Done with training. Saving model...")
            self.save_model(model_file)





# Reads in the corpus
def read_andrews(filename):
    tic = datetime.datetime.now()
    logging.info("reading corpus...")
    wordids = []
    featids = []
    wordcts = []
    with open(filename) as f:
        for d_s in f.readlines():
            dids = []
            fids = []
            dcts = []
            for w in d_s.strip().split():
                colon = w.rindex(":")
                i, c = w[:colon], int(w[colon+1:])
                if "," in i:
                    # TODO: we'll need to change this if we go to higher dimensional
                    i, f = i.split(",")
                    i = int(i)
                    f = int(f)
                else:
                    i = int(i)
                    f = 0

                dids.append(i)
                dcts.append(c)
                fids.append(f)

            wordcts.append(n.array(dcts))
            wordids.append(n.array(dids))
            featids.append(n.array(fids))

    featids = n.array(featids)
    wordids = n.array(wordids)
    wordcts = n.array(wordcts)

    toc = datetime.datetime.now()
    logging.info("Finished reading corpus. (took %s)" % (toc - tic))

    return (wordcts, wordids, featids)

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
                        help='Hyperparamater eta. (Default 1/V)')
    parser.add_argument('--alpha', metavar='FLOAT', type=float,
                        help='Hyperparameter alpha. (Default 1/k)')
    parser.add_argument('--mu', metavar='FLOAT', type=float,
                        help='Hyperparameter mu. (Default 1/F)')
    parser.add_argument('--randomseed', metavar='INT', type=int,
                        help='Supply the seed for the random number generator.')
    args = parser.parse_args()

    if args.randomseed:
        seed(args.randomseed)
        n.random.seed(args.randomseed)

    logging.info("Online Variational Bayes inference.")
    logging.info("Calling read_andrews")
    corpus = read_andrews(args.input)

    D = len(corpus[0])
    vocab = range(max(wid for doc in corpus[1] for wid in doc)+1)
    feats = range(max(fid for doc in corpus[2] for fid in doc)+1)

    k = args.topics
    batchsize = args.batchsize
    tau0 = args.tau0
    kappa = args.kappa
    numiterations = args.iterations
    eta = args.eta and args.eta or 1./len(vocab)
    mu = args.mu and args.mu or 1./len(feats)
    alpha = args.alpha and args.alpha or 1./k

    logging.info("Initializing OnlineLDA object.")
    olda = OnlineLDA(args.input, vocab, feats, k, D, alpha, eta, mu, tau0, kappa)
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


#!/usr/bin/env python

import sys
import logging
import argparse

import aesir

logging.basicConfig(
    format="[ %(levelname)-10s %(module)-8s %(asctime)s  %(relativeCreated)-10d ]  %(message)s",
    datefmt="%H:%M:%S:%m",
    level=logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(
                description='Runs the Andrews model.')
    parser.add_argument('--input', '-i', metavar='FILE',
                        help='Load a precreated Numpy Array as the data matrix.')
    parser.add_argument('--output', '-o', metavar='FILE', help='Save the model.')
    parser.add_argument('--topics', '-k', metavar='INT', default=100, type=int,
                        help='The number of topics to load.')
    parser.add_argument('--burnin', '-b', metavar='INT', type=int, default=0,
                        help='Burnin samples.')
    parser.add_argument('--iterations', '-I', metavar='INT', default=1000, type=int,
                        help='Number of iterations.')
    parser.add_argument('--threads', '-t', metavar='INT', default=4, type=int,
                        help='The number of separate threads to run.')
    args = parser.parse_args()

    logging.info("Loading data...")
    data = aesir.dataread(args.input)
    logging.warning("Initializing model...")
    model = aesir.freyr(data, K=args.topics, model_out=args.output)
    logging.info("Finished initializing.")
    logging.info("Starting MCMC...")
    model.burnin_iterations = args.burnin
    model.mcmc_iterations_max = args.iterations
    model.mcmc(cores=args.threads)
    logging.info("Finished with MCMC!")
    logging.info("Saving model to '%s'..." % args.output)
    model.save_model(args.output)
    logging.info("Model saved.")
    logging.info("All finished.")




if __name__ == '__main__':
    main()


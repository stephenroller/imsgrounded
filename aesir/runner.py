#!/usr/bin/env python

import sys
import aesir
import datetime
import logging
logging.basicConfig(
    format="[ %(levelname)-10s %(module)-8s %(asctime)s  %(relativeCreated)-10d ]  %(message)s",
    datefmt="%H:%M:%S:%m",
    level=logging.DEBUG)

now = datetime.datetime.now

data_f, k, model_out_file = sys.argv[1:]
k = int(k)

logging.info("Loading data...")
data = aesir.dataread(data_f)
logging.warning("Initializing model...")
model = aesir.freyr(data, K=k)
logging.info("Finished initializing.")
logging.info("Starting MCMC...")
model.mcmc()
logging.info("Finished with MCMC!")
logging.info("Saving model..")
#model.save_model(model_out_file)
logging.info("Model saved.")
logging.info("All finished.")



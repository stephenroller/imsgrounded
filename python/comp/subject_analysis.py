#!/usr/bin/env python

import sys
import pandas as pd
from os.path import basename
from scipy.stats import spearmanr

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

filenames = sys.argv[1:]
dataframes = [pd.read_csv(x) for x in sys.argv[1:]]
if len(filenames) > 1:
    filenames += ["CONCAT"]
    dataframes += [pd.concat(dataframes)]


# go ahead and open the big pdf
pp = PdfPages("subjects.pdf")
pyplot.locator_params(tight=True)

for fileno, (filename, data) in enumerate(zip(filenames, dataframes)):
    judgement_columns = data.columns[2:]

    means = data[judgement_columns].transpose().mean()
    stds = data[judgement_columns].transpose().std()

    # graphs and stats for individual subjects
    output = []
    for j in judgement_columns:
        ratings = data[j]
        other_subjects = list(set(judgement_columns) - set([j]))
        num_judgements = ratings.notnull().sum()
        exclusive_means = data[other_subjects].transpose().mean()
        rho1, p1 = spearmanr(ratings, means)
        rho2, p2 = spearmanr(ratings, exclusive_means)
        record = {
            'filename': basename(filename).replace(".csv", ""),
            'subject': j,
            'n': num_judgements,
            'min': ratings.min(),
            'max': ratings.max(),
            'q1': ratings.quantile(.25),
            'q3': ratings.quantile(.75),
            'mean': ratings.mean(),
            'median': ratings.median(),
            'stddev': ratings.std(),
            'rho-mean (w/)': rho1,
            'rho-mean (w/o)': rho2,
        }
        for i in xrange(1, 8):
            count = (ratings == i).sum()
            record['n%d' % i] = count

        other_rhos = []
        for k in other_subjects:
            rho_o, p = spearmanr(ratings, data[k])
            other_rhos.append(rho_o)
        record['avg rho w/ others'] = pd.Series(other_rhos).mean()
        record['std rho w/ others'] = pd.Series(other_rhos).std()

        output.append(record)

    headers = [
        'filename', 'subject', 'n', 'min', 'q1', 'mean', 'median', 'q3', 'max',
        'stddev', 'rho-mean (w/)', 'rho-mean (w/o)'
    ]
    headers += ['avg rho w/ others', 'std rho w/ others']
    headers += ['n%d' % i for i in xrange(1, 8)]

    out_data = pd.DataFrame(output)
    out_data.to_csv(sys.stdout, cols=headers, index=False, header=(fileno == 0))

    # first global stats graphs
    pyplot.figure(figsize=(25, 15))
    pyplot.suptitle("Distributions of global results (%s)" % basename(filename), fontsize=32)
    pyplot.subplot(2, 3, 1)
    pyplot.title("Histogram of mean comp judgements")
    pyplot.hist(means, color="blue", range=(1, 7), bins=7, normed=True)
    pyplot.subplot(2, 3, 2)
    pyplot.title("Boxplot of mean comp judgements")
    pyplot.boxplot(means)
    pyplot.subplot(2, 3, 3)
    pyplot.title("Histogram of stddev of mean comp judgements")
    pyplot.hist(stds, color="blue", range=(stds.min(), stds.max()), normed=True)
    pyplot.subplot(2, 3, 4)
    pyplot.title("Mean v. Stddev of comp judgements")
    pyplot.scatter(means, stds)
    pyplot.subplot(2, 3, 5)
    pyplot.title("Histogram of Spearman rho b/t subject and mean comps w/o subject")
    rhos = out_data["rho-mean (w/o)"]
    pyplot.hist(rhos, range=(rhos.min(), rhos.max()), normed=True, color="blue")

    pp.savefig()
    pyplot.clf()



    pyplot.figure(figsize=(25, 10))
    pyplot.suptitle("Boxplot of subject judgements (%s)" % basename(filename), fontsize=32)
    pyplot.boxplot([data[j].values for j in judgement_columns])
    pyplot.xlabel("Subject")
    pyplot.ylabel("Rating")
    pp.savefig()
    pyplot.clf()

    pyplot.figure(figsize=(25,20))
    pyplot.suptitle("Histograms of subject judgements (%s)" % basename(filename), fontsize=32)
    for i, j in enumerate(judgement_columns):
        i = i + 1
        pyplot.subplot(5, 6, i)
        pyplot.hist(data[j], label="Subject " + j, color="blue", range=(1, 7), bins=7, normed=True)
        pyplot.title("Subject " + j)
        pyplot.ylim(0.0, 0.6)
    pp.savefig()
    pyplot.clf()

pp.close()


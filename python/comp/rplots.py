#!/usr/bin/env python

import pandas as pd
from rpy2 import robjects
import rpy2.robjects.lib.ggplot2 as gg2
from rpy2.robjects.packages import importr
import pandas.rpy.common as common


require = robjects.r['require']
require('ggplot2')
pdf = robjects.r['pdf']
grdevices = importr('grDevices')
dev_off = robjects.r['dev.off']
ordered = robjects.r['ordered']
ggtitle = robjects.r['ggtitle']
xlabel = robjects.r['xlab']
ylabel = robjects.r['ylab']
seq = robjects.r['seq']


def line_plot(pdf_file, data, x, y, var,
        null_label="N/A",
        linetype = None,
        title=None,
        xlab=None,
        ylab=None,
        colorname=None,
        linename=None,
        **extra_aes_params):

    pdf(pdf_file, width=11.7, height=8.3, paper="a4r")
    if any(data[x].isnull()):
        labels = [null_label] + map(str, sorted(set(data[data[x].notnull()][x])))
        labels = robjects.StrVector(labels)
        nulls = data[x].isnull()
        label_vals = dict(zip(labels, range(len(labels))))
        data[x] = data[x].astype("str")
        data[x][nulls] = null_label
        data['sortcol'] = data[x].map(label_vals.__getitem__)
        data.sort('sortcol', inplace=True)
    else:
        labels = None

    if linetype and linetype != var:
        data['group'] = data[var].map(str) + data[linetype].map(str)
    else:
        data['group'] = data[var]

    rdata = common.convert_to_r_dataframe(data)
    if labels:
        ix = rdata.names.index(x)
        rdata[ix] = ordered(rdata[ix], levels=labels)

    gp = gg2.ggplot(rdata)
    pp = (gp + gg2.geom_point(size=3) +
            gg2.scale_colour_hue(name=(colorname or var)) +
            #gg2.scale_colour_continuous(low="black") +
            gg2.aes_string(x=x, y=y, color=var, variable=var) +
            ggtitle(title or "") +
            xlabel(xlab or x) +
            ylabel(ylab or y) #+
            #gg2.scale_y_continuous(breaks=seq(0.0, 1.0, 0.05))
            )

    # line type stuff
    if linetype:
        pp += gg2.geom_path(gg2.aes_string(group='group', linetype=linetype), size=0.5)
        pp += gg2.scale_linetype(name=(linename or linetype))
    else:
        pp += gg2.geom_path(gg2.aes_string(group='group'), size=0.5)


    pp.plot()
    dev_off()


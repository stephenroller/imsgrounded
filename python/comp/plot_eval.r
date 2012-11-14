# for plotting the w/ lines

require(ggplot2)

data <- read.csv("~/x2.txt")

data$min_rho[is.na(data$min_rho)] <- "NA"
data$min_rho <- ordered(data$min_rho, levels=unique(data$min_rho))

data$zscore[is.na(data$zscore)] <- "NA"
# data$zscore <- ordered(data$zscore, levels=unique(data$zscore))

data$elo_rerank[is.na(data$elo_rerank)] <- "NA"
data$elo_rerank <- ordered(data$elo_rerank, levels=c("NA", "1"))

data$svd_k[is.na(data$svd_k)] <- "NA"
data$svd_k = ordered(data$svd_k, levels=unique(data$svd_k))

rebin_vals <- unique(data$rebin[!is.na(data$rebin)])
data$rebin[is.na(data$rebin)] <- "NA"
data$rebin <- ordered(data$rebin, levels=c("NA", rebin_vals))

p <- ggplot(data[data$min_rho == "NA" & data$zscore == "NA",], aes(x=variable, fill=elo_rerank, y=value))
p + geom_bar(position="dodge", color="black") + xlab("Evaluation Metric") + ylab("Spearman Rho") + scale_fill_hue(name="Elo Rerank?", h.start=40) + coord_flip()

p <- ggplot(data[data$zscore == "NA" & data$svd_k == "NA",], aes(x=min_rho, y=value, group=variable, color=variable))
p + geom_line(size=1) + geom_point(size=3) + xlab("Minimum subject agreement") + ylab("Spearman Rho") + scale_color_hue(name="Eval Metric")

p <- ggplot(data[data$min_rho == "NA" & data$svd_k == "NA",], aes(x=zscore, group=variable, color=variable, y=value))
p + geom_line(size=1) + geom_point(size=3) + xlab("Maximum Z-score") + ylab("Spearman Rho") + scale_color_hue(name="Eval Metric")

p <- ggplot(data, aes(x=rebin, y=value, color=variable, group=variable))
p + geom_point(size=5) + xlab("Rebin mapping") + ylab("Spearman Rho") + scale_color_hue(name="Eval Metric")

p <- ggplot(data[data$min_rho == "NA" & data$zscore == "NA",], aes(x=svd_k, y=value, color=variable, group=variable))
p + geom_line(size=1) + geom_point(size=3) + xlab("SVD Rank (k)") + ylab("Spearman Rho") + scale_color_hue(name="Eval Metric")

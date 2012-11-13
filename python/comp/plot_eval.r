# for plotting the w/ lines

require(ggplot2)

data <- read.csv("~/x2.txt")
data$dev_subj[is.na(data$dev_subj)] <- "NA"
data$dev_subj <- ordered(data$dev_subj, levels=unique(data$dev_subj))
data$zscore[is.na(data$zscore)] <- "NA"
# data$zscore <- ordered(data$zscore, levels=unique(data$zscore))
data$elo_rerank[is.na(data$elo_rerank)] <- "NA"
data$elo_rerank <- ordered(data$elo_rerank, levels=c("NA", "1"))

rebin_vals <- unique(data$rebin[!is.na(data$rebin)])
data$rebin[is.na(data$rebin)] <- "NA"
data$rebin <- ordered(data$rebin, levels=c("NA", rebin_vals))

p <- ggplot(data[data$dev_subj == "NA" & data$zscore == "NA",], aes(x=variable, fill=elo_rerank, y=value))
p + geom_bar(position="dodge", color="black") + xlab("Evaluation Metric") + ylab("Spearman Rho") + scale_fill_hue(name="Elo Rerank?", h.start=40) + coord_flip()

p <- ggplot(data[data$zscore == "NA" & data$elo_rerank == "NA",], aes(x=dev_subj, y=value, group=variable, color=variable))
p + geom_line(size=1) + geom_point(size=3) + xlab("Minimum subject agreement") + ylab("Spearman Rho") + scale_color_hue(name="Eval Metric")

p <- ggplot(data[data$dev_subj == "NA" & data$elo_rerank == "NA",], aes(x=zscore, group=variable, color=variable, y=value))
p + geom_line(size=1) + geom_point(size=3) + xlab("Maximum Z-score") + ylab("Spearman Rho") + scale_color_hue(name="Eval Metric")

p <- ggplot(data, aes(x=rebin, y=value, color=variable, group=variable))
p + geom_point(size=5) + xlab("Rebin mapping") + ylab("Spearman Rho") + scale_color_hue(name="Eval Metric")

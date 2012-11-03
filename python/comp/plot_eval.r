# for plotting the w/ lines

data <- read.csv("~/x.txt"); View(data)
data$dev_subj[is.na(data$dev_subj)] <- "NA"
data$dev_subj <- ordered(data$dev_subj, levels=unique(data$dev_subj))
data$zscore[is.na(data$zscore)] <- "NA"
data$zscore <- ordered(data$zscore, levels=unique(data$zscore))
p <- ggplot(data[data$zscore == "NA",], aes(x=dev_subj, y=value, group=variable, color=variable))
p + geom_line() + xlab("Minimum subject agreement") + ylab("Correlation") + scale_color_hue(name="Eval Metric")

p <- ggplot(data[data$dev_subj == "NA",], aes(x=zscore, group=variable, color=variable, y=value))
p + geom_line() + xlab("Maximum Z-score") + ylab("Correlation") + scale_color_hue(name="Eval Metric")

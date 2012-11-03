
# individual bar graphs

data <- read.csv("~/x.txt"); View(x)
data$dev_subj[is.na(data$dev_subj)] <- "NA"
data$dev_subj <- ordered(data$dev_subj, levels=unique(data$dev_subj))
data$zscore[is.na(data$zscore)] <- "NA"
data$zscore <- ordered(data$zscore, levels=unique(data$zscore))
p <- ggplot(data[data$zscore == "NA",], aes(x=dev_subj, y=with_wholes_prod))
p + geom_bar(position=position_dodge(), colour="black") + coord_cartesian(ylim=c(0.65, 0.8)) + xlab("Minimum subject agreement") + ylab("Correlation w/ whole judgements")

p <- ggplot(data[data$dev_subj == "NA",], aes(x=zscore, y=with_wholes_prod))
p + geom_bar(position=position_dodge(), colour="black") + coord_cartesian(ylim=c(0.65, 0.8)) + xlab("Minimum subject agreement") + ylab("Correlation w/ whole judgements")


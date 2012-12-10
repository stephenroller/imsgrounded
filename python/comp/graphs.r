require(ggplot2)
require(reshape)
library(grid)

cbbPalette <- c("#000000", "#E69F00", "#3b7ca1", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")


mytheme <- theme_bw() + 
    theme(
        legend.position = "bottom", 
        title = element_text(size=rel(0.90)), 
        legend.text = element_text(size=rel(0.90)), 
        legend.key = element_rect(color = 0),
        # legend.margin = unit(c(0.0, 0.0, 0.0, 0.0), "cm"),
        plot.margin = unit(c(0.0, 0.1, 0.0, 0.1), "cm"),
        panel.margin = unit(0.0, "null")
        )

mycolors <- scale_color_manual(name="", values=cbbPalette)
myfills <- scale_fill_manual(name="", values=cbbPalette)

WIDTH <- 5.5
HEIGHT <- 4.0
NOISE <- 0.25
    
intrinsic_scale <- scale_y_continuous(breaks=seq(0.72, 0.8, by=0.01), limits=c(0.72, 0.8))
extrinsic_scale <- scale_y_continuous(breaks=seq(from=0.4, to=0.65, by=0.05), limits=c(0.40, 0.65))

baseline_line <- geom_hline(y=0.78619053728978872, color="#DDDDDD", size=2)

# -------------------------------------------

cu <- read.csv("~/Working/imsgrounded/results/comp/cleanups.csv")
cu$zscore[is.na(cu$zscore)] <- "None"

cu$zscore2 <-
    ordered(cu$zscore, 
            levels=c("None", "4",   "3.75", "3.5", "3.25", "3",   "2.75", "2.5", "2.25", "2",   "1.75", "1.5", "1.25", "1"),
            labels=c("N/A",  "4.0", "3.75", "3.5", "3.25", "3.0", "2.75", "2.5", "2.25", "2.0", "1.75", "1.5", "1.25", "1.0"))

cu$variable2 <-
    ordered(cu$variable,
            levels=c("Association Norms (Indiv)", "Association Norms (Whole)", "Parts Cleaned", "Whole Cleaned", "Parts & Whole Cleaned"),
            labels=c("Assoc Norms (Indiv)", "Assoc Norms (Whole)", "Cleaned Indiv", "Cleaned Whole", "Cleaned Indiv & Whole"))

pdf("~/Working/papers/germany/comp/graphs/zscore-intrinsic.pdf", width=WIDTH, height=HEIGHT)
D <- cu[cu$zscore2 != "N/A" | (cu$zscore2 == "N/A" & is.na(cu$min_rho) & is.na(cu$svd_k)),]
D <- D[D$variable2 != "Assoc Norms (Indiv)" & D$variable2 != "Assoc Norms (Whole)",]
ggplot(data=D, aes(x=zscore2, y=value, group=variable2)) + 
    baseline_line +
    geom_line(size=1, aes(color=variable2, linetype=variable2)) + 
    geom_point(aes(color=variable2)) +
    mytheme +
    xlab("Maximum Z-score of Judgements") +
    ylab("Consistency between ratings\n(Spearman's rho)") + 
    scale_x_discrete(breaks=c("N/A", "4.0","3.0","2.0","1.0")) +
    ggtitle("(a) Intrinsic Evaluation of Z-score Filtering") +
    scale_linetype(name="") +
    intrinsic_scale +
    mycolors
dev.off()

pdf("~/Working/papers/germany/comp/graphs/zscore-extrinsic.pdf", width=WIDTH, height=HEIGHT)
D <- cu[cu$zscore2 != "N/A" | (cu$zscore2 == "N/A" & is.na(cu$min_rho) & is.na(cu$svd_k)),]
D <- D[D$variable2 == "Assoc Norms (Indiv)" | D$variable2 == "Assoc Norms (Whole)",]
ggplot(data=D, aes(x=zscore2, y=value, group=variable2)) + 
    geom_line(size=1, aes(color=variable2, linetype=variable2)) + 
    geom_point(aes(color=variable2)) +
    mytheme +
    xlab("Maximum Z-score of Judgements") +
    ylab("Correlation with Association Norm Overlap\n(Spearman's rho)") + 
    scale_x_discrete(breaks=c("N/A", "4.0","3.0","2.0","1.0")) +
    ggtitle("(b) Extrinsic Evaluation of Z-score Filtering") +
    extrinsic_scale +
    scale_linetype(name="") +
    mycolors
dev.off()

pdf("~/Working/papers/germany/comp/graphs/zscore-retained.pdf", width=WIDTH, height=HEIGHT)
D <- cu[cu$zscore2 != "N/A" | (cu$zscore2 == "N/A" & is.na(cu$min_rho) & is.na(cu$svd_k)),]
D <- D[D$variable2 == "Assoc Norms (Indiv)",]
D$value <- NULL
R <- melt(D, measure.vars=c("retained_indiv", "retained_whole", "retained_both"), variable_name="retention_measure")
R$retention_measure <- ordered(R$retention_measure, levels=c("retained_indiv", "retained_whole", "retained_both"),
                        labels=c("Indiv", "Whole", "Both"))
ggplot(data=R, aes(x=zscore2, group=retention_measure, y=value, linetype=retention_measure, color=retention_measure)) + 
    mytheme + mycolors +
    geom_line(size=1) +
    geom_point() +
    xlab("Maximum Z-score of Judgements") +
    ylab("Fraction Data Retained") + 
    ggtitle("Data Retention with Z-score Filtering") +
    scale_x_discrete(breaks=c("N/A", "4.0","3.0","2.0","1.0")) +
    scale_y_continuous(breaks=seq(0, 1, by=0.10), limits=c(0, 1)) +
    scale_linetype(name="")
dev.off()

pdf("~/Working/papers/germany/comp/graphs/minrho-intrinsic.pdf", width=WIDTH, height=HEIGHT)
D <- cu[!is.na(cu$min_rho) | (cu$zscore2 == "N/A" & is.na(cu$min_rho) & is.na(cu$svd_k)),]
D <- D[D$variable2 != "Assoc Norms (Indiv)" & D$variable2 != "Assoc Norms (Whole)",]
ggplot(data=D, aes(x=min_rho, y=value, group=variable2)) + 
    baseline_line + 
    geom_line(size=1, aes(color=variable2, linetype=variable2)) + 
    geom_point(aes(color=variable2)) +
    mytheme + mycolors +
    xlab("Minimum Subject-Average Correlation\n(Spearman's rho)") +
    ylab("Consistency between ratings\n(Spearman's rho)") +
    ggtitle("(a) Intrinsic Evaluation of MSA Filtering") +
    scale_x_continuous(breaks=seq(0.1, 0.6, 0.1)) +
    intrinsic_scale +
    scale_linetype(name="")
dev.off()

pdf("~/Working/papers/germany/comp/graphs/minrho-extrinsic.pdf", width=WIDTH, height=HEIGHT)
D <- cu[!is.na(cu$min_rho) | (cu$zscore2 == "N/A" & is.na(cu$min_rho) & is.na(cu$svd_k)),]
D <- D[D$variable2 == "Assoc Norms (Indiv)" | D$variable2 == "Assoc Norms (Whole)",]
ggplot(data=D, aes(x=min_rho, y=value, group=variable2)) + 
    geom_line(size=1, aes(color=variable2, linetype=variable2)) + 
    geom_point(aes(color=variable2)) +
    mytheme + mycolors +
    xlab("Minimum Subject-Average Correlation\n(Spearman's rho)") +
    ylab("Correlation with Association Norm Overlap\n(Spearman's rho)") +
    ggtitle("(b) Extrinsic Evaluation of MSA Filtering") +
    scale_x_continuous(breaks=seq(0.1, 0.6, 0.1)) +
    extrinsic_scale +
    scale_linetype(name="")
dev.off()

pdf("~/Working/papers/germany/comp/graphs/minrho-retained.pdf", width=WIDTH, height=HEIGHT)
D <- cu[!is.na(cu$min_rho) | (cu$zscore2 == "N/A" & is.na(cu$min_rho) & is.na(cu$svd_k)),]
D <- D[D$variable2 == "Assoc Norms (Indiv)",]
D$value <- NULL
R <- melt(D, measure.vars=c("retained_indiv", "retained_whole", "retained_both"), variable_name="retention_measure")
R$retention_measure <- ordered(R$retention_measure, levels=c("retained_indiv", "retained_whole", "retained_both"),
                        labels=c("Indiv", "Whole", "Both"))
ggplot(data=R, aes(x=min_rho, group=retention_measure, y=value, linetype=retention_measure, color=retention_measure)) + 
    mytheme + mycolors +
    geom_line(size=1) +
    geom_point() +
    ylab("Fraction Data Retained") +
    ggtitle("Data Retention with MSA Filtering") + 
    xlab("Minimum Subject-Average Correlation\n(Spearman's rho)") +
    scale_x_continuous(breaks=seq(0.1, 0.6, 0.1)) +
    scale_y_continuous(breaks=seq(0, 1, by=0.10), limits=c(0, 1)) +
    scale_linetype(name="")
dev.off()













cu$svd_k <-
    ordered(cu$svd_k, 
            levels=c("None", "20", "19", "18", "17", "16", "15", "14", "13", "12", "11", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1"),
            labels=c("Full", "20", "19", "18", "17", "16", "15", "14", "13", "12", "11", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1"))


cu$svd_k[is.na(cu$svd_k)] <- "Full"
D <- cu[cu$svd_k != "Full" | (cu$zscore2 == "N/A" & is.na(cu$min_rho) & cu$svd_k == "Full"),]
D <- D[D$variable2 != "Assoc Norms (Indiv)" & D$variable2 != "Assoc Norms (Whole)",]
pdf("~/Working/papers/germany/comp/graphs/svd-intrinsic.pdf", width=WIDTH, height=HEIGHT)
ggplot(data=D, aes(x=svd_k, y=value, group=variable2)) + 
    baseline_line + 
    geom_line(size=1, aes(color=variable2, linetype=variable2)) + 
    geom_point(aes(color=variable2)) +
    mytheme +
    xlab("Retained Dimensionality (k)") +
    ylab("Consistency between ratings\n(Spearman's rho)") + 
    scale_x_discrete(breaks=c("Full", "10", "5", "1"), limits=c("Full", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1")) +
    intrinsic_scale +
    scale_linetype(name="") +
    mycolors +
    ggtitle("(a) Dimensionality Reduction Intrinsic Evaluation")
dev.off()

D <- cu[cu$svd_k != "Full" | (cu$zscore2 == "N/A" & is.na(cu$min_rho) & cu$svd_k == "Full"),]
D <- D[D$variable2 == "Assoc Norms (Indiv)" | D$variable2 == "Assoc Norms (Whole)",]
pdf("~/Working/papers/germany/comp/graphs/svd-extrinsic.pdf", width=WIDTH, height=HEIGHT)
ggplot(data=D, aes(x=svd_k, y=value, group=variable2, color=variable2)) + 
    geom_line(size=1, aes(linetype=variable2)) + 
    geom_point(aes(color=variable2)) +
    mytheme +
    xlab("Retained Dimensionality (k)") +
    ylab("Correlation with Association Norms\n(Spearman's rho)") + 
    # scale_x_discrete(breaks=c("Full", "20", "15", "10", "5", "1")) +
    scale_x_discrete(breaks=c("Full", "10", "5", "1"), limits=c("Full", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1")) +
    scale_linetype(name="") +
    extrinsic_scale +
    mycolors +
    ggtitle("(b) Dimensionality Reduction Extrinsic Evaluation")
dev.off()


# --------------------------------------------

# -------

mr <- read.csv("~/Working/imsgrounded/results/comp/reshaped-minrho.csv")

mr$variable2 <- 
    ordered(mr$variable, 
            levels=c("orig_clean", "clean_orig", "orig_noisy", "noisy_orig", "clean_noisy", "noisy_clean", "clean_clean", "noisy_noisy"),
            labels=c("Cleaned Whole", "Cleaned Indiv", "Noisy Whole", "Noisy Indiv", "Cleaned Indiv, Noisy Whole", "Noisy Indiv, Cleaned Whole", "Cleaned Indiv & Whole", "Noisy Indiv & Whole"))

pdf("~/Working/papers/germany/comp/graphs/minrho-noise-indiv.pdf", width=WIDTH, height=HEIGHT)
ggplot(data=mr[mr$variable %in% c("clean_orig", "noisy_orig" ),], aes(x=n, y=mean, group=variable, linetype=variable2)) + 
    baseline_line + 
    geom_line(size=1, aes(color=variable2)) + 
    geom_point(aes(color=variable2)) +
    geom_ribbon(alpha=0.2, aes(ymin=(mean - 2*std/sqrt(trials)), ymax=(mean + 2*std/sqrt(trials)), fill=variable2)) + 
    mytheme +
    xlab("Number of Subjects Randomized/Removed") + 
    ylab("Consistency between ratings\n(Spearman's rho)") + 
    ggtitle("(a) Removing Indiv Subjects with Artificial Noise") +
    scale_linetype(name="") + mycolors + myfills +
    scale_y_continuous(breaks=seq(from=0.2, to=0.8, by=0.1), limits=c(0.15, 0.85))
dev.off()


pdf("~/Working/papers/germany/comp/graphs/minrho-noise-whole.pdf", width=WIDTH, height=HEIGHT)
ggplot(data=mr[mr$variable %in% c("orig_clean", "orig_noisy"),], aes(x=n, y=mean, group=variable, linetype=variable2)) + 
    baseline_line + 
    geom_line(size=1, aes(color=variable2)) + 
    geom_point(aes(color=variable2)) +
    geom_ribbon(alpha=0.2, aes(ymin=(mean - 2*std/sqrt(trials)), ymax=(mean + 2*std/sqrt(trials)), fill=variable2)) + 
    mytheme +
    xlab("Number of Subjects Randomized/Removed") + 
    ylab("Consistency between ratings\n(Spearman's rho)") + 
    ggtitle("(b) Removing Whole Subjects with Artificial Noise") +
    # scale_y_continuous(breaks=seq(from=0.2, to=0.8, by=0.1), limits=c(0.15, 0.85)) +
    scale_y_continuous(breaks=c(0.65, 0.7, 0.75, 0.8), limits=c(0.65, 0.8)) +
    mycolors + myfills + scale_linetype(name="")
dev.off()

# --------------------------------



zu <- read.csv("~/Working/imsgrounded/results/comp/reshaped-zscore-uniform.csv")

zu$variable2 <- 
    ordered(zu$variable, 
            levels=c("orig_clean", "clean_orig", "orig_noisy", "noisy_orig", "clean_noisy", "noisy_clean", "clean_clean", "noisy_noisy"),
            labels=c("Cleaned Whole", "Cleaned Indiv", "Noisy Whole", "Noisy Indiv", "Cleaned Indiv, Noisy Whole", "Noisy Indiv, Cleaned Whole", "Cleaned Indiv & Whole", "Noisy Indiv & Whole"))

zu$zscore <-
    ordered(zu$zscore, 
            levels=c("None", "4.0", "3.75", "3.5", "3.25", "3.0", "2.75", "2.5", "2.25", "2.0", "1.75", "1.5", "1.25", "1.0"),
            labels=c("N/A", "4.0", "3.75", "3.5", "3.25", "3.0", "2.75", "2.5", "2.25", "2.0", "1.75", "1.5", "1.25", "1.0"))

zu$group = paste(zu$variable, zu$p)

pdf("~/Working/papers/germany/comp/graphs/zscore-noise-uniform-whole.pdf", width=WIDTH, height=HEIGHT)
D <- cu[cu$zscore2 != "N/A" | (cu$zscore2 == "N/A" & is.na(cu$min_rho) & is.na(cu$svd_k)),]
D <- D[D$variable2 == "Cleaned Whole",]
ggplot(data=zu[zu$variable %in% c("orig_clean", "orig_noisy") & zu$p %in% c(NOISE),], aes(x=zscore, y=mean, group=group, linetype=variable2)) + 
    baseline_line + 
    geom_line(size=1, aes(color=variable2)) + 
    geom_point(aes(color=variable2)) +
    geom_ribbon(alpha=0.2, aes(ymin=(mean - 2*std/sqrt(trials)), ymax=(mean + 2*std/sqrt(trials)), fill=variable2)) + 
    mytheme +
    xlab("Maximum Z-score of Judgements") + 
    ylab("Consistency between ratings\n(Spearman's rho)") + 
    ggtitle("(b) Removing Whole Judgements with Uniform Noise") +
    mycolors + myfills + scale_linetype(name="") +
    scale_y_continuous(breaks=c(0.65, 0.7, 0.75, 0.8), limits=c(0.65, 0.8)) +
    scale_x_discrete(breaks=c("N/A", "4.0","3.0","2.0","1.0"))
dev.off()

pdf("~/Working/papers/germany/comp/graphs/zscore-noise-uniform-indiv.pdf", width=WIDTH, height=HEIGHT)
D <- cu[cu$zscore2 != "N/A" | (cu$zscore2 == "N/A" & is.na(cu$min_rho) & is.na(cu$svd_k)),]
D <- D[D$variable2 == "Cleaned Indiv",]
ggplot(data=zu[zu$variable %in% c("clean_orig", "noisy_orig") & zu$p %in% c(NOISE),], aes(x=zscore, y=mean, group=group, linetype=variable2)) + 
    baseline_line + 
    geom_line(size=1, aes(color=variable2)) + 
    geom_point(aes(color=variable2)) +
    geom_ribbon(alpha=0.2, aes(ymin=(mean - 2*std/sqrt(trials)), ymax=(mean + 2*std/sqrt(trials)), fill=variable2)) + 
    mytheme +
    xlab("Maximum Z-score of Judgements") + 
    ylab("Consistency between ratings\n(Spearman's rho)") + 
    ggtitle("(a) Removing Indiv Judgements with Uniform Noise") +
    mycolors + myfills + scale_linetype(name="") +
    scale_y_continuous(breaks=c(0.65, 0.7, 0.75, 0.8), limits=c(0.65, 0.8)) +
    scale_x_discrete(breaks=c("N/A", "4.0","3.0","2.0","1.0"))
dev.off()

# --------------


su <- read.csv("~/Working/imsgrounded/results/comp/reshaped-svd-uniform.csv")

su$variable2 <- 
    ordered(su$variable, 
            levels=c("orig_clean", "clean_orig", "orig_noisy", "noisy_orig", "clean_noisy", "noisy_clean", "clean_clean", "noisy_noisy"),
            labels=c("Cleaned Whole", "Cleaned Indiv", "Noisy Whole", "Noisy Indiv", "Cleaned Indiv, Noisy Whole", "Noisy Indiv, Cleaned Whole", "Cleaned Indiv & Whole", "Noisy Indiv & Whole"))

su$k <-
    ordered(su$k, 
            levels=c("None", "20", "19", "18", "17", "16", "15", "14", "13", "12", "11", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1"),
            labels=c("Full", "20", "19", "18", "17", "16", "15", "14", "13", "12", "11", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1"))

su$group = paste(su$variable, su$p)

pdf("~/Working/papers/germany/comp/graphs/svd-noise-uniform-whole.pdf", width=WIDTH, height=HEIGHT)
ggplot(data=su[su$variable %in% c("orig_clean", "orig_noisy") & su$p %in% c(NOISE),], aes(x=k, y=mean, group=group, linetype=variable2)) + 
    baseline_line + 
    geom_line(size=1, aes(color=variable2)) + 
    geom_ribbon(alpha=0.2, aes(ymin=(mean - 2*std/sqrt(trials)), ymax=(mean + 2*std/sqrt(trials)), fill=variable2)) + 
    mytheme +
    xlab("Dimensionality (k)") + 
    ylab("Consistency between ratings\n(Spearman's rho)") + 
    ggtitle("(b) Whole Dimensionality Reduction with Uniform Noise") +
    mycolors + myfills + scale_linetype(name="") +
    scale_y_continuous(breaks=c(0.65, 0.7, 0.75, 0.8), limits=c(0.65, 0.8)) +
    scale_x_discrete(breaks=c("Full", "20","15","10","5","1"))
dev.off()

pdf("~/Working/papers/germany/comp/graphs/svd-noise-uniform-indiv.pdf", width=WIDTH, height=HEIGHT)
ggplot(data=su[su$variable %in% c("clean_orig", "noisy_orig") & su$p %in% c(NOISE),], aes(x=k, y=mean, group=group, linetype=variable2)) + 
    baseline_line + 
    geom_line(size=1, aes(color=variable2)) + 
    geom_ribbon(alpha=0.2, aes(ymin=(mean - 2*std/sqrt(trials)), ymax=(mean + 2*std/sqrt(trials)), fill=variable2)) + 
    mytheme +
    xlab("Dimensionality (k)") + 
    ylab("Consistency between ratings\n(Spearman's rho)") + 
    ggtitle("(a) Indiv Dimensionality Reduction with Uniform Noise") +
    mycolors + myfills + scale_linetype(name="") +
    # intrinsic_scale + 
    scale_y_continuous(breaks=c(0.65, 0.7, 0.75, 0.8), limits=c(0.65, 0.8)) +
    scale_x_discrete(breaks=c("Full", "20","15","10","5","1"))
dev.off()

# -------------------

so <- read.csv("~/Working/imsgrounded/results/comp/reshaped-svd-offset.csv")

so$variable2 <- 
    ordered(so$variable, 
            levels=c("orig_clean", "clean_orig", "orig_noisy", "noisy_orig", "clean_noisy", "noisy_clean", "clean_clean", "noisy_noisy"),
            labels=c("Cleaned Whole", "Cleaned Indiv", "Noisy Whole", "Noisy Indiv", "Cleaned Indiv, Noisy Whole", "Noisy Indiv, Cleaned Whole", "Cleaned Indiv & Whole", "Noisy Indiv & Whole"))

so$k <-
    ordered(so$k, 
            levels=c("None", "20", "19", "18", "17", "16", "15", "14", "13", "12", "11", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1"),
            labels=c("Full", "20", "19", "18", "17", "16", "15", "14", "13", "12", "11", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1"))

so$group = paste(so$variable, so$p)

pdf("~/Working/papers/germany/comp/graphs/svd-noise-offset-whole.pdf", width=WIDTH, height=HEIGHT)
ggplot(data=so[so$variable %in% c("orig_clean", "orig_noisy") & so$p %in% c(NOISE),], aes(x=k, y=mean, group=group, linetype=variable2)) + 
    baseline_line + 
    geom_line(size=1, aes(color=variable2)) + 
    geom_ribbon(alpha=0.2, aes(ymin=(mean - 2*std/sqrt(trials)), ymax=(mean + 2*std/sqrt(trials)), fill=variable2)) + 
    mytheme +
    xlab("Dimensionality (k)") + 
    ylab("Consistency between ratings\n(Spearman's rho)") + 
    ggtitle("(b) Whole Dimensionality Reduction with Offset Noise") +
    mycolors + myfills + scale_linetype(name="") +
    scale_y_continuous(breaks=c(0.6, 0.65, 0.7, 0.75, 0.8), limits=c(0.60, 0.8)) +
    scale_x_discrete(breaks=c("Full", "20","15","10","5","1"))
dev.off()

pdf("~/Working/papers/germany/comp/graphs/svd-noise-offset-indiv.pdf", width=WIDTH, height=HEIGHT)
ggplot(data=so[so$variable %in% c("clean_orig", "noisy_orig") & so$p %in% c(NOISE),], aes(x=k, y=mean, group=group, linetype=variable2)) + 
    baseline_line + 
    geom_line(size=1, aes(color=variable2)) + 
    geom_ribbon(alpha=0.2, aes(ymin=(mean - 2*std/sqrt(trials)), ymax=(mean + 2*std/sqrt(trials)), fill=variable2)) + 
    mytheme +
    xlab("Dimensionality (k)") + 
    ylab("Consistency between ratings\n(Spearman's rho)") + 
    ggtitle("(a) Indiv Dimensionality Reduction with Offset Noise") +
    mycolors + myfills + scale_linetype(name="") +
    scale_y_continuous(breaks=c(0.65, 0.7, 0.75, 0.8), limits=c(0.65, 0.8)) +
    scale_x_discrete(breaks=c("Full", "20","15","10","5","1"))
dev.off()



# -----------------------


zo <- read.csv("~/Working/imsgrounded/results/comp/reshaped-zscore-offset.csv")

zo$variable2 <- 
    ordered(zo$variable, 
            levels=c("orig_clean", "clean_orig", "orig_noisy", "noisy_orig", "clean_noisy", "noisy_clean", "clean_clean", "noisy_noisy"),
            labels=c("Cleaned Whole", "Cleaned Indiv", "Noisy Whole", "Noisy Indiv", "Cleaned Indiv, Noisy Whole", "Noisy Indiv, Cleaned Whole", "Cleaned Indiv & Whole", "Noisy Indiv & Whole"))

zo$zscore <-
    ordered(zo$zscore, 
            levels=c("None", "4.0", "3.75", "3.5", "3.25", "3.0", "2.75", "2.5", "2.25", "2.0", "1.75", "1.5", "1.25", "1.0"),
            labels=c("N/A", "4.0", "3.75", "3.5", "3.25", "3.0", "2.75", "2.5", "2.25", "2.0", "1.75", "1.5", "1.25", "1.0"))

zo$group = paste(zo$variable, zo$p)

pdf("~/Working/papers/germany/comp/graphs/zscore-noise-offset-whole.pdf", width=WIDTH, height=HEIGHT)
D <- cu[cu$zscore2 != "N/A" | (cu$zscore2 == "N/A" & is.na(cu$min_rho) & is.na(cu$svd_k)),]
D <- D[D$variable2 == "Cleaned Whole",]
ggplot(data=zo[zo$variable %in% c("orig_clean", "orig_noisy") & zo$p %in% c(NOISE),], aes(x=zscore, y=mean, group=group, linetype=variable2)) + 
    baseline_line + 
    geom_line(size=1, aes(color=variable2)) + 
    geom_ribbon(alpha=0.2, aes(ymin=(mean - 2*std/sqrt(trials)), ymax=(mean + 2*std/sqrt(trials)), fill=variable2)) + 
    mytheme +
    xlab("Maximum Z-score of Judgements") + 
    ylab("Consistency between ratings\n(Spearman's rho)") + 
    ggtitle("Removing Judgements with Offset Noise") +
    mycolors + myfills + scale_linetype(name="") +
    scale_y_continuous(breaks=c(0.65, 0.7, 0.75, 0.8), limits=c(0.65, 0.8)) + 
    scale_x_discrete(breaks=c("N/A", "4.0","3.0","2.0","1.0"))




dev.off()

pdf("~/Working/papers/germany/comp/graphs/zscore-noise-offset-indiv.pdf", width=WIDTH, height=HEIGHT)
D <- cu[cu$zscore2 != "N/A" | (cu$zscore2 == "N/A" & is.na(cu$min_rho) & is.na(cu$svd_k)),]
D <- D[D$variable2 == "Cleaned Indiv",]
ggplot(data=zo[zo$variable %in% c("clean_orig", "noisy_orig") & zo$p %in% c(NOISE),], aes(x=zscore, y=mean, group=group, linetype=variable2)) + 
    baseline_line + 
    geom_line(size=1, aes(color=variable2)) + 
    geom_ribbon(alpha=0.2, aes(ymin=(mean - 2*std/sqrt(trials)), ymax=(mean + 2*std/sqrt(trials)), fill=variable2)) + 
    mytheme +
    xlab("Maximum Z-score of Judgements") + 
    ylab("Consistency between ratings\n(Spearman's rho)") + 
    ggtitle("Removing Judgements with Offset Noise") +
    mycolors + myfills + scale_linetype(name="") +
    scale_y_continuous(breaks=c(0.65, 0.7, 0.75, 0.8), limits=c(0.65, 0.8)) +
    scale_x_discrete(breaks=c("N/A", "4.0","3.0","2.0","1.0"))
dev.off()




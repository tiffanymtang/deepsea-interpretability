# R script for making visualizations for the final presentation
library(tidyverse)
library(TFBSTools)
library(PWMEnrich)
library(reshape2)
library(seqLogo)
library(ggseqlogo)

# load the pwms (in a txt file)
pwm_files <- list.files("../data/pwms")
pwms <- lapply(paste0("../data/pwms/", pwm_files[7]), function(file){
  tmp <- read.delim(file, sep = "", header = FALSE)
  indices <- seq(1, nrow(tmp), 4)
  indices <- c(indices, nrow(tmp) + 1)
  lapply(1:(length(indices) - 1), function(i){
    tmp2 <- tmp[indices[i]:(indices[i + 1] - 1), ] %>% as.matrix()
    rownames(tmp2) <- c("A", "C", "G", "T")
    tmp2
  })
})

# compute similarity between all of the matrices 
pwm_sim <- lapply(1:length(pwms), function(i){
  sim <- matrix(rep(0, length(pwms[[i]]) * length(pwms[[i]])), 
                nrow = length(pwms[[i]]))
  for(j in 1:nrow(sim)){
    for(k in 1:j){
      sim[j, k] <- PWMSimilarity(pwms[[i]][[j]], pwms[[i]][[k]])
      sim[k, j] <- sim[j, k]
    }
  }
  sim
})

# let's look at the heatmap
order <- hclust(1 - as.dist(pwm_sim[[2]]))$order
sim_mat <- pwm_sim[[2]] 
ggplot(melt(sim_mat[order, order])) + 
  geom_tile(aes(x = Var1, y = Var2, fill = value))

# plot all of the sequence logos 
seq_logos <- lapply(1:320, function(i){
  ggseqlogo(pwms[[1]][[i]]) + 
    theme(axis.title = element_blank(), 
          axis.text.x = element_blank(), 
          axis.text.y = element_blank()) + 
    ylim(c(0, 2))
})
seq_logos_rev <- lapply(1:length(pwms[[1]]), function(i){
  tmp <- flip.matrix(mirror.matrix(pwms[[1]][[i]]))
  rownames(tmp) <- c("A", "C", "G", "T")
  ggseqlogo(tmp) + 
    theme(axis.title = element_blank(), 
          axis.text.x = element_blank(), 
          axis.text.y = element_blank()) + 
    ylim(c(0, 2))
})

for(i in 1:320){
  png(filename = paste0("logos/filter", str_pad(i, 3, pad = "0"), ".png"), 
      height = 0.5, width = 1, units = "in", res = 150)
  gridExtra::grid.arrange(seq_logos[[i]] + 
                            theme(plot.margin=grid::unit(c(0,0,0,0), "mm")))
  dev.off()
}

# plotting the seq logos for the paper 
# load tomtom data
tomtom <- read.delim("../data/tomtom/PWMs_max_active_per_seq.tsv", sep = "\t")

# jaspar data
jaspar_files <- list.files("../data/paper-jaspar")
jaspar_pwms <- lapply(jaspar_files, function(file){
  tmp <- read.table(paste0("../data/paper-jaspar/", file), 
                    sep = " ", header = FALSE)
  tmp <- tmp[, c(1, 3, 5, 7)]
  colnames(tmp) <- c("A", "C", "G", "T")
  tmp
})

jaspar_plots <- lapply(1:2, function(i){
  tmp <- t(jaspar_pwms[[i]])
  rownames(tmp) <- c("A", "C", "G", "T")
  ggseqlogo(tmp) + 
    theme(axis.title = element_blank(), 
          axis.text.x = element_blank(), 
          axis.text.y = element_blank()) + 
    ylim(c(0, 2))
})

# ctcf plots 
ids <- c(276, 79, 279, 17, 292)
ctcf_plts <- lapply(1:5, function(i){
  orientation <- (tomtom %>% 
                    filter(Query_ID == ids[i], 
                           Target_ID == "MA0139.1"))$Orientation %>%
    as.character()
  ifelse(orientation == "+", 
         seq_logos[ids[i]],
         seq_logos_rev[ids[i]]
  )
})
png(filename = "ctcf_pwm.png", width = 295, height = 60)
ggpubr::ggarrange(jaspar_plots[[1]])
dev.off()

png(filename = "ctcf_matches.png", width = 200, height = 240)
ggpubr::ggarrange(plotlist = list(ctcf_plts[[1]][[1]], 
                                  ctcf_plts[[2]][[1]],
                                  ctcf_plts[[3]][[1]], 
                                  ctcf_plts[[4]][[1]], 
                                  ctcf_plts[[5]][[1]]),
                  ncol = 1)
dev.off()

ids <- c(121, 108, 304, 266, 234)
junb_plts <- lapply(1:5, function(i){
  orientation <- (tomtom %>% 
                    filter(Query_ID == ids[i], 
                           Target_ID == "MA0490.2"))$Orientation %>%
    as.character()
  ifelse(orientation == "+", 
         seq_logos[ids[i]],
         seq_logos_rev[ids[i]]
  )
})
png(filename = "junb_pwm.png", width = 295, height = 60)
ggpubr::ggarrange(jaspar_plots[[2]])
dev.off()

png(filename = "junb_matches.png", width = 200, height = 240)
ggpubr::ggarrange(plotlist = list(junb_plts[[1]][[1]], 
                                  junb_plts[[2]][[1]],
                                  junb_plts[[3]][[1]], 
                                  junb_plts[[4]][[1]], 
                                  junb_plts[[5]][[1]]),
                  ncol = 1)
dev.off()
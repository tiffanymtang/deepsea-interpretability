# R script for making visualizations for the final presentation
library(tidyverse)
library(TFBSTools)
library(PWMEnrich)
library(reshape2)
library(seqLogo)
library(ggseqlogo)
library(Thermimage)

# load the pwms (in a txt file)
tmp <- read.delim("data/pwms/pwms_modisco_trans.txt", sep = "", header = FALSE)
na_cols <- which(!complete.cases(tmp))
na_cols <- c(0, na_cols, nrow(tmp) + 1)
pwms <- lapply(1:(length(na_cols) - 1), function(i){
  tmp2 <- tmp[(na_cols[i] + 1):(na_cols[i + 1] - 1), ] %>% as.matrix()
  colnames(tmp2) <- c("A", "C", "G", "T")
  tmp2
})

# read in tomtom data
tomtom <- read.delim("data/tomtom/PWMs_modisco_trans.tsv", sep = "\t")

# make sequence logos
seq_logos <- lapply(1:length(pwms), function(i){
  ggseqlogo(t(pwms[[i]])) + 
    theme(axis.title = element_blank(), 
          axis.text.x = element_blank(), 
          axis.text.y = element_blank()) + 
    ylim(c(0, 2))
})
ggpubr::ggarrange(plotlist = seq_logos[sample(1:122, 10)])

for(i in 1:length(pwms)){
  png(filename = paste0("modisco_logos/", str_pad(i, 3, pad = "0"), ".png"), 
      height = 0.5, width = 1, units = "in", res = 150)
  gridExtra::grid.arrange(seq_logos[[i]] + 
                            theme(plot.margin=grid::unit(c(0,0,0,0), "mm")))
  dev.off()
}

# make reverse complement images
seq_logos_rev <- lapply(1:length(pwms), function(i){
  tmp <- t(flip.matrix(mirror.matrix(pwms[[i]])))
  rownames(tmp) <- c("A", "C", "G", "T")
  ggseqlogo(tmp) + 
    theme(axis.title = element_blank(), 
          axis.text.x = element_blank(), 
          axis.text.y = element_blank()) + 
    ylim(c(0, 2))
})
ggpubr::ggarrange(seq_logos[[2]], seq_logos_rev[[2]])

for(i in 1:length(pwms)){
  png(filename = paste0("modisco_logos/", str_pad(i, 3, pad = "0"), ".png"), 
      height = 0.5, width = 1, units = "in", res = 150)
  gridExtra::grid.arrange(seq_logos[[i]] + 
                            theme(plot.margin=grid::unit(c(0,0,0,0), "mm")))
  dev.off()
}

# get query IDs of top 5 matches for each TF 
jaspar_meta <- read.table("data/jaspar_meta.txt")
tfs <- c("EGR1", "CEBPB", "E2F6", "ELF1")
ids <- jaspar_meta$matrix_id[jaspar_meta$tf_name %in% tfs] %>% as.character
query_ids <- sapply(ids, function(id){
  (tomtom %>% filter(Target_ID == id) %>% 
    arrange(p.value) %>%
    head(5))$Query_ID
})

# jaspar PWMS 
jaspar_files <- list.files("data/tf-modisco-jaspar")
jaspar_files <- jaspar_files[c(4, 1, 2, 3)]
jaspar_pwms <- lapply(jaspar_files, function(file){
  tmp <- read.table(paste0("data/tf-modisco-jaspar/", file), 
                    sep = " ", header = FALSE)
  tmp <- tmp[, c(1, 3, 5, 7)]
  colnames(tmp) <- c("A", "C", "G", "T")
  tmp
})

jaspar_plots <- lapply(1:4, function(i){
  tmp <- t(jaspar_pwms[[i]])
  rownames(tmp) <- c("A", "C", "G", "T")
  ggseqlogo(tmp) + 
    theme(axis.title = element_blank(), 
          axis.text.x = element_blank(), 
          axis.text.y = element_blank()) + 
    ylim(c(0, 2))
})

plts <- lapply(1:ncol(query_ids), function(j){
  tmp <- lapply(1:nrow(query_ids), function(i){
    id <- colnames(query_ids)[j]
    orientation <- (tomtom %>% 
                      filter(Query_ID == query_ids[i, j], 
                             Target_ID == id))$Orientation
    ifelse(orientation == "+", 
           seq_logos[query_ids[i, j]],
           seq_logos_rev[query_ids[i, j]]
           )
  })
  list(jaspar_plots[[j]], 
       tmp[[1]][[1]], tmp[[2]][[1]], tmp[[3]][[1]], 
       tmp[[4]][[1]], tmp[[5]][[1]])
})

p1 <- ggpubr::ggarrange(plotlist = plts[[1]], ncol = 1)
p2 <- ggpubr::ggarrange(plotlist = plts[[2]], ncol = 1)
p3 <- ggpubr::ggarrange(plotlist = plts[[3]], ncol = 1)
p4 <- ggpubr::ggarrange(plotlist = plts[[4]], ncol = 1)

png(filename = "tf_modisco.png", width = 600, height = 300)
ggpubr::ggarrange(p1, p2, p3, p4, ncol = 4)
dev.off()
# for these top matches are they the top of all matches? 
sapply(ids, function(id){
  (tomtom %>% filter(Target_ID == id) %>% 
     arrange(p.value) %>%
     head(10))$p.value
}) %>% round(3)

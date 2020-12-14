# Looking at the layers of the network
library(tidyverse)
library(reshape2)
library(matrixStats)
library(ggpubr)
library(ggfortify)
library(jaccard)
library(ggrepel)
library(grid)
library(gridGraphics)
library(png)
library(ggimage)

# load JASPAR results 
tomtom <- read.delim("data/tomtom/PWMs_max_active_per_seq.tsv", sep = "\t")

# load the pwms (in a txt file)
first_tmp <- read.delim("data/pwms/layer1.txt", sep = "", header = FALSE)
indices <- seq(1, nrow(first_tmp), 4)
indices <- c(indices, nrow(first_tmp) + 1)
first <- lapply(1:(length(indices) - 1), function(i){
  tmp2 <- first_tmp[indices[i]:(indices[i + 1] - 1), ] %>% as.matrix()
  rownames(tmp2) <- c("A", "C", "G", "T")
  tmp2
})
first_flat <- sapply(1:length(first), function(i){
  first[[i]] %>% c()
})

second_tmp <- read.delim("data/pwms/layer2.txt", sep = "", header = FALSE)
indices <- seq(1, nrow(second_tmp), 480)
indices <- c(indices, nrow(second_tmp) + 1)
second <- lapply(1:(length(indices) - 1), function(i){
  second_tmp[indices[i]:(indices[i + 1] - 1), ] %>% as.matrix()
})
second_flat <- sapply(1:length(second), function(i){
  second[[i]] %>% c()
})

# looking at the first slice
second_tmp2 <- read.delim("data/pwms/layer2_slice0.txt",
                         sep = "", header = FALSE)
indices <- seq(1, nrow(second_tmp2), 320)
indices <- c(indices, nrow(second_tmp2) + 1)
second_slice0 <- lapply(1:(length(indices) - 1), function(i){
  second_tmp2[indices[i]:(indices[i + 1] - 1), ] %>% as.matrix()
})
second_slice0_flat <- sapply(1:length(second_slice0), function(i){
  second_slice0[[i]] %>% c()
})

# distribution of filters
ggplot(melt(second_flat)) + 
  geom_density(aes(value, group = Var2, color = Var2))

# distribution of max and min
df <- data.frame(maxes = colMaxs(second_flat), 
                 mins = colMins(second_flat),
                 sds = colSds(second_flat),
                 means = colMeans(second_flat),
                 medians = colMedians(second_flat)) %>%
  mutate(diffs = maxes - mins)

p1 <- ggplot(melt(df) %>% 
         filter(variable %in% c("maxes", "mins", "diffs"))) + 
  geom_histogram(aes(value, fill = variable), alpha = 0.5)
p2 <- ggplot(melt(df) %>% 
         filter(variable %in% c("sds", "means", "medians"))) + 
  geom_histogram(aes(value, fill = variable), alpha = 0.5)
ggarrange(p1, p2)

# checking if second layer matrices corresponding to high match filters look 
# different in some way than the others 
ggplot() + 
  geom_histogram(aes(second_flat[, 234]))

# are the filters that have high max values the filters with very strong 
# matches 
high_mean <- second_flat[, which(order(df$means, decreasing = TRUE) < 5)]
low_mean <- second_flat[, which(order(df$means, decreasing = TRUE) > 316)]
ggplot() + 
  geom_histogram(aes(value, group = Var2, fill = "1"), 
                 data = melt(high_mean), bins = 100, alpha = 0.5) + 
  geom_histogram(aes(value, group = Var2 + 5, fill = "2"), 
                 data = melt(low_mean), bins = 100, alpha = 0.5)

# are any of the 1st layer filters co-occuring in the second layer slice 0 
cor_mat <- cor(second_flat) 
clusts <- hclust(1 - as.dist(cor_mat))
ggplot(melt(cor_mat[clusts$order, clusts$order])) + 
  geom_tile(aes(x = Var1, y = Var2, fill = value))

df <- second_flat %>% t()
rownames(df) <- paste0("X", 1:nrow(df))
pcs <- prcomp(df) 
p1 <- ggplot() + 
  geom_point(aes(x = pcs$x[, 1], y = pcs$x[, 2]), alpha = 0.5) + 
  geom_text(aes(x = pcs$x[, 1], y = pcs$x[, 2], label = rownames(df)), 
            hjust = 1, vjust = 1)
ggplot() + 
  geom_point(aes(x = pcs$x[, 1], y = pcs$x[, 2]), alpha = 0.5) + 
  geom_text_repel(aes(x = pcs$x[, 1][pcs$x[, 1] > 0.25 | 
                                       pcs$x[, 1] < -0.25 |  
                                       pcs$x[, 2] < -0.2], 
                      y = pcs$x[, 2][pcs$x[, 1] > 0.25| 
                                       pcs$x[, 1] < -0.25 |  
                                       pcs$x[, 2] < -0.2], 
                      label = rownames(df)[pcs$x[, 1] > 0.25| 
                                             pcs$x[, 1] < -0.25 |  
                                             pcs$x[, 2] < -0.2]), 
            hjust = 1, vjust = 1)

which(pcs$x[, 1] > 0.25)

# looking at filters that have similar matches
tomtom_tbl <- table(tomtom$Query_ID, tomtom$Target_ID)
filter_sim_tomtom <- matrix(rep(0, 320*320), nrow = 320)
for(i in 1:320){
  for(j in 1:i){
    filter_sim_tomtom[i, j] = ifelse(i %in% rownames(tomtom_tbl) & 
                                       j %in% rownames(tomtom_tbl), 
                                     jaccard(tomtom_tbl[as.character(i), ], 
                                             tomtom_tbl[as.character(j), ]), 
                                     0)
    filter_sim_tomtom[j, i]
  }
}
diag(filter_sim_tomtom) <- 1

df2 <- tomtom_tbl %>% as.matrix() 
rownames(df2) <- paste0("X", 1:nrow(df2))
pcs2 <- prcomp(df2) 
p2 <- ggplot() + 
  geom_point(aes(x = pcs2$x[, 1], y = pcs2$x[, 2]), alpha = 0.5) + 
  geom_label_repel(aes(x = pcs2$x[, 1][pcs2$x[, 1] > 0 | pcs2$x[, 2] > 0], 
                 y = pcs2$x[, 2][pcs2$x[, 1] > 0 | pcs2$x[, 2] > 0], 
                 label = rownames(df2)[pcs2$x[, 1] > 0 | pcs2$x[, 2] > 0]), 
            hjust = 1, vjust = 1) 
ggarrange(p1, p2, nrow = 2)


# adding sequence logos to plot
images <- lapply(1:320, function(i){
  readPNG(paste0("logos/filter", i, ".png"))
})

# make a dataframe
files <- list.files("logos/")
df <- data.frame(x = pcs$x[, 1], 
                 y = pcs$x[, 2],
                 image = paste0("logos/", files))
rows <- sample(1:nrow(df), 20)

ggplot(df, aes(x, y)) + geom_image(aes(image=image))

# plot the first layer pcs 
df <- first_flat %>% t()
rownames(df) <- paste0("X", 1:nrow(df))
pcs_first <- prcomp(df) 

df <- data.frame(x = pcs_first$x[, 1], 
                 y = pcs_first$x[, 2],
                 image = paste0("logos/", files))
rows <- sample(1:nrow(df), 20)

ggplot(df, aes(x, y)) + geom_image(aes(image=image))

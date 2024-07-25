library(ggplot2)
library(dplyr)
library(ggtree)
library(ape)

#ol
df1 = read.csv("linker.csv", header=T, sep=",", row.names=1)

dist_mat1 <- dist(df1)
hc1 <- hclust(dist_mat1)
phylo_tree1 <- as.phylo(hc1)

color_map1 <- data.frame(node = c(29, 31, 32),
                         color = c("#1373b2", "#42a6cb", "#77cac5"))

ggtree(phylo_tree1, layout = "circular", yscale = "none") + 
  theme_tree2() +
  geom_nodelab(aes(label=node, fontface="bold.italic"))

t1 <- ggtree(phylo_tree1, layout = "circular", size=1) %<+% color_map1 +
  geom_tree(aes(color = color), size = 1) +  
  theme_tree2() +
  geom_tiplab(hjust = -0.5, size = 6) +
  #geom_nodelab(aes(label = node, color = color)) +  # node tab
  geom_cladelabel(node =29,label = " ",color = "#1373b2",barsize = 2,offset = -0.10) +
  geom_cladelabel(node =31,label = " ",color = "#42a6cb",barsize = 2,offset = -0.10) +
  geom_cladelabel(node =32,label = " ",color = "#77cac5",barsize = 2,offset = -0.10) +
  geom_cladelab(node=28,label="tree 1",textcolor="#1373b2", offset=4) +
  theme(
    axis.title = element_blank(), 
    axis.text = element_blank(),     
    axis.ticks = element_blank(),    
    axis.line = element_blank()) +
  scale_color_identity() 

t1

ggsave(filename = 't1.pdf', width = 8, height = 6)

#sbu
df2 = read.csv("sbu.csv", header=T, sep=",", row.names=1)

dist_mat2 <- dist(df2)
hc2 <- hclust(dist_mat2)
phylo_tree2 <- as.phylo(hc2)

color_map2 <- data.frame(node = c(31, 37, 38, 36, 35),
                         color = c("#084384", "#1373b2", "#42a6cb", "#77cac5", "#b2e1b9"))

t2 <- ggtree(phylo_tree2, layout = "circular", size=1) %<+% color_map2 +
  geom_tree(aes(color = color), size = 1) +  
  theme_tree2() +
  geom_tiplab(hjust = -0.5, size = 6) +
  #geom_nodelab(aes(label = node, color = color)) +  # node tab
  geom_cladelabel(node = 31, label = " ", color = "#084384", barsize = 2, offset = -0.05) +
  geom_cladelabel(node = 37, label = " ", color = "#1373b2", barsize = 2, offset = -0.05) +
  geom_cladelabel(node = 38, label = " ", color = "#42a6cb", barsize = 2, offset = -0.05) +
  geom_cladelabel(node = 36, label = " ", color = "#77cac5", barsize = 2, offset = -0.05) +
  geom_cladelabel(node = 35, label = " ", color = "#b2e1b9", barsize = 2, offset = -0.05) +
  theme(
    axis.title = element_blank(), 
    axis.text = element_blank(),     
    axis.ticks = element_blank(),    
    axis.line = element_blank()) +
  scale_color_identity() 

t2

ggsave(filename = 't2.pdf', width = 8, height = 6)

#topo
df3 = read.csv("topo.csv", header=T, sep=",", row.names=1)

dist_mat3 <- dist(df3)
hc3 <- hclust(dist_mat3)
phylo_tree3 <- as.phylo(hc3)

color_map3 <- data.frame(node = c(22, 29, 28, 27, 26),
                        color = c("#084384", "#1373b2", "#42a6cb", "#77cac5", "#b2e1b9"))

t3 <- ggtree(phylo_tree3, layout = "circular", size=1) %<+% color_map3 +
  geom_tree(aes(color = color), size = 1) +  
  theme_tree2() +
  geom_tiplab(aes(fontface="bold.italic"), hjust = -0.4, size = 7) +
  #geom_nodelab(aes(label = node, color = color)) +  # node tab
  geom_cladelabel(node = 22, label = " ", color = "#084384", barsize = 2, offset = -0.55) +
  geom_cladelabel(node = 29, label = " ", color = "#1373b2", barsize = 2, offset = -0.05) +
  geom_cladelabel(node = 28, label = " ", color = "#42a6cb", barsize = 2, offset = -0.05) +
  geom_cladelabel(node = 27, label = " ", color = "#77cac5", barsize = 2, offset = -0.05) +
  geom_cladelabel(node = 26, label = " ", color = "#b2e1b9", barsize = 2, offset = -0.05) +
  theme(
    axis.title = element_blank(), 
    axis.text = element_blank(),     
    axis.ticks = element_blank(),    
    axis.line = element_blank()) +
  scale_color_identity() 

ggtree(phylo_tree3, layout = "circular", yscale = "none") + 
  theme_tree2() +
  geom_nodelab(aes(label=node, fontface="bold.italic"))

t3

ggsave(filename = 't3.pdf', width = 10, height = 8)

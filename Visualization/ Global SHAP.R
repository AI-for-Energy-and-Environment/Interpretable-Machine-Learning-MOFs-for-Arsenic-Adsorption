library(ggplot2)       #Drawing
library(RColorBrewer)  #Changing color
library(patchwork)
library(ggpubr)
library(dplyr)


getwd()

data_files <- c("env.csv","struc.csv","topo.csv")#

colors <- rev(c("#084384", "#1373b2", "#42a6cb", "#77cac5", "#b2e1b9", "#d6efd0"))

for (i in 1:length(data_files)) {

  data <- read.csv(data_files[i], sep=",", na.strings="NA", stringsAsFactors=FALSE)
  
  if(nrow(data) > 10) { #
    data <- data %>%
      arrange(desc(Value)) %>%
      head(10)
  }
  
  plot <- ggplot(data, aes(x = reorder(Set, Value), y = Value, fill = Value)) +
    geom_bar(stat="identity", position="stack", color="black", width=0.7, size=0.25) +
    scale_fill_gradientn(colours = colors) +  
    xlab("Feature name") + ylab("Mean │ SHAP Value │") + 
    coord_flip() + 
    theme_bw() +
    theme(axis.text.x = element_text(size = 18), 
          axis.text.y = element_text(size = 18), 
          axis.title.x = element_text(size = 20), # , face = "bold"
          axis.title.y = element_text(size = 20),
          legend.text = element_text(size = 18), 
          legend.title = element_text(size = 20)) 
  
  ggsave(paste0("z", i, ".pdf"), plot, width = 6, height = 5)
}


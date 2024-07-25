library(ggplot2)
library(ggExtra)

getwd()

#Data
df1 <- read.csv("DT1results.csv", sep=",", na.strings="NA", stringsAsFactors=FALSE)
df2 <- read.csv("DT2results.csv", sep=",", na.strings="NA", stringsAsFactors=FALSE)
df3 <- read.csv("RF1results.csv", sep=",", na.strings="NA", stringsAsFactors=FALSE)
df4 <- read.csv("RF2results.csv", sep=",", na.strings="NA", stringsAsFactors=FALSE)
df5 <- read.csv("GBDT1results.csv", sep=",", na.strings="NA", stringsAsFactors=FALSE)
df6 <- read.csv("GBDT2results.csv", sep=",", na.strings="NA", stringsAsFactors=FALSE)
df7 <- read.csv("HGB1results.csv", sep=",", na.strings="NA", stringsAsFactors=FALSE)
df8 <- read.csv("HGB2results.csv", sep=",", na.strings="NA", stringsAsFactors=FALSE)
df9 <- read.csv("XGB1results.csv", sep=",", na.strings="NA", stringsAsFactors=FALSE)
df10 <- read.csv("XGB2results.csv", sep=",", na.strings="NA", stringsAsFactors=FALSE)
df11 <- read.csv("CB1results.csv", sep=",", na.strings="NA", stringsAsFactors=FALSE)
df12 <- read.csv("CB2results.csv", sep=",", na.strings="NA", stringsAsFactors=FALSE)
df_list <- list(df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12)

#Fig
for (i in 1:12) {
  df_name <- paste0("df", i)
  df <- get(df_name)
  
  p <- ggplot(df, aes(x = Experimental, y = Predicted, color = TrainTest)) +
    geom_point(size = 5, alpha = 0.8) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
    theme_bw() + theme(legend.position="left") +
    theme(
      axis.text.x = element_text(size = 15),
      axis.text.y = element_text(size = 15),
      axis.title.x = element_text(size = 18),
      axis.title.y = element_text(size = 18),
      legend.title = element_text(size = 18),
      legend.text = element_text(size = 12)
    ) + labs(color = "Data set") + 
    scale_color_manual(values = c("#084384", "#77cac5"))

  p_marginal <- ggMarginal(p, type = "histogram", bins = 30, groupColour = TRUE, groupFill = TRUE)

  plot_name <- paste0("t", i, ".pdf")
  ggsave(filename = plot_name, plot = p_marginal, width = 10, height = 8)
}

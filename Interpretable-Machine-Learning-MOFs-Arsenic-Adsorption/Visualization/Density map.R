library(ggplot2)

getwd()

df <- read.csv("trainingdata.csv", sep=",", na.strings="NA", stringsAsFactors=FALSE)

variables <- c("PLD", "LCD", "ρ", "VSA", "GSA", "VF", "AV")

for (variable in variables) {

  p <- ggplot(df, aes_string(x = variable)) +
    geom_density(fill = "#b2e1b9", color = "#1373b2", alpha = 0.7, size=3.5) +
    labs(y = "Density") +
    theme_bw() +
    theme(
      axis.text.x = element_text(size = 25),
      axis.text.y = element_text(size = 25),
      axis.title.x = element_text(size = 30),
      axis.title.y = element_text(size = 30),
      legend.title = element_text(size = 18),
      legend.text = element_text(size = 12),
      panel.border = element_rect(linetype=1,size=3.5)
    )
  
  pdf_filename <- paste0("Density_Plot_of_", variable, ".pdf")
  
  ggsave(pdf_filename, plot = p, width = 8, height = 8)
}


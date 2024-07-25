library(ggplot2)
library(viridis)
library(fields)
library(scales)

data <- read.csv('5-1.csv')

# Perform thin plate spline interpolation
tps1 <- Tps(data[,c("learning_rate", "n_estimators")], data$RMSE) 

# Create grid
grid_x1 <- seq(min(data$learning_rate) - 0.01, max(data$learning_rate) + 0.01, length = 500)
grid_y1 <- seq(min(data$n_estimators) - 10, max(data$n_estimators) + 10, length = 500)
grid1 <- expand.grid(learning_rate = grid_x1, n_estimators = grid_y1)

# Predict interpolation values
grid1$RMSE <- predict(tps1, grid1)

# Point add
manual_point <- data.frame(learning_rate = 1, n_estimators = 902, RMSE = 10.51461864)

# Visualization
breaks_seq <- seq(min(grid1$RMSE), max(grid1$RMSE), length.out = 5)  

p1 <- ggplot(grid1, aes(x = learning_rate, y = n_estimators, fill = RMSE)) +
  geom_tile() +
  scale_fill_distiller(palette = "GnBu", direction = 1, 
                       breaks = breaks_seq,
                       labels = label_number(accuracy = 0.001)) +
  geom_point(data = manual_point, aes(x = learning_rate, y = n_estimators), color = '#ED8481', size = 5) +
  labs(
    x = 'Learning_rate',
    y = 'N_estimators',
    fill = 'RMSE') +
  theme_bw() +
  theme(
    axis.title.x = element_text(size = 24),
    axis.title.y = element_text(size = 24),
    legend.title = element_text(size = 24, face = "italic"),
    legend.text = element_text(size = 22),
    axis.text.x = element_text(size = 22),
    axis.text.y = element_text(size = 22)
  )

p1

ggsave('p1.pdf', plot = p1, width = 8, height = 6)
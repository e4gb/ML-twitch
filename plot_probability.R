source("settings.R")

library(jsonlite)
library(ggplot2)
library(scales)
library(gridExtra)
library(grid)

# Load prediction report
report <- jsonlite::read_json(
  "prediction_report.json",
  simplifyVector = FALSE
)

# Extract probabilities
probabilities <- unlist(
  report$chunks[[1]]$probabilities,
  use.names = FALSE
)

names(probabilities) <- streamers

# Create plotting data
df <- data.frame(
  streamer = streamers,
  probability = as.numeric(probabilities)
)

df$streamer <- reorder(df$streamer, df$probability)

# Create probability plot
p <- ggplot(df, aes(x = streamer, y = probability, fill = streamer)) +
  geom_col(show.legend = FALSE, width = 0.7) +
  geom_text(
    aes(label = percent(probability, accuracy = 0.01)),
    hjust = -0.15,
    size = 3.5
  ) +
  coord_flip() +
  scale_y_continuous(
    labels = percent,
    limits = c(0, max(df$probability) * 1.2)
  ) +
  labs(
    title = "Twitch Streamer Classification",
    subtitle = sprintf(
      "Predicted: %s | Confidence: %.2f%% | Chunk: %d",
      report$chunks[[1]]$predicted_streamer,
      report$chunks[[1]]$confidence * 100,
      report$chunks[[1]]$chunk
    ),
    x = NULL,
    y = "Predicted probability"
  ) +
  theme_minimal(base_size = 12)

# Model and training settings
settings_text <- paste(
  "MODEL SETTINGS",
  sprintf("Messages per chunk: %d", messages_per_chunk),
  sprintf("Embedding dimension: %d", embedding_dim),
  sprintf("Dense layers: %d / %d", dense_units_1, dense_units_2),
  "",
  "TRAINING SETTINGS",
  sprintf("Epochs: %d", epochs),
  sprintf("Batch size: %d", batch_size),
  sprintf("Learning rate: %g", learning_rate),
  sprintf("Validation split: %.0f%%", validation_split * 100),
  sep = "\n"
)

# Display plot and settings
print(
  grid.arrange(
    p,
    textGrob(
      settings_text,
      x = 0.05,
      just = "left",
      gp = gpar(fontsize = 10, fontfamily = "mono")
    ),
    ncol = 1,
    heights = c(4, 1.5)
  )
)

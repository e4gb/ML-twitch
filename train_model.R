library(keras3)
library(reticulate)
library(jsonlite)

# First set your working directory to where your .R files are: setwd("")
# Check your current working directory with: getwd()

source("settings.R")
source("prepare_training_data.R")



# Install keras if you have not already (once)
"
install_keras(
  backend = 'tensorflow',
  gpu = TRUE
)
"


# ============================================================
# Prepare data to session (~6 GB)
# ============================================================

# Prepare training and validation data.
# save files to to R session

training_data <- prepare_training_data(
  file.path(getwd(), "vectors"),
  streamers,
  messages_per_chunk = messages_per_chunk,
  validation_fraction = validation_split
)

tf <- import("tensorflow")

# Convert both datasets to TensorFlow ragged tensors.
for (dataset_name in c("x", "x_val")) {

  inputs <- training_data[[dataset_name]]

  for (i in seq_along(inputs)) {

    names(inputs[[i]]) <- NULL

    for (j in seq_along(inputs[[i]])) {
      names(inputs[[i]][[j]]) <- NULL
    }

    inputs[[i]] <- tf$ragged$constant(
      inputs[[i]],
      dtype = tf$int32
    )
  }

  training_data[[dataset_name]] <- inputs
}



# ============================================================
# SHARED TRAINABLE EMBEDDING
# ============================================================
#
# This is ONE matrix:
#
# vocab_size × embedding_dim    
#
# (+1 for the "0" for unidentified words)
vocab_size <- nrow(read.csv("lookup.csv")) + 1 
#
# Every one of the 100 messages uses this SAME matrix.
#
# It is trainable and will be updated by backpropagation.

embedding <- layer_embedding(
  input_dim = vocab_size,
  output_dim = embedding_dim,
  name = "word_embedding"
)


# ============================================================
# 100 INPUTS
# ============================================================
#
# Each input is ONE message.
#
# One word is one integer.
#
# Each integer is then turned into a 64 dimensional vector.
#
# The message can contain any number of tokens (any Fvoca of words)
#
# Example:
#
# message 1: [17, 42]
# message 2: [91, 12, 55]
# message 3: [31]
#
# No padding.
# No maximum message length.
#
# ragged = TRUE tells Keras that the token sequence length
# can vary between examples.

inputs <- lapply(seq_len(messages_per_chunk), function(i) {
  keras_input(
    shape = c(NA),
    dtype = "int32",
    ragged = TRUE,
    name = paste0("message_", i)
  )
})


# ============================================================
# PROCESS EACH MESSAGE
# ============================================================
#
# Every message uses the SAME embedding layer.
#
# Each message:
#
# token IDs
# ↓
# embedding lookup
# ↓
# N × 64
# ↓
# average over N
# ↓
# 64
#
# The number of tokens N can be different for every message.

message_vectors <- lapply(inputs, function(x) {
  x |>
    embedding() |>
    layer_global_average_pooling_1d()
})


# ============================================================
# COMBINE THE 100 MESSAGE VECTORS
# ============================================================
#
# Each message is now:
#
# 64
#
# Therefore:
#
# 100 × 64
#
# We concatenate them into:
#
# 6400
#
# The order is preserved:
#
# message 1 → first 64 numbers
# message 2 → next 64 numbers
# ...
# message 100 → final 64 numbers

x <- layer_concatenate(
  message_vectors,
  axis = -1
)


# ============================================================
# NORMAL ANN
# ============================================================

outputs <- x |>
  layer_dense(
    units = dense_units_1,
    activation = "relu",
    name = "hidden_layer_1"
  ) |>
  layer_dense(
    units = dense_units_2,
    activation = "relu",
    name = "hidden_layer_2"
  ) |>
  layer_dense(
    units = num_streamers,
    activation = "softmax",
    name = "classifier"
  )



# ============================================================
# CREATE MODEL
# ============================================================

model <- keras_model(
  inputs = inputs,
  outputs = outputs
)


# ============================================================
# COMPILE
# ============================================================

model |> compile(
  optimizer = optimizer_adam(learning_rate = learning_rate),
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy",
  jit_compile = FALSE
)


# ============================================================
# INSPECT
# ============================================================

model |> summary()


history <- model |> fit(
  x = training_data$x,
  y = training_data$y,
  epochs = epochs,
  batch_size = batch_size,
  validation_data = list(
    training_data$x_val,
    training_data$y_val
  ),
  verbose = 2
)


#get_weights(model)

save_model(model, "twitch_streamer_model.keras")






library(jsonlite)

# This script only has function definitions

source("settings.R")

set.seed(123321)

make_chunks <- function(
    messages,
    chunk_size = messages_per_chunk
) {

  n_complete <- floor(length(messages) / chunk_size)

  if (n_complete == 0) {
    return(list())
  }

  messages <- messages[
    seq_len(n_complete * chunk_size)
  ]

  split(
    messages,
    ceiling(seq_along(messages) / chunk_size)
  )
}


prepare_training_data <- function(
    vectors_directory,
    streamers,
    messages_per_chunk = messages_per_chunk,
    validation_fraction = validation_split
) {

  train_chunks <- list()
  train_labels <- integer()

  val_chunks <- list()
  val_labels <- integer()



  for (streamer in streamers) {

    streamer_directory <- file.path(
      vectors_directory,
      streamer
    )

    session_files <- list.files(
      streamer_directory,
      pattern = "\\.json$",
      full.names = TRUE
    )

    if (length(session_files) < 2) {
      stop(
        "Streamer ", streamer,
        " needs at least two recordings for a train/validation split."
      )
    }

    # Randomly split original recordings.
    session_files <- sample(session_files)

    n_validation <- max(
      1,
      floor(length(session_files) * validation_fraction)
    )

    n_validation <- min(
      n_validation,
      length(session_files) - 1
    )

    validation_files <- session_files[
      seq_len(n_validation)
    ]

    training_files <- session_files[
      -(seq_len(n_validation))
    ]

    label <- match(streamer, streamers) - 1

    # Process training recordings.
    for (file in training_files) {

      messages <- fromJSON(
        file,
        simplifyVector = FALSE
      )

      chunks <- make_chunks(
        messages,
        messages_per_chunk
      )

      train_chunks <- c(train_chunks, chunks)

      train_labels <- c(
        train_labels,
        rep(label, length(chunks))
      )
    }

    # Process validation recordings.
    for (file in validation_files) {

      messages <- fromJSON(
        file,
        simplifyVector = FALSE
      )

      chunks <- make_chunks(
        messages,
        messages_per_chunk
      )

      val_chunks <- c(val_chunks, chunks)

      val_labels <- c(
        val_labels,
        rep(label, length(chunks))
      )
    }

    if (!any(train_labels == label)) stop("No training chunks for: ", streamer)
    if (!any(val_labels == label)) stop("No validation chunks for: ", streamer)

    cat(
      streamer,
      "| Training recordings:", length(training_files),
      "| Validation recordings:", length(validation_files),
      "\n"
    )
  }

  if (length(train_chunks) == 0 || length(val_chunks) == 0) {
    stop("No complete training or validation chunks were produced.")
  }

  # Shuffle training and validation independently.
  train_shuffle <- sample(seq_along(train_chunks))

  train_chunks <- train_chunks[train_shuffle]
  train_labels <- train_labels[train_shuffle]

  val_shuffle <- sample(seq_along(val_chunks))

  val_chunks <- val_chunks[val_shuffle]
  val_labels <- val_labels[val_shuffle]

  # Convert chunks into model inputs.
  make_model_inputs <- function(chunks) {

    lapply(
      seq_len(messages_per_chunk),
      function(message_number) {

        lapply(
          chunks,
          function(chunk) {
            chunk[[message_number]]
          }
        )
      }
    )
  }

  list(
    x = make_model_inputs(train_chunks),
    y = train_labels,
    x_val = make_model_inputs(val_chunks),
    y_val = val_labels
  )
}


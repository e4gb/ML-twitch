library(keras3)
library(reticulate)
library(jsonlite)

# This script only has function definitions

source("settings.R")
source_python("traindata_prepare.py")


tf <- import("tensorflow")

model <- load_model("twitch_streamer_model.keras")


clean_csv_from_twitchchatdownloader.com <- function(csv) {

  df <- read.csv(csv, header = TRUE)
  df <- as.character(df[, 4])

  temp_file <- tempfile(fileext = ".txt")
  valid_file <- tempfile(fileext = ".json")

  writeLines(df, temp_file)

  vectors <- transform_messages_to_vectors(
    temp_file,
    "lookup.csv",
    return_dataset = "true",
    print_unknown_words = TRUE,
    valid_messages_file = valid_file
  )

  messages <- jsonlite::fromJSON(
    valid_file,
    simplifyVector = FALSE
  )

  unlink(c(temp_file, valid_file))

return(list(vectors = vectors,
            messages = messages))
}


#predict_input_chunk <- function(all_messages) {
#  if (length(all_messages) < messages_per_chunk) {
#    stop("Fewer than messages_per_chunk messages supplied. The model requires exactly messages_per_chunk #messages.")
#  }
#  
#  if (length(all_messages) > messages_per_chunk) {
#    message("More than messages_per_chunk messages supplied. Only the first messages_per_chunk messages will #be used.")
#    all_messages <- all_messages[1:messages_per_chunk]
#  }
#  prediction_input <- lapply(all_messages, function(x) {
#    tf$ragged$constant(
#      list(as.list(as.integer(x))),
#      dtype = tf$int32
#    )
#  })
#prediction <- predict(
#  model,
#  prediction_input
#)
#print(paste(
#  seq_along(streamers) - 1,
#  streamers,
#  sep = " = "
#))
#print(prediction)
#}



##################### ChatGPT and Claude solution to some problems (START)
predict_input_chunks <- function(
    valid_data,
    model,
    streamers,
    messages_per_chunk,
    tf,
    output_file = "prediction_report.json",
    verbose = TRUE,
) {

  # ---------------------------------------------------------------------------
  # Validate arguments
  # ---------------------------------------------------------------------------
  if (is.null(valid_data$vectors) || is.null(valid_data$messages)) {
    stop("valid_data must contain `vectors` and `messages`.")
  }

  if (
    !is.numeric(messages_per_chunk) ||
    length(messages_per_chunk) != 1 ||
    is.na(messages_per_chunk) ||
    messages_per_chunk < 1 ||
    messages_per_chunk != as.integer(messages_per_chunk)
  ) {
    stop("messages_per_chunk must be a single positive whole number.")
  }

  messages_per_chunk <- as.integer(messages_per_chunk)

  all_vectors  <- valid_data$vectors
  all_messages <- valid_data$messages

  if (length(all_vectors) != length(all_messages)) {
    stop("Mismatch between message vectors and original messages.")
  }

  # ---------------------------------------------------------------------------
  # Work out chunks
  # ---------------------------------------------------------------------------
  n_chunks <- length(all_vectors) %/% messages_per_chunk

  if (n_chunks == 0) {
    stop("Not enough valid messages for one complete chunk.")
  }

  n_used   <- n_chunks * messages_per_chunk
  leftover <- length(all_vectors) - n_used

  if (leftover > 0) {
    message(
      leftover,
      " valid message(s) will not be used because they do not fill a complete chunk."
    )
  }

  # Rows = message position within a chunk, columns = chunk.
  # Column i holds the consecutive message indices belonging to chunk i.
  index_matrix <- matrix(seq_len(n_used), nrow = messages_per_chunk)

  # ---------------------------------------------------------------------------
  # Build model inputs: one ragged tensor per message position
  # ---------------------------------------------------------------------------
  prediction_input <- lapply(
    seq_len(messages_per_chunk),
    function(message_number) {

      tf$ragged$constant(
        lapply(index_matrix[message_number, ], function(j) {
          as.list(as.integer(all_vectors[[j]]))
        }),
        dtype = tf$int32
      )
    }
  )

  # ---------------------------------------------------------------------------
  # Predict every chunk
  # ---------------------------------------------------------------------------
  predictions <- as.matrix(predict(model, prediction_input))

  if (nrow(predictions) != n_chunks || ncol(predictions) != length(streamers)) {
    stop(
      "Prediction shape (", nrow(predictions), " x ", ncol(predictions),
      ") does not match the number of chunks (", n_chunks,
      ") and streamers (", length(streamers), ")."
    )
  }

  predicted_index <- max.col(predictions, ties.method = "first")

  if (verbose) {
    # Class mapping
    print(paste(
      seq_along(streamers) - 1,
      streamers,
      sep = " = "
    ))

    # Complete probability matrix
    print(predictions)
  }

  # ---------------------------------------------------------------------------
  # One report entry per chunk
  # ---------------------------------------------------------------------------
  chunk_results <- lapply(seq_len(n_chunks), function(i) {

    probabilities <- as.list(as.numeric(predictions[i, ]))
    names(probabilities) <- streamers

    list(
      chunk              = i,
      predicted_streamer = streamers[predicted_index[i]],
      class_id           = predicted_index[i] - 1,
      confidence         = as.numeric(predictions[i, predicted_index[i]]),
      probabilities      = probabilities,
      messages           = lapply(
        index_matrix[, i],
        function(j) all_messages[[j]]
      )
    )
  })

  # Messages that did not fill a complete chunk
  unused_messages <- if (leftover > 0) {
    lapply(
      seq.int(n_used + 1, length(all_messages)),
      function(j) all_messages[[j]]
    )
  } else {
    list()
  }

  # ---------------------------------------------------------------------------
  # Assemble and save report
  # ---------------------------------------------------------------------------
  report <- list(
    messages_per_chunk    = messages_per_chunk,
    total_valid_messages  = length(all_messages),
    total_chunks          = n_chunks,
    chunks                = chunk_results,
    unused_valid_messages = unused_messages
  )

  dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)

  jsonlite::write_json(
    report,
    output_file,
    pretty     = TRUE,
    auto_unbox = TRUE,
    digits     = NA   # keep full precision for probabilities
  )

  # ---------------------------------------------------------------------------
  # Console summary
  # ---------------------------------------------------------------------------
  for (result in chunk_results) {
    cat(
      "Chunk", result$chunk,
      "->", result$predicted_streamer,
      "| Confidence:", round(result$confidence, 4),
      "\n"
    )
  }

  cat("\nSaved ", output_file, "\n", sep = "")

  invisible(report)
}
##################### ChatGPT and Claude solution to some problems (END)



predict_twitch <- function(predict_file = "predict.csv") {

  valid_data <- clean_csv_from_twitchchatdownloader.com(
    predict_file
  )

  predict_input_chunks(
    valid_data = valid_data,
    model = model,
    streamers = streamers,
    messages_per_chunk = messages_per_chunk,
    tf = tf
  )
}



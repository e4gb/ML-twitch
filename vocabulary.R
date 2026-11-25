library(jsonlite)
library(stringr)
library(reticulate)

source("settings.R")
source_python("traindata_prepare.py")

# Initialize streamer messages.
# Required structure (relative to the current working directory):
#
# ./
# └── <streamer_name>/
#     ├── <vod_1>.json
#     └── <vod_2>.json
#
# Each JSON must contain:
# {
#   "messages": [
#     {"message": "nerd?"},
#     {"message": "lmfao"}
#   ]
# }
#
# Additional keys are allowed and ignored. 
# <streamer_name> folders must match the exact streamer names in settings.
#
#
# This script has 3 parts:
# 1. calls create_vocabulary() for EVERY name in "streamers" from settings 
# 2. This script creates "combined_vocabulary.csv" from all .csv in getwd()
# 3. Creates lookup.csv from "combined_vocabulary.csv"
# 4. Creates /vectors folder with the data for training and validation


# Begin script:

# 1.

# START FUNCTION (create_vocabulary)
create_vocabulary <- function(person, X = n_keep) {

# Your JSON file
folder <- file.path(getwd(), person)

# ============================================================
# FIND ALL JSON FILES
# ============================================================

files <- list.files(
  folder,
  pattern = "\\.json$",
  full.names = TRUE
)

cat("Found", length(files), "JSON files\n")


# ============================================================
# PROCESS FILES ONE AT A TIME
# ============================================================

word_counts <- integer()

for (i in seq_along(files)) {

  file <- files[i]

  cat(
    "\rProcessing file",
    i,
    "of",
    length(files),
    ":",
    basename(file)
  )

  # ----------------------------------------------------------
  # Read one JSON file
  # ----------------------------------------------------------

  data <- fromJSON(file)

  # ----------------------------------------------------------
  # Extract chat messages
  # ----------------------------------------------------------

  messages <- data$messages$message

  # ----------------------------------------------------------
  # Convert everything to lowercase
  # Then extract individual words
  # ----------------------------------------------------------

  words <- str_extract_all(
    tolower(messages),
    "\\b[[:alnum:]']+\\b"
  ) |>
    unlist()

  # ----------------------------------------------------------
  # Count words in this file
  # ----------------------------------------------------------

  counts <- table(words)

  # ----------------------------------------------------------
  # Add this file's counts to the total vocabulary
  # ----------------------------------------------------------

  for (word in names(counts)) {

    if (word %in% names(word_counts)) {

      word_counts[word] <-
        word_counts[word] + counts[[word]]

    } else {

      word_counts[word] <-
        counts[[word]]
    }
  }

  # ----------------------------------------------------------
  # Remove temporary objects before reading the next file
  # ----------------------------------------------------------

  rm(data, messages, words, counts)

}

cat("\nFinished processing all files.\n")


# ============================================================
# SORT BY FREQUENCY
# ============================================================

word_counts <- sort(
  word_counts,
  decreasing = TRUE
)


# ============================================================
# KEEP ONLY THE n_keep MOST COMMON WORDS
# ============================================================

vocabulary <- head(
  word_counts,
  n_keep
)


# ============================================================
# CONVERT TO DATA FRAME
# ============================================================

vocabulary <- data.frame(
  word = names(vocabulary),
  frequency = as.integer(vocabulary),
  row.names = NULL
)


# ============================================================
# CREATE FOLDER AND SAVE VOCABULARY
# ============================================================

output_dir <- file.path(getwd(), "vocabulary_frequency")

dir.create(output_dir, showWarnings = FALSE)

output_file <- file.path(
  output_dir,
  paste0("vocabulary_", person, "_", n_keep, ".csv"))

write.csv(
  vocabulary,
  output_file,
  row.names = FALSE
)


# ============================================================
# DONE
# ============================================================

cat(
  "\nSaved the top",
  nrow(vocabulary),
  "words to:\n",
  output_file,
  "\n"
)

}
# END FUNCTION (create_vocabulary)



for (i in streamers) {
  create_vocabulary(i, n_keep)
}


# 2.

library(dplyr)

# Folder containing the CSV files
directory_voc <- file.path(getwd(), "vocabulary_frequency")
directory_save <- file.path(getwd())

  
# Find all CSV files
files <- list.files(
  directory_voc,
  pattern = "\\.csv$",
  full.names = TRUE
)

cat("Found", length(files), "CSV files\n")

# Read all CSVs and combine them
all_words <- lapply(files, function(file) {
  
  cat("Reading:", basename(file), "\n")
  
  read.csv(file, stringsAsFactors = FALSE)
  
}) |>
  bind_rows()

# Add frequencies for identical words
combined_vocabulary <- all_words |>
  group_by(word) |>
  summarise(
    frequency = sum(frequency),
    .groups = "drop"
  ) |>
  arrange(desc(frequency))

# Save result
output_file <- file.path(
  directory_save,
  "combined_vocabulary.csv"
)

write.csv(
  combined_vocabulary,
  output_file,
  row.names = FALSE
)

cat(
  "\nSaved combined vocabulary to:\n",
  output_file,
  "\n"
)


# 3. 

df <- read.csv(
  file.path(getwd(), "combined_vocabulary.csv"),
  stringsAsFactors = FALSE
)

lookup <- data.frame(
  word = df$word,
  integer = seq_len(nrow(df))
)

write.csv(
  lookup,
  file.path(getwd(), "lookup.csv"),
  row.names = FALSE
)


# 4.

extract_messages_from_json(
  'input_data.json',
  'extracted_messages.txt'
)

transform_messages_to_vectors(
  'extracted_messages.txt',
  'lookup.csv',
  'final_dataset.json'
)


# Create files from loose parts in each streamers folder
for (streamer in streamers) {
  create_streamer_vectors(
    streamer,
    getwd(), 
    'lookup.csv', # main working dir word vocab
    'vectors' # folder created
  )
}


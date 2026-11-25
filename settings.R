#messages_per_chunk	Messages per training example
#embedding_dim	    Word representation capacity
#dense_units_1/2	Classifier capacity
#epochs	            Number of training passes
#batch_size	        Examples per gradient update
#learning_rate	    Size of gradient updates
#validation_split	Validation proportion

# ============================================================
# DATA
# ============================================================

messages_per_chunk <- 100
# for the training data, the input in implementation can be any size


# Number of most common words to keep for each streamer's messages in the vocabulary data
n_keep <- 170000

# ============================================================
# MODEL
# ============================================================

embedding_dim <- 64
dense_units_1 <- 256
dense_units_2 <- 128

# ============================================================
# TRAINING
# ============================================================

epochs <- 10
batch_size <- 32
learning_rate <- 0.001
validation_split <- 0.2

# ============================================================
# STREAMERS
# ============================================================

# Order matters for which streamer is associated with which integer. But the choice is arbitrary, just remember it.
streamers <- c(
  "cinna", 
  "dantes",  
  "extraemily", 
  "ishowspeed", 
  "jasontheween",
  "ludwig", 
  "marlon", 
  "moistcr1tikal", 
  "xqc" 
)

num_streamers <- length(streamers)


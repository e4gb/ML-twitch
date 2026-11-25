source(implement.R)

# predict.csv to predict

"
this is the first message chat,
Maybe another here,
lol that is funny,
must be at least 100 lines of messages,
...

Notes:
All messages should belong to the same twitch streamer 
(You are trying to classify)
At least 'messages_per_chunk' messages (settings.R) are needed for prediction
"

if (length(streamers) != model$output_shape[[2]]) {
  stop("Number of streamers does not match model output dimensions.")
}

predict_twitch("predict.csv")

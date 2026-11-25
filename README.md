# ML-twitch
The data for a machine learning project for classification using keras in R for 9 twitch streamers based on the chatters.  

Tokenization ID rules: "lookup.csv" not including 0, where 0 is saved for <unk> (unknown words).

#### Streamers:
  cinna 0,
  dantes 1,
  extraemily 2,
  ishowspeed 3,
  jasontheween 4,
  ludwig 5,
  marlon 6,
  moistcr1tikal 7,
  xqc 8

### If you want to train your own model
Recommended Twitch-chat downloader: 
* "TwitchDownloader" (Copyright (©) lay295).
* Consider [Twitch TOS](https://legal.twitch.com/en/legal/terms-of-service/) 
	* Section 7: "(d) use of any data mining, robots, or similar data gathering or extraction methods; (e) downloading (except page caching) of any portion of the Twitch Services, the Materials, or any information contained in them, except as expressly permitted on the Twitch Services"
	* Section 10: access any website, server, ... by Twitch, ..., by means of any robot, spider, scraper, crawler, or other automated means for any purpose
	* Section 18: Twitch reserves the right ...  to block or prevent your future access to and use of the Twitch Services for any reason, including without limitation if: (a) your use of the Twitch Services violates these Terms of Service ...

##### Needed:
install.packages(c(  
"keras3",  
"reticulate",  
"jsonlite",  
"stringr",  
"dplyr"  
)) 

install_keras(  
backend = 'tensorflow',  

)

##### Construct the following folder structure:
./ 
└──  Files in this repo.

./
└── <streamer_name>/
    ├── <vod_1>.json
    └── <vod_2>.json

* <streamer_name> folders must match the exact streamer names in settings.

Each JSON must contain (where additional keys are allowed in both depths and are ignored):
{
  "messages": [
    {"message": "nerd?"},
    {"message": "lmfao"}
    ...
  ]
}

##### Then do as follows:
0. Set "settings.R" as you wish
1. Run "vocabulary.R" 
2. Run "train_model.R" 
3. Run "predict.R" to use the model

Have fun!

### Disclaimer
* Computational efficiency is not regarded, specifically when running "vocabulary.R". 
	* As a test, running "vocabulary.R" for 9 streamers with very active chatters took exactly 14 minutes for single thread on a high end CPU (November, 2026) for streams per streamer as 10, 21, 14, 15, 9, 15, 15, 15, 14.
	* The training maxed out at around 5.5 GiB of R-memory.
* Messages with all words unknown are discarded. Consequently, prediction can receive fewer messages than expected and stop if fewer than a number of `messages_per_chunk` remain (which you type in "settings.R" before running any script).
* If you train the model with  `messages_per_chunk` = 10, then you need at least 10 messages to use "predict.R". 

### Improvements made
* No identical stream can end up having chunks in both training and validation. (Reduced data leakage)
###### FROM:
* Created chunks
* Shuffle chunks and split 80/20 

####### TO IMPROVEMENT:
Training streams -> Training chunks
Validation streams -> Validation chunks


### Needed improvements
* Word order within each individual message is lost through average pooling
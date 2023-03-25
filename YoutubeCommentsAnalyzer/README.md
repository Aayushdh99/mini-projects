# YouTube Comments Sentiment Analyzer
  
## Details:
The goal of this project is to create a sentiment analysis app for YouTube comments. The app will use Huggingface model `distilroberta-base`, fine-tuned on `youtube-statistics` dataset to predict the sentiment of each comment. The app will display the percentage and count of positive, negative, and neutral comments for a specific YouTube video.

The dataset used for this project is from Kaggle and includes video IDs, comments, and their associated sentiment. Before training the Huggingface models, we performed data cleaning and kept only comments in English. We then used a wrapper library built on top of the YouTube API to extract comments and like/dislike numbers for a specific video URL.

To provide an overview of the sentiment analysis results, we generated charts representing the overall status of the comments. The app will make use of these charts to give users an idea of the general sentiment of the comment section. With this app, users can quickly determine the overall sentiment of a video's comments without reading through every single one.
	
Dataset Used: https://www.kaggle.com/datasets/advaypatil/youtube-statistics

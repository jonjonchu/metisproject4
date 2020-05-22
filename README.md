# Reddit Topics

The website [reddit](https://www.reddit.com) works as a great news aggregator, with users submitting posts to specific subreddits - but the site is a RAM hog and it can take some time to scroll through enough posts to get an idea of what is currently happening.

Enter [reddit-topics](reddit-topics.herokuapp.com), the news aggregator-aggregator that scrapes the current state of a subreddit and performs NLP to extract topics. reddit-topics includes built-in visualizations courtesy of Bokeh.

## Data and Methods

Data is acquired through the Reddit API and its associated Python wrapper PRAW. Submission titles, urls, dates created, upvote score, upvote ratio, and number of comments are saved.

Next, the raw title data for each post is fed into a custom NLP pipeline (found in ```nlp_helper_funcs.py```) that cleans and vectorizes the titles into 2- and 3-word grams (terms) using TF-IDF vectorization. The resulting vectorized document-term matrix is then passed through SVD to extract topics.

## Visualizations

The Bokeh library was used to create inline, interactive plots for each set of scraped data.

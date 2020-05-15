from flask import Flask, request, render_template
import praw

import nlp_helper_funcs

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.decomposition import TruncatedSVD

# Initialize NLP systems
vectorizer = CountVectorizer(ngram_range=(
    1, 3), stop_words=stopwords.words('english'))
tokenizer = TreebankWordTokenizer().tokenize
stemmer = PorterStemmer()


#%% Initialize the app
app = Flask(__name__)

# Homepage
@app.route("/")
def home():
    subreddit="worldnews"
    postType="hot"
    topics = extract_topics(subreddit, postType)
    return (render_template('home.html',
                            subreddit=subreddit,
                            post_type=postType,
                            topic1=topics[1],
                            topic2=topics[2],
                            topic3=topics[3],
                            topic4=topics[4],
                            topic5=topics[5],
                            topic6=topics[6],
                            topic7=topics[7],
                            # topic8=topics[8],
                            # topic9=topics[9],
                            # topic10=topics[10],
                            ))

def display_topics(model, feature_names, num_top_words, topic_names=None):
    topics = {}
    for ix, topic in enumerate(model.components_):
        topics[ix+1] = ", ".join([feature_names[i]
                         for i in topic.argsort()[:-num_top_words - 1:-1]])
    return topics

def scrape_titles(subreddit, post_type, limit=100):
    '''
    Scrapes the titles of posts in subreddit, 
    where post_type=[hot,new,top,rising,controversial,gilded]
    and limit is the number of posts to scrape
    
    Returns a post titles as a list of strings
    '''
    reddit = praw.Reddit("bot1")

    if post_type == 'hot':
        posts = reddit.subreddit(subreddit).hot(limit=limit)
    elif post_type == 'new':
        posts = reddit.subreddit(subreddit).new(limit=limit)
    elif post_type == 'top':
        posts = reddit.subreddit(subreddit).top(limit=limit)
    elif post_type == 'rising':
        posts = reddit.subreddit(subreddit).rising(limit=limit)
    elif post_type == 'controversial':
        posts = reddit.subreddit(subreddit).controversial(limit=limit)
    elif post_type == 'gilded':
        posts = reddit.subreddit(subreddit).gilded(limit=limit)
    titles = []
    for post in posts:
        titles.append(post.title)
    return titles

def extract_topics(subreddit, post_type):
    '''
    NLP
    scrapes reddit, performs nlpreproc, extracts topics
    returns dict of topics
    '''
    num_topics = 7
    num_grams_per_topic = 10

    titles = scrape_titles(subreddit, post_type)

    nlp = nlp_helper_funcs.nlp_preprocessor(vectorizer=vectorizer, cleaning_function=None,
                                            tokenizer=tokenizer, stemmer=None)
    nlp.fit(titles)
    doc_word = nlp.transform(titles).toarray()

    lsa = TruncatedSVD(num_topics)
    doc_topic = lsa.fit_transform(doc_word)

    topics = display_topics(
        lsa, vectorizer.get_feature_names(), num_grams_per_topic)

    return topics

@app.route("/f", methods=["GET","POST"])
def predict():
    query = request.args.to_dict()
    if query['subreddit'] is '':
        query['subreddit'] = 'all'
    topics = extract_topics(query['subreddit'], query['postType'])
    return (render_template('home.html',
                            subreddit=query['subreddit'],
                            post_type=query['postType'],
                            topic1=topics[1],
                            topic2=topics[2],
                            topic3=topics[3],
                            topic4=topics[4],
                            topic5=topics[5],
                            topic6=topics[6],
                            topic7=topics[7],
                            # topic8=topics[8],
                            # topic9=topics[9],
                            # topic10=topics[10],
                            ))


#%%--------- RUN WEB APP SERVER ------------#
# Start the app server on port XXXX
# (The default website port)
if __name__ == '__main__':
    app.run(debug=True)
    # debug=True

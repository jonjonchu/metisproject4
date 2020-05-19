from flask import Flask, request, render_template
import praw
from collections import deque

import nlp_helper_funcs

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.decomposition import TruncatedSVD

# Initialize NLP systems
vectorizer = TfidfVectorizer(ngram_range=(
    2, 3), stop_words=stopwords.words('english'))
tokenizer = TreebankWordTokenizer().tokenize
stemmer = PorterStemmer()


#%% Initialize the app
app = Flask(__name__)

def display_topics(model, feature_names, no_top_words, topic_names=None, verbose=False):
    topics = {}
    for ix, topic in enumerate(model.components_):
        gram_list = [feature_names[i].split()
                     for i in topic.argsort()[:-no_top_words - 1:-1]]

        #instantiate variables
        deques = {}
        for i in range(0, len(gram_list)):
            deques[i] = deque(gram_list[i])

        final_text = []
        clauses = []

        if verbose:
            print("DEQUES:", deques)

        while len(deques) > 0:
            # initialize clause as first deque in dictionary
            clauses.append(deques[list(deques)[0]])
            del deques[list(deques)[0]]

            for clause in clauses:
                while len(deques) > 0:
                    for i in deques.copy():
                        gram = deques[i]
                        overlap = False
                        # check for overlapping words and append
                        for word in gram:
                            if word in clause:
                                overlap = True

                            elif word not in clause and overlap == True:
                                clause.append(word)
                        overlap = False
                        # reverse words in gram to prepend
                        for word in deque(reversed(gram)):
                            if word in clause:
                                overlap = True

                            elif word not in clause and overlap == True:
                                clause.appendleft(word)

                        if overlap == True:
                            del deques[i]
                        if verbose:
                            print("OVERLAP=", overlap, "for gram", i)
                    if verbose:
                        print("CLAUSE:", clauses)
                        print("DEQUES:", deques)
                        print('end of while loop')
                    if overlap == False:
                        break
            #convert clause from deque to list of strings
            final_text.append(list(clause))

            topics[ix+1] = final_text
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
    titles = []
    for post in posts:
        titles.append(post.title)
    return titles

def extract_topics(subreddit, post_type, limit):
    '''
    NLP
    scrapes reddit, performs nlpreproc, extracts topics
    returns dict of topics
    '''
    num_topics = 10
    num_grams_per_topic = 10

    titles = scrape_titles(subreddit, post_type, limit)

    nlp = nlp_helper_funcs.nlp_preprocessor(vectorizer=vectorizer, cleaning_function=None,
                                            tokenizer=tokenizer, stemmer=None)
    nlp.fit(titles)
    doc_word = nlp.transform(titles).toarray()

    lsa = TruncatedSVD(num_topics)
    doc_topic = lsa.fit_transform(doc_word)

    topics = display_topics(
        lsa, vectorizer.get_feature_names(), num_grams_per_topic)

    return topics

def readable_topics(topic):
    readables = []
    for i in range(0,len(topic)):
        sentence = " ".join(topic[i])
        readables.append(sentence)
    readables = ", ".join(readables)
    return readables

def google_topics(topic):
    search_query = []
    for i in range(0,len(topic)):
        query_part = "+".join(topic[i])
        search_query.append(query_part)
    search_query = "+".join(search_query)
    return search_query

@app.route("/", methods=["GET","POST"])
def predict():
    query = request.args.to_dict()
    if query == {}:
            query['subreddit'] = "worldnews"
            query['postType'] = "hot"
            query['limit'] = 100
    else:
        if query['subreddit'] is '':
            query['subreddit'] = 'all'
    topics = extract_topics(query['subreddit'], query['postType'], int(query['limit']))
    return (render_template('home.html',
                            subreddit=query['subreddit'],
                            post_type=query['postType'],
                            topic1=readable_topics(topics[1]),
                            g_query1=google_topics(topics[1]),
                            topic2=readable_topics(topics[2]),
                            g_query2=google_topics(topics[2]),
                            topic3=readable_topics(topics[3]),
                            g_query3=google_topics(topics[3]),
                            topic4=readable_topics(topics[4]),
                            g_query4=google_topics(topics[4]),
                            topic5=readable_topics(topics[5]),
                            g_query5=google_topics(topics[5]),
                            topic6=readable_topics(topics[6]),
                            g_query6=google_topics(topics[6]),
                            topic7=readable_topics(topics[7]),
                            g_query7=google_topics(topics[7]),
                            limit=query['limit']
                            ))


#%%--------- RUN WEB APP SERVER ------------#
# Start the app server on port XXXX
# (The default website port)
if __name__ == '__main__':
    app.run()
    # debug=True

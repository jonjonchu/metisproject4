from flask import Flask, request, render_template

# Scraping
import praw
# Data Collection/Cleaning for visualization
import pandas as pd
from numpy import arange
from datetime import datetime
from collections import deque

# NLPre-Processing
# import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TreebankWordTokenizer
# from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import nlp_helper_funcs

# Topic Extraction using SVD
from sklearn.decomposition import TruncatedSVD

# Visualization with Bokeh
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import HoverTool, Panel
from bokeh.models.widgets import Tabs
import bokeh.palettes

# path_to_spacy = './data/en_core_web_sm/en_core_web_sm-2.2.5'
# nlp = spacy.load(path_to_spacy, disable=["parser", "ner", "textcat"])

# Initialize NLP systems
vectorizer = TfidfVectorizer(ngram_range=(
    2, 3), stop_words=stopwords.words('english'))
tokenizer = TreebankWordTokenizer().tokenize
# stemmer = PorterStemmer()



#%% Initialize the app
app = Flask(__name__)

def display_topics(model, feature_names, no_top_words, topic_names=None, verbose=False):
    '''
    Gets most significant grams from each topic and tries to string them together 
    into coherent sentences if words in the grams overlap.

    Time complexity O(n!): thank god we're only taking the top 10 words
    '''
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

def readable_topics(topic):
    '''
    Parses a list of lists into clauses separated by commas.
    Returns a string i.e. [['hello','world','hello'], ['my','name']]
    becomes 'hello world hello, my name'
    Time Complexity O(n): n = number of all elems in 'topic'
    '''
    readables = []
    for i in range(0, len(topic)):
        sentence = " ".join(topic[i])
        readables.append(sentence)
    readables = ", ".join(readables)
    # Capitalize first character of sentence
    return readables[0].upper() + readables[1:]

def google_topics(topic):
    '''
    Takes in a list of list of words and concatenates them with '+'.
    Used as a GET query for google.com
    Returns a string i.e. [['hello','world','hello'], ['my','name']]
    becomes 'hello+world+hello+my+name'
    Time Complexity O(n): n = number of all elems in 'topic'
    '''
    search_query = []
    for i in range(0, len(topic)):
        query_part = "+".join(topic[i])
        search_query.append(query_part)
    search_query = "+".join(search_query)
    return search_query

def scrape_reddit(subreddit, post_type, limit=100):
    '''
    Scrapes the titles of posts in subreddit, 
    where post_type=[hot,new,top,rising,controversial,gilded]
    and limit is the number of posts to scrape
    
    Returns a post titles as a list of strings
    Time Complexity O(n): n = number of posts
    '''
    reddit = praw.Reddit("bot1")

    if post_type == 'hot':
        raw_posts = reddit.subreddit(subreddit).hot(limit=limit)
    elif post_type == 'new':
        raw_posts = reddit.subreddit(subreddit).new(limit=limit)
    elif post_type == 'top':
        raw_posts = reddit.subreddit(subreddit).top(limit=limit)
    elif post_type == 'rising':
        raw_posts = reddit.subreddit(subreddit).rising(limit=limit)
    elif post_type == 'controversial':
        raw_posts = reddit.subreddit(subreddit).controversial(limit=limit)
    posts = { "title":[], "url":[], "id":[], "created_utc":[], 
            "num_comments": [], "upvote_ratio": [], "score":[], "edited":[]
            }
    for post in raw_posts:
        posts["title"].append(post.title)
        posts['url'].append(post.url)
        posts["id"].append(post.id)
        posts["created_utc"].append(post.created_utc)
        posts["num_comments"].append(post.num_comments)
        posts["upvote_ratio"].append(post.upvote_ratio)
        posts["score"].append(post.score)
        posts["edited"].append(post.edited)
    return pd.DataFrame(posts)

def spacy_preproc(titles):
    # don't forget to select only the properties we require
    spacy_words = list(nlp.pipe(titles))
    cleaned_titles = []
    for doc in spacy_words:
        cleaned_title = []
        for token in doc:
            # add token.lemma_ if it's not punctuation or a stopword
            if token.pos_ != "PUNCT":
                if token.is_stop is False:
                    cleaned_title.append(token.lemma_)
        cleaned_title = " ".join(cleaned_title)
        cleaned_titles.append(cleaned_title)
    return cleaned_titles

def bokeh_plot(source, x, y, labels:tuple, topics:list, num_topics=10):

    # simple plot
    simpleScatterPlot = figure(
        plot_width=800, plot_height=400, sizing_mode="scale_both")

    topic_num = [str(i) for i in arange(0, num_topics)]

    source['color'] = [bokeh.palettes.d3['Category10'][10]
                       [int(source['topic'].loc[i])] for i in source.index]

    for topic in topic_num:
        simpleScatterPlot.circle(x, y, source=source[source.topic == topic], size=10,
                                 legend_label=topics[int(topic)],
                                 color='color',
                                 alpha=0.7)
    # Set background colors to match page
    simpleScatterPlot.background_fill_color = "#EFF7FF"
    simpleScatterPlot.border_fill_color = "#EFF7FF"
    # Set labels
    simpleScatterPlot.xaxis.axis_label = labels[0]
    simpleScatterPlot.yaxis.axis_label = labels[1]
    # Set legend
    simpleScatterPlot.legend.location = "top_left"
    simpleScatterPlot.legend.click_policy = "hide"

    tooltips = [
        ("", "@title"),
        ("Date", "@Datetime"),
        ("URL", "@url"),
        ("Topic", "@topic_words")
    ]

    hover = HoverTool(tooltips=tooltips)
    simpleScatterPlot.add_tools(hover)

    return simpleScatterPlot

def bokeh_viz(source, topics:list, num_topics=10):
    # import tabs, panel from bokeh
    # create a tab for each plot i want
    tab = []
    plots = {
        'x': ['upvote_ratio', 'upvote_ratio', 'created_utc', ],
        'y' : ['num_comments', 'score','score', ],
        'bokeh_labels' : [('Upvote Ratio', 'Number of Comments'),
                          ('Upvote Ratio', 'Upvotes (Score)'),
                          ('Date Submitted (POSIX Epoch)', 'Upvotes (Score)'),
                         ],
        'title' : ['#comments v upvote ratio', 'upvote ratio to score', 'score v date submitted'],
        'plot': [],
    }
    # create tabs for each plot defined above
    for i in range(0, len(plots['title'])):
        plots['plot'].append( bokeh_plot(source, plots['x'][i], plots['y'][i], labels=plots['bokeh_labels'][i],
                topics=topics, num_topics=num_topics) )

        tab.append(Panel(child=plots['plot'][i], title=plots['title'][i]))

    tabs = Tabs(tabs=tab)
    return components(tabs)

def extract_topics(subreddit, post_type, limit, num_topics=10, num_grams_per_topic=10):
    '''
    NLP
    scrapes reddit, performs nlpreproc, extracts topics
    returns dict of topics
    '''
    
    df_posts = scrape_reddit(subreddit, post_type, limit)
    df_posts['Datetime'] = pd.to_datetime(df_posts.created_utc, unit='s')
    df_posts['Datetime'] = df_posts['Datetime'].apply(
        lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

    titles = [title for title in df_posts['title']]

    # Old nlp pipeline
    nlp = nlp_helper_funcs.nlp_preprocessor(vectorizer=vectorizer, cleaning_function=None,
                                            tokenizer=tokenizer, stemmer=None)
    nlp.fit(titles)
    doc_word = nlp.transform(titles).toarray()

    # Use SpaCy to clean the text
    # cleaned_titles = spacy_preproc(titles)
    # run cleaned text through vectorizer
    # vectorizer.fit(cleaned_titles)
    # doc_word = vectorizer.transform(cleaned_titles).toarray()

    # initialize dimension reduction tool and reduce dims
    lsa = TruncatedSVD(num_topics)
    doc_topic = lsa.fit_transform(doc_word)

    # parse returned topics
    topics = display_topics(
        lsa, vectorizer.get_feature_names(), num_grams_per_topic)
    readable_topics_list = [readable_topics(topics[i]) for i in topics.keys()]

    # Add topics to df_posts
    df_topics = pd.DataFrame(doc_topic, columns=[str(i) for i in arange(0, num_topics)])
    df_topics['topic'] = df_topics.idxmax(axis=1)
    df_topics['topic_words'] = [ readable_topics_list[int(df_topics['topic'].loc[i])] for i in df_topics.index ]
    df_posts = df_posts.join(df_topics)


    return topics, df_posts

@app.route("/", methods=["GET","POST"])
def predict():
    num_topics=10
    query = request.args.to_dict()
    if query == {}:
            query['subreddit'] = "worldnews"
            query['postType'] = "hot"
            query['limit'] = 100
    else:
        if query['subreddit'] is '':
            query['subreddit'] = 'news+worldnews+upliftingnews+truenews'
    topics, df_posts = extract_topics(query['subreddit'], query['postType'], int(query['limit']))
    
    readable_topics_list = [readable_topics(topics[i]) for i in topics.keys()]
    google_topic_queries = [google_topics(topics[i]) for i in topics.keys()]

    bokeh_js1, bokeh_div1 = bokeh_viz(df_posts, topics= readable_topics_list, num_topics=num_topics)
    
    return (render_template('home.html',
                            subreddit=query['subreddit'],
                            post_type=query['postType'],
                            topic1=readable_topics_list[0],
                            g_query1=google_topic_queries[0],
                            topic2=readable_topics_list[1],
                            g_query2=google_topic_queries[1],
                            topic3=readable_topics_list[2],
                            g_query3=google_topic_queries[2],
                            topic4=readable_topics_list[3],
                            g_query4=google_topic_queries[3],
                            topic5=readable_topics_list[4],
                            g_query5=google_topic_queries[4],
                            topic6=readable_topics_list[5],
                            g_query6=google_topic_queries[5],
                            topic7=readable_topics_list[6],
                            g_query7=google_topic_queries[6],
                            topic8=readable_topics_list[7],
                            g_query8=google_topic_queries[7],
                            topic9=readable_topics_list[8],
                            g_query9=google_topic_queries[8],
                            topic10=readable_topics_list[9],
                            g_query10=google_topic_queries[9],
                            limit=query['limit'],
                            bokeh_div1=bokeh_div1,
                            bokeh_js1=bokeh_js1,
                            ))

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html")

@app.errorhandler(500)
def not_found(e):
    return render_template("500.html")

#%%--------- RUN WEB APP SERVER ------------#
# Start the app server on port XXXX
# (The default website port)
if __name__ == '__main__':
    app.run(debug=False)
    # debug=True

#!/usr/bin/env python
# coding: utf-8

# In[2]:


# System and file libraries & objects.
import os
import time
import datetime
from io import BytesIO
from PIL import Image
# Numpy and Pandas libraries.
import numpy as np
import pandas as pd
# NLP and preprocessing libraries and resources.
import re
import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer 
import emoji
import string
# Wordcloud and other related resources.
from wordcloud import WordCloud, STOPWORDS
import base64
# Tensorflow resources for loading model and doing pad sequencing..
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
# Sklearn scaling used in the emotion intensity calculations.
from sklearn.preprocessing import MinMaxScaler
# Plotly and Dash libraries, components and other resources.
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, State, Input
import plotly.express as px
# Tweepy and json resources.
import tweepy
import json


# In[3]:


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# In[4]:


# Tweepy related variables
consumer_key = "1KajOA3x2090GAGEdxtCNjr8F"
consumer_secret = "nRCHFtDFNabvI8uqAzo8jrCfTVgkwW56Tgh4VuwzMj3MERLMt5"
access_token = "548448748-6QCWBbyOo2RkBl3LnxfPKhZMFnGCe6MlfY3H1umg"
access_token_secret = "Pbc8FSuG2RCeoetZV9lNyBHZq1hJ7gWjxnXb4JVsGuhRx"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# In[5]:


# Main tweet fetching function; it is efficient due to working with vars rather than using the tweepy.Cursor object.
def extract_timeline_as_df(timeline_list):
    columns = set()
    allowed_types = [str, int, datetime.datetime]
    tweets_data = []
    for status in timeline_list:
        status_dict = dict(vars(status))
        keys = status_dict.keys()
        single_tweet_data = {"user": status.user.screen_name, "author": status.author.screen_name}
        for k in keys:
            try:
                v_type = type(status_dict[k])
            except:
                v_type = None
            if v_type != None:
                if v_type in allowed_types:
                    single_tweet_data[k] = status_dict[k]
                    columns.add(k)
        tweets_data.append(single_tweet_data)


    header_cols = list(columns)
    header_cols.append("user")
    header_cols.append('author')
    df = pd.DataFrame(tweets_data, columns=header_cols)
    return df


# In[6]:


# Preprocessing function; it takes a DataFrame and returns a cleaned, lemmatized, tokenized, demojized copy. It also drops empty lines and returns a list of the dropped indexes.
lemma = nltk.WordNetLemmatizer()  
def prepare(dataframe):
    df = dataframe.copy()
    # extract hashtags
    df["text"]=df["text"].apply(lambda x: re.sub(r"#",' ',x))
    # translate emojis
    df["text"]=df["text"].apply(lambda x: emoji.demojize(x))
    # remove urls
    df["text"]=df["text"].apply(lambda x: re.sub(r'https?:/(/[\w|.]*)+',' ',x))
    # remove @'s
    df["text"]=df["text"].apply(lambda x: re.sub(r'@\w+',' ',x))
    # blankspace
    df["text"]=df["text"].apply(lambda x: re.sub(r'\s+',' ',x))
    # lemmatize
    df["text"]=df["text"].apply(lambda x: lemma.lemmatize(str(x).lower()))
    # tokenize
    df["text"]=df["text"].apply(lambda x: nltk.word_tokenize(str(x).lower()))
 
    # remove stopwords
    df["text"]=df["text"].apply(lambda x: [y for y in x if (y not in stopwords.words('english'))])
    # remove punctuation
    df["text"]=df["text"].apply(lambda x: [re.sub(r'['+string.punctuation+']','',y) for y in x])
    # remove breaks
    df["text"]=df["text"].apply(lambda x: [re.sub('\n','',y) for y in x])
    # remove weird, small words
    df["text"]=df["text"].apply(lambda x: [y for y in x if len(y) > 2])
    
    lis_deleted=[]
    for i in range(len(df)):
        if len(df['text'][i])<1:
            df=df.drop(i)
            lis_deleted.append(i)
    df = df.reset_index(drop=True)
    return df, lis_deleted


# In[7]:


# Input encoding function, the same we used for training the model. It is necessary in order to predict emotions for tweet data. Returns a Series with each document encoded.
def encode(dataframe):
  # corpus_words is a list comprised of lists of words.  
  corpus_words = []
  for i in range(len(dataframe)):
    corpus_words.append(dataframe.text[i]) 

  # all_words is a list of all the words in all of corpus_words lists
  allwords = []
  for i in range(len(corpus_words)):
    allwords += corpus_words[i] # All words, from all documents.

  # ordered_words has an index for every word in allwords, ordered by relative frequency.
  ordered_words = pd.Series(allwords).value_counts().index # All words, ordered by increasing relative frequency.

  # dict_words has every index from ordered_words ?????
  dict_words = {}
  for i in range(len(ordered_words)): # Using ordered_words 
    dict_words[ordered_words[i]] = i+1

  # Series of a list that contains the code for each word in each document in the data.
  lis = []
  for i in np.arange(len(dataframe.text)):
      lis.append(pd.Series(dataframe.text[i], dtype = str).apply(lambda x: dict_words[x]))
  return pd.Series(lis)


# In[8]:


# Function which returns a Series with a sum of the scores (for each emotion in our emotion intensity lexicon) for an entire collection of tweets. Input and output are DataFrames.
emotions_df = pd.read_csv ('data/emotion-intensity.csv')
Scaler = MinMaxScaler()
def emotion_scores (dataframe):
    df = dataframe.copy() 
    aux = nltk.word_tokenize(str(df.Text))# Text debe tener la T mayúscula
    words = emotions_df.index.values
    emotions = emotions_df.columns

    for i in range(len(aux)):
        aux[i] = lemma.lemmatize(aux[i]) 
  
    w = np.zeros(7,)
    for i in aux:
        if i in words:
            p = emotions_df.index.get_loc(i)
        else: continue
  
        w = np.add(list(emotions_df.iloc[p].values), w)
    c = pd.Series(w, index = emotions)
    Scaler.fit_transform(np.array(c).reshape(-1,1))
    df_scores = pd.DataFrame(index=c.index, data=c, columns=["score"]).rename_axis('emotion').reset_index()
    return df_scores


# In[9]:


# This function returns the y_pred indexes (0-5) from a model.predict(x) probabilities for each emotion index.
def prediction(predicted):
  index = []
  for i in range(len(predicted)):
    max_p = max(predicted[i]) # highest value in each row of probabilities (one for each emotion).
    index.append(list(predicted[i]).index(max_p)) # we append that max probability emotion to a list and return that.
  return index

# This function returns the y_pred emotion labels from the indexes given by prediction(model.predict(x))
def prediction_labels(prediction):
  emotions_dict = {k:v for (k,v) in zip(range(0,6), ['anger', 'fear', 'joy', 'love', 'sadness','surprise'])}
  l = []

  for i in range(len(prediction)):
        dict_key = prediction[i] # the index key for the dictionary
        l.append(emotions_dict[dict_key])

  return np.array(l)


# In[10]:


# We want to load our model so that we can make predictions for the histogram plot.
model = load_model('model/2021-06-30_0.833.h5')


# In[11]:


# Function which classifies a dataframe using model predictions. Uses deleted_index to account for correct indexing of results (some lines are always dropped due to being empty).
def classify_data(dataframe):
    df = dataframe.copy()
    maxlen = np.max([len(a) for a in df['text']])

    df1, deleted_index = prepare(df) # we save the deleted indexes from prepare() for later
    df2 = encode(df1)
    df_predict = sequence.pad_sequences(df2, maxlen, padding='post')

    y_pred = prediction(model.predict(df_predict))
    labels = prediction_labels(y_pred)

    for i in range(len(deleted_index)):
        df = df.drop(i)
    df = df.reset_index(drop=True)
    df['Text'] = df['text'] #Tengo que ponerlo aquí en mayúsculas para que quede bonito en el datatable
    df['Emotion'] = labels
    
    return df


# In[12]:


# Function used to create the WordCloud object used to plot the actual image later
stopwords1=set(STOPWORDS)
def plot_wordcloud(data):
    d = {a: x for a, x in data.values}
    mask=np.array(Image.open('assets/mask.png'))
    wc = WordCloud(background_color='white', width=720, height=480, font_path='assets/LandasansUltraLight-qZ080.otf', stopwords=stopwords1, mask=mask)
    wc.fit_words(d)
    return wc.to_image()


# In[13]:


# This function creates a dataframe with all words and their frequencies from another dataframe. It is used to prepare the wordcloud image.
def wordfreqs_dict(dataframe):
    df = dataframe.copy()
    df, x = prepare(df)
    
    # corpus_words is a list comprised of lists of words.  
    corpus_words = []
    for i in range(len(df)):
        corpus_words.append(df['text'][i]) 

    # all_words is a list of all the words in all of corpus_words lists
    allwords = []
    for i in range(len(corpus_words)):
        allwords += corpus_words[i] # All words, from all documents.

    wrds = []
    freqs = []
    for i in allwords:
        wrds.append(i)
        freqs.append(allwords.count(i))
    wordfreqs_dict = {'word':wrds,'freq':freqs}
    
    return pd.DataFrame(wordfreqs_dict)


# In[20]:


# This function creates groups of the data to plot them in a multi-line chart
def line_chart(data, number):
    new_row = {}
    df_chart = pd.DataFrame()
    n = number
    grsz = int(len(data)/n)
    for i in range(1,n):
        q = grsz*i
        qi = (i-1)*grsz
        df_group = data[qi:q] # Group range referred to data Dataframe
        val = df_group['Emotion'].value_counts() # Pandas series with the values
        values = pd.DataFrame(columns = ['sadness','anger','surprise','fear','joy','love'])
        values = values.append(val.to_dict(),ignore_index=True) # Dataframe conversion
        new_row = {# Row of the data of the group
            'Date':data['Date'][qi],
            'sadness':values['sadness'][0],
            'anger':values['anger'][0],
            'surprise':values['surprise'][0],
            'fear':values['fear'][0],
            'joy':values['joy'][0],
            'love':values['love'][0],
              }
        df_chart = df_chart.append(new_row, ignore_index = True)
    df_chart2 = df_chart.fillna(0) # Fill de NaN values
    df_chart3 = df_chart2.drop(columns=['Date']).astype('int64') # Converting to int values
    df_chart3['Date'] = df_chart['Date'] # Take de good Date column
    
    return df_chart3


# In[ ]:


# Start the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, prevent_initial_callbacks=True)
app.title = "Twitter Emotion Analyzer"

colors = {
    'background': '#E5E5E5',
    'text': '#7FDBFF'
}

# HTML structure layout

app.layout = html.Div(
    className="body",
    style={"backgroundColor": colors["background"],"backgroundImage":"url('assets/image201.png')", "margin": 0, "padding": "16px","height":"100%"},
    children=[
        html.Header(className="header", style={"border-radius":"25px","font-size":"24px","margin-bottom":"25px","padding":"28px","text-align":"left"}, children=[
            html.H1(children=["🐦 Twitter Emotion Analyzer 💙"], style={"text-align":"center"}),
            html.P(
                "This web app fetches ~200 tweets live from a given timeline and classifies this data into 6 emotions using a neural network."
                ),
            html.P(
                "The initial design concept was a tool that would be helpful in tracking sadness or self-harm related messages in Twitter."
                ),
            html.Hr(style={"border-top": "3px dashed white"}),
            dcc.Markdown('''    
### 💡 About this project

This website is hosted by a local server running a [Dash](https://dash.plotly.com/introduction) app. Inside of the app we load a Keras model and define several useful functions, which are employed to generate the visualizations below.

There are 4 different visualizations visible below:
* A histogram graph showing the count for every emotion label predicted by the model.
* A wordcloud representing the frequency of each word in the total amount of data collected .
* A pie chart graph which accounts for the intensity of different emotions in the total amount of data collected.
* A historical graph based off the amount of tweets labeled as each emotion posted during a time screen.

            ''',
            style={"fontSize":"16px"})  
            ]
        ),
        html.Div(
            style={"padding":"28px","border-top-left-radius":"25px","border-top-right-radius":"25px","vertical-align":"middle","backgroundColor": "#3d93cc","padding":"10px","text-align":"left"},
            id="group1",
            children=[
                html.Div(
                    className="left", 
                    style={"padding":"28px","border-top-left-radius":"25px","border-top-right-radius":"25px","vertical-align":"middle","backgroundColor": "#3d93cc","padding":"10px","text-align":"left"},
                    id="center1",
                    children=[
                        html.H4(style={"vertical-align":"middle","display":"inline","margin-right":"15px","color":"white","text-align":"center"},
                                children=["Fetch latest 200 Tweets by @"]),
                        html.Div(
                            style={"display":"inline"},                            
                            children=[
                            dcc.Input(
                                id="user_input", style={"margin-right":"10px"},
                                type="text",
                                placeholder="Type a Twitter username..."
                            ),
                            html.H4(style={"vertical-align":"middle","display":"inline","margin-right":"15px","color":"white","text-align":"center"},
                                children=["and classify them by emotion: "]),
                            html.Button(
                                "Fetch tweets",
                                style={"backgroundColor": "#b7b7b7","color":"white","margin-right":"10px"},
                                id="user_input_submit",
                            ),
                            html.H4(style={"vertical-align":"middle","display":"inline","margin-left":"5px","color":"white","text-align":"center"},
                                children=["Then, press this button: "]),
                            html.Button(
                                "Process data",
                                style={"backgroundColor": "#b7b7b7","color":"white","float":"right"},
                                id = "user_process_submit",                            
                            )]
                        )
                    ]
                )
            ]
        ),
        html.Div(
            className="twelve-columns",
            style= {"border-bottom-left-radius":"25px","border-bottom-right-radius":"25px","backgroundColor": "#3d93cc","padding":"18px","text-align":"left","padding-bottom":"60px","padding-top":"10px"},
            id="group2",
            children=[
                dcc.Loading(
                    id="loading-1",
                    type="dot",
                    color="white",
                    children = [
                        html.Div(
                            children = [
                                dash_table.DataTable(
                                    id="datatable",
                                    columns = [{"name": i, "id": i} for i in ['Text', 'Emotion','Date']],
                                    data = "",
                                    style_as_list_view=True,
                                    style_cell={"color":"black",
                                                "textAlign":"left",
                                                "fontSize": 14,
                                                "whiteSpace": "normal",
                                                "height": "auto",
                                                "backgroundColor" : " #ebf2ff"
                                               },
                                    page_size=8,
                                    style_header={"backgroundColor": "#c6dbff",
                                                  "fontSize": 16,"fontWeight":"bold",
                                                  "padding":"4px"
                                                 },
                                    #sort_action="native",
                                    #sort_mode="multi",
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
        html.Div(
            className="row",
            id = "group3",
            children=[
                html.Div(
                    style={"margin-top":"20px","padding":"16px","padding-top":"0px","border-radius":"25px","border-top-right-radius":"25px","color":"white","backgroundColor": "#3d93cc","text-align":"center"},
                    className="six columns",
                    id="left",
                    children=[
                        html.H4("Emotion Histogram"),
                        html.Div(
                            children=[
                                dcc.Loading(
                                    id="loading-histogram",
                                    type="dot",
                                    color="white",
                                    children = [
                                        html.Div(
                                            children=[
                                                dcc.Graph(id='emotions_hist',figure={})
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )                    
                    ]
                ),
                html.Div(
                    style={"margin-top":"20px","padding":"16px","padding-top":"0px","border-radius":"25px","border-top-right-radius":"25px","color":"white","backgroundColor": "#3d93cc","text-align":"center"},
                    className="six columns",
                    id="right",
                    children=[
                        html.H4("WordCloud"),
                        html.Div(
                            children=[
                                dcc.Loading(
                                    id="loading-wordcloud",
                                    type="dot",
                                    color="white",
                                    children=[
                                        html.Div(
                                            children=[
                                                html.Img(id="image_wc",src="assets/default.png", style={"padding":"0px","width":"100%","height":"450px"})
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        ),                
        html.Div(
            className="row",
            id = "group4",
            children=[
                html.Div(
                    style={"margin-top":"20px","padding":"16px","padding-top":"0px","border-radius":"25px","border-top-right-radius":"25px","color":"white","backgroundColor": "#3d93cc","text-align":"center"},
                    className="six columns",
                    id="left2",
                    children=[
                        html.H4("Emotion Intensity Pie Chart"),
                        html.Div(
                            children=[
                                dcc.Loading(
                                    id="loading-pie",
                                    type="dot",
                                    color="white",
                                    children=[
                                        html.Div(
                                            children=[
                                                dcc.Graph(id="pie-chart", figure={})
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                        
                    ]
                ),

                html.Div(
                    style={"margin-top":"20px","padding":"16px","padding-top":"0px","border-radius":"25px","border-top-right-radius":"25px","color":"white","backgroundColor": "#3d93cc","text-align":"center"},
                    className="six columns",
                    id="right2",
                    children=[
                        html.H4("Historic of Emotions"),
                        html.Div([
                                                    dcc.Slider(
                                                        id='historic_slider',
                                                        min=1,
                                                        max=90,
                                                        step=1,
                                                        value=7,
                                                        marks={
                                                            1: {'label': '1','style': {'color': 'white'}},
                                                            7: {'label': '7','style': {'color': 'white'}},
                                                            30: {'label': '30','style': {'color': 'white'}},
                                                            90: {'label': '90','style': {'color': 'white'}}
                                                        },
                            
                                                    ),],
                            style={"padding-top":"5px","padding-bottom":"40px"},
                                                    
                                                ),
                        
                        html.Div(
                            children=[
                                dcc.Loading(
                                    id="loading-last",
                                    type="dot",
                                    color="white",
                                    children=[
                                        html.Div(
                                            children=[
                                                
                                                dcc.Graph(id="historic") 
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                                          
                    ]
                )

            ]
        ),
    html.Footer(className="header", style={"border-radius":"25px","font-size":"24px","margin-top":"25px","padding":"28px","text-align":"left"}, children=[
            
            dcc.Markdown('''    

#### 💬 This app was our final project for the 2nd Ed. [Samsung Innovation Campus AI Course](https://www.samsung.com/sa_en/innovation-campus/artificial-intelligence/) with the [University of Málaga](https://www.uma.es/#gsc.tab=0) (2021).
While this project is a test of skill and aptitudes acquired during a programming course, the inspiration for it was the increasingly worrysome issue of depression and teenage self harm/suicidal behaviours, which is an everpresent problem throughout the world.


            ''',
            style={"fontSize":"16px"})  
            ]
        )
    ] #body children
)

# Callbacks

#Update WordCloud img
@app.callback(
    Output('image_wc', 'src'),
    [Input("datatable", "data"),Input("user_process_submit","n_clicks")]
    )
def make_image(data, n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if "user_process_submit" in changed_id:
        df = pd.DataFrame(data)
        img = BytesIO()
        dfreqs = wordfreqs_dict(df)
        plot_wordcloud(data=dfreqs).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

#Update histogram (model classification in emotions)
@app.callback(
    Output('emotions_hist','figure'),
    [Input("datatable", "data"), Input("user_process_submit","n_clicks")]
    )
def update_graph(data, n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if "user_process_submit" in changed_id:
        df = pd.DataFrame(data)
        df_labels = classify_data(df)
        a = list(df_labels['Emotion'].unique())
        b = list(df_labels['Emotion'].value_counts())
    
    return {'data':[{'x':a,'y':b,'type':'bar'}], 'layout':{'title':'Predicted emotions','autosize':True}}

#Update pie chart (calculate emotion intensities)
@app.callback(
    Output("pie-chart", "figure"), 
    [Input("datatable", "data"), Input("user_process_submit","n_clicks")]
    )
def generate_chart(data, n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if "user_process_submit" in changed_id:
        df = pd.DataFrame(data)
        df_scores = emotion_scores(df)
        fig = px.pie(df_scores, values='score', names='emotion')
    return fig

#Update data
@app.callback(
    Output('datatable','data'),
    [Input('user_input_submit','n_clicks')],
    State('user_input','value')
    )
def update_datatable(n_clicks,value):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'user_input_submit' in changed_id:
        user = api.get_user(value)
        user_timeline = user.timeline(count=200)
        df = extract_timeline_as_df(user_timeline)
        df = df.rename({'created_at':'Date'}, inplace = False, axis=1)
        df_text = pd.DataFrame(df[['text','Date']])#en vez del drop() en extract_timeline_as_df() es mejor coger aquí la columna 'text'
        data = classify_data(df_text)
        return data.to_dict('records')

# Update historic
@app.callback(
    Output("historic", "figure"), 
    [Input("datatable", "data"),
     Input("user_process_submit","n_clicks"),
     Input("historic_slider","value")]
    )
def update_historic(data,n_clicks,value):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if ('user_process_submit' in changed_id) or ("historic_slider" in changed_id):
        df = pd.DataFrame(data)
        df_chart = line_chart(df,value)
        fig = px.line(
            df_chart,
            x='Date',
            y=['sadness','anger','surprise','fear','joy','love'],
            title=f"Last {value} days counts by emotion"
        )
        fig.update_traces(mode='markers+lines')
    return fig
    
#Loading animation for datatable
@app.callback(Output("datatable", "children"),
              Input("user_input_submit", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value

#Loading animation for histogram graph
@app.callback(Output("emotions_hist", "children"),
              Input("user_process_submit", "n_clicks")
              )
def input_triggers_spinner(value):
    time.sleep(1)
    return value

#Loading animation for wordcloud graph
@app.callback(Output("image_wc", "children"),
              Input("user_process_submit", "n_clicks"),
              State("image_wc","src")
              )
def input_triggers_spinner(value):
    time.sleep(1)
    return value

#Loading animation for pie chart graph
@app.callback(Output("pie-chart", "children"),
              Input("user_process_submit", "n_clicks"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value

#Loading animation for historic graph
@app.callback(Output("historic", "children"),
              Input("user_process_submit", "n_clicks"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value


# Start the app in a web server
if __name__ == "__main__":
    app.run_server(debug=False)


# In[ ]:





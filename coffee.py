#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st


import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import re
import requests
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import pairwise_distances

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def load_model(model):
    loaded_model = pickle.load(model)
    return loaded_model
with open('coffee_words.pickle','rb') as read_file:
    coffee = pickle.load(read_file)
with open('coffee_ratings.pickle','rb') as read_file:
    ratings = pickle.load(read_file)
with open('combined.pickle','rb') as read_file:
    combined = pickle.load(read_file)
with open('df_full.pickle','rb') as read_file:
    df = pickle.load(read_file)
with open('df_topic_breakdown.pickle','rb') as read_file:
    df_topic_breakdown = pickle.load(read_file)
with open('sentiment.pickle','rb') as read_file:
    sentiment = pickle.load(read_file)

with open('blindtfidf_vec.pickle', 'rb') as read_file:
    blindtfidf = pickle.load(read_file)
with open('blindtfidf_mat.pickle', 'rb') as read_file:
    tfidf_blind = pickle.load(read_file)
ratings = ratings.reset_index().rename(columns={'index':'Roaster'})

with open('nmf_tfidfblind.pickle', 'rb') as read_file:
    nmf_tfidfblind = pickle.load(read_file)

with open('blindvectorizer.pickle', 'rb') as read_file:
    blindvectorizer = pickle.load(read_file)
with open('blindtfidf_topic.pickle', 'rb') as read_file:
    blindtfidf_topic = pickle.load(read_file)
with open('blindtopic_tfidf.pickle', 'rb') as read_file:
    blindtopic_tfidf = pickle.load(read_file)


with open('words_to_score_rf.pickle','rb') as read_file:
    rfr = pickle.load(read_file)
with open('words_to_score_linear.pickle','rb') as read_file:
    lm = pickle.load(read_file)
with open('subcats_to_score_lasso.pickle','rb') as read_file:
    lasso = pickle.load(read_file)
with open('lm_aroma.pickle','rb') as read_file:
    lm_aroma = pickle.load(read_file)
with open('lm_acidity.pickle','rb') as read_file:
    lm_acidity = pickle.load(read_file)
with open('lm_aftertaste.pickle','rb') as read_file:
    lm_aftertaste = pickle.load(read_file)
with open('lm_flavor.pickle','rb') as read_file:
    lm_flavor = pickle.load(read_file)
with open('lm_body.pickle','rb') as read_file:
    lm_body = pickle.load(read_file)

st.title('Coffee Review Recommendations')    

# Set up model
doc_word = tfidf_blind

nmf_model = nmf_tfidfblind
doc_topic = blindtfidf_topic
topic_word = nmf_model.components_

words = blindtfidf.get_feature_names()
t = nmf_model.components_.argsort(axis=1)[:,-1:-7:-1]

topic_words = [[words[e] for e in l] for l in t]

topic_features = ['pos','neg','compound','length','word count','bright_floral_citrus', 'choc_woody_dark', 'tart_sweet_smooth','cacao_nut_clean', 'sweet_nut_pine', 'juicy_cacao_honey', 'red_berries','woody_nut_caramel', 'cherry_vinuous_choc']

if st.checkbox('Predict overall and category score predictions for a coffee description'):
    user_coffee_description = st.text_input("Provide a couple sentence descripton of the flavors, acid level, aroma, aftertaste, and body of your coffee.", '')
    user_text = [user_coffee_description]
    vt = blindtfidf.transform(user_text).todense()
    tt1 = nmf_model.transform(vt)
    
    word_count = pd.DataFrame()
    word_count['text'] = user_text
    word_count['length'] = word_count.text.str.replace(r'\d+','',regex=True).str.len()
    word_count['word count'] = pd.DataFrame(blindvectorizer.transform(user_text).toarray()).sum(axis=1)[0]
    word_count.drop(columns='text',inplace=True)
    
    sid = SentimentIntensityAnalyzer()
    sentiment_vector = pd.DataFrame()
    sentiment_vector['text'] = user_text
    sentiment_vector['scores'] = sentiment_vector.text.apply(lambda Text: sid.polarity_scores(Text))
    sentiment_vector['pos']  = sentiment_vector['scores'].apply(lambda score_dict: score_dict['pos'])
    sentiment_vector['neg']  = sentiment_vector['scores'].apply(lambda score_dict: score_dict['neg'])
    sentiment_vector['compound']  = sentiment_vector['scores'].apply(lambda score_dict: score_dict['compound'])
    sentiment_vector.drop(columns=['text','scores'],inplace=True)
    
    attributes = pd.concat([sentiment_vector,word_count],axis=1)
    attributes = pd.concat([attributes,pd.DataFrame(tt1)],axis=1)

    
    overall = lm.predict(attributes)
    aroma = lm_aroma.predict(attributes)
    acidity = lm_acidity.predict(attributes)
    aftertaste = lm_aftertaste.predict(attributes)
    flavor = lm_flavor.predict(attributes)
    body = lm_body.predict(attributes)

    if user_coffee_description == '':
        st.write('Excited to predict the score of your coffee!')
    else:
        st.write('Based on your input coffee, I predict it to receive a score of:',overall[0].round(2),'\n\n',
                'An aroma score of (out of 10):',aroma[0].round(2),'\n\n',
                'An acidity score of (out of 10):',acidity[0].round(2),'\n\n',
                'An aftertaste score of (out of 10):',aftertaste[0].round(2),'\n\n',
                'A flavor score of (out of 10):',flavor[0].round(2),'\n\n',
                'A body score of (out of 10):',body[0].round(2))

if st.checkbox('Predict overall score for a coffee based on its subcategory scores'):
    user_coffee_description = st.text_input("Provide a couple sentence descripton of the flavors, acid level, aroma, aftertaste, and body of your coffee.", '')
    text = [user_coffee_description]
    vt = blindtfidf.transform(text).todense()
    tt1 = nmf_model.transform(vt)
    overall = lm.predict(tt1)
    if user_coffee_description == '':
        st.write('Excited to predict the scores of your coffee!')
    else:
        st.write('Based on your input coffee, I predict it to receive a score of:')
        
if st.checkbox('Get a recommendation for coffee from a description'):
    # Format inputs
    user_coffee_description = st.text_input("Give a couple sentences here of how you describe your ideal coffee. Try to include as much as you can about your desired flavor profile.", '')
    text = [user_coffee_description]
    vt = blindtfidf.transform(text).todense()
    tt1 = nmf_model.transform(vt)

    #Find Recommendations
    indices = pairwise_distances(tt1.reshape(1,-1),doc_topic,metric='cosine').argsort()
    recs = list(indices[0][0:4])
    # df_topic_breakdown.iloc[recs]
    # st.write('The coffee you liked was described as:',t[0])
    st.write('\n')
    if user_coffee_description == '':
        st.write('Excited to recommend a coffee for you!')
    else:
        st.write('Based on your input coffee, I recommend you try:','\n\n',ratings.iloc[recs[0]]['Roaster'],'who roast a bean from',ratings.iloc[recs[0]]['Coffee Origin'],'.','\n\n','It could be desribed as:','\n\n',coffee.iloc[recs[0]].Review)


# In[ ]:





# In[ ]:





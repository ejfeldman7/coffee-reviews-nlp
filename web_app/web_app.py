#!/usr/bin/env python
# coding: utf-8

# ## Streamlit Application
# 
# Streamlit application for deploying coffee recommender, score predictor, and review generator.

# In[8]:


# !pip install pipreqsnb
# !pipreqsnb C:\Users\ejfel\Documents\metis_repos\Coffee-Reviews-NLP\coffee.py


# In[2]:


import streamlit as st

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import re
import requests
import pickle
from collections import defaultdict
import random

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import pairwise_distances

#Paths
#'/app/Coffee-Reviews-NLP/web_app/models_embeddings/'
#'/app/Coffee-Reviews-NLP/web_app/data/'

with open('/app/coffee-reviews-nlp/web_app/coffee_words.pickle','rb') as read_file:
    coffee = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/coffee_ratings.pickle','rb') as read_file:
    ratings = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/combined.pickle','rb') as read_file:
    combined = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/df_full.pickle','rb') as read_file:
    df = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/df_topic_breakdown.pickle','rb') as read_file:
    df_topic_breakdown = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/sentiment.pickle','rb') as read_file:
    sentiment = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/generating_reviews.pickle','rb') as read_file:
    generating_reviews = pickle.load(read_file)

with open('/app/coffee-reviews-nlp/web_app/blindtfidf_vec.pickle', 'rb') as read_file:
    blindtfidf = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/blindtfidf_mat.pickle', 'rb') as read_file:
    tfidf_blind = pickle.load(read_file)
ratings = ratings.reset_index().rename(columns={'index':'Roaster'})

with open('/app/coffee-reviews-nlp/web_app/nmf_tfidfblind.pickle', 'rb') as read_file:
    nmf_tfidfblind = pickle.load(read_file)

with open('/app/coffee-reviews-nlp/web_app/blindvectorizer.pickle', 'rb') as read_file:
    blindvectorizer = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/blindtfidf_topic.pickle', 'rb') as read_file:
    blindtfidf_topic = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/blindtopic_tfidf.pickle', 'rb') as read_file:
    blindtopic_tfidf = pickle.load(read_file)


with open('/app/coffee-reviews-nlp/web_app/words_to_score_rf.pickle','rb') as read_file:
    rfr = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/num_to_score_RF.pickle','rb') as read_file:
    rfr_num = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/words_to_score_linear.pickle','rb') as read_file:
    lm = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/subcats_to_score_lasso.pickle','rb') as read_file:
    lasso = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/lm_aroma.pickle','rb') as read_file:
    lm_aroma = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/lm_acidity.pickle','rb') as read_file:
    lm_acidity = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/lm_aftertaste.pickle','rb') as read_file:
    lm_aftertaste = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/lm_flavor.pickle','rb') as read_file:
    lm_flavor = pickle.load(read_file)
with open('/app/coffee-reviews-nlp/web_app/lm_body.pickle','rb') as read_file:
    lm_body = pickle.load(read_file)

import os.path

# create a button in the side bar that will move to the next page/radio button choice
next = st.sidebar.button('Next on list')

# will use this list and next button to increment page, MUST BE in the SAME order
# as the list passed to the radio button
new_choice = ['Home','Recommender','Score from Text','Score from Score','Generated Reviews']

# This is what makes this work, check directory for a pickled file that contains
# the index of the page you want displayed, if it exists, then you pick up where the
#previous run through of your Streamlit Script left off,
# if it's the first go it's just set to 0
if os.path.isfile('next.p'):
    next_clicked = pickle.load(open('next.p', 'rb'))
    # check if you are at the end of the list of pages
    if next_clicked == len(new_choice):
        next_clicked = 0 # go back to the beginning i.e. homepage
else:
    next_clicked = 0 #the start

# this is the second tricky bit, check to see if the person has clicked the
# next button and increment our index tracker (next_clicked)
if next:
    #increment value to get to the next page
    next_clicked = next_clicked +1

    # check if you are at the end of the list of pages again
    if next_clicked == len(new_choice):
        next_clicked = 0 # go back to the beginning i.e. homepage

# create your radio button with the index that we loaded
choice = st.sidebar.radio("go to",('Home','Recommender','Score from Text','Score from Score','Generated Reviews'), index=next_clicked)

# pickle the index associated with the value, to keep track if the radio button has been used
pickle.dump(new_choice.index(choice), open('next.p', 'wb'))

# finally get to whats on each page
if choice == 'Home':
    st.title('Welcome to my data analysis app for coffee reviews!')
    '''
    This project was built from just under 6000 reviews from  www.coffeereview.com. \n
    The blind reviews were used to create nine-dimensional flavor vectors for comparisons between coffees. \n
    These vectors and additional features can be used for recommendations, predicting scores, and more.  \r\n
    This site was created by Ethan Feldman. You can find him on [GitHub](https://github.com/ejfeldman7), [LinkedIn](https://www.linkedin.com/in/feldmanethan/),
    [Medium/TDS](https://ethan-feldman.medium.com/) and eventually on his website (link to come)!  \r\n
    \r\n
    On the side bar on the left you will find a few different application  \r\n
    Below is a quick table of contents for the different pages of the site
    '''
    '''
    1. This is the __Home Page__
    2. Use the __Recommender__ app to get a coffee recommendation based on your flavor description
    3. Use the __Score from Text__ app to generate a prediction for overall and subcategory score based on a coffee's description
    4. Use the __ Score from Scores__ app to generate an overall score prediction based on subcategory scores
    5. Use the __Generated Reviews__ app to create a computer generated review for a coffee depending on different flavor attributes
    '''
    
elif choice == 'Recommender':
    st.title('Coffee Recommender')
    st.write('Get a new coffee recommendation')
    # Format inputs
    user_coffee_description = st.text_input("Give a couple sentences here of how you describe your ideal coffee. Try to include as much as you can about your desired flavor profile.", '')
    text = [user_coffee_description]
    doc_topic = blindtfidf_topic
    vt = blindtfidf.transform(text).todense()
    tt1 = nmf_tfidfblind.transform(vt)

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

elif choice == 'Score from Text':
    st.title('Score Predictor')
    st.write('Predict coffee scores from reviews')
    
    user_coffee_description = st.text_input("Provide a couple sentence descripton of the flavors, acid level, aroma, aftertaste, and body of your coffee.", '')
    user_text = [user_coffee_description]
    vt = blindtfidf.transform(user_text).todense()
    tt1 = nmf_tfidfblind.transform(vt)

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

elif choice == 'Score from Score':
    st.title('Score Predictor (if you have the details)')
    st.write('Predict an overall score from subcategories')
    
    aroma = st.slider('aroma',min_value=1,max_value=10,step=1)
    body = st.slider('body',min_value=1,max_value=10,step=1)
    flavor = st.slider('flavor',min_value=1,max_value=10,step=1)
    aftertaste = st.slider('aftertaste',min_value=1,max_value=10,step=1)
    acidity = st.slider('acidity',min_value=1,max_value=10,step=1)

    features = ['aroma', 'body', 'flavor', 'aftertaste', 'acidity']
    df_feat = pd.DataFrame(columns = features)
    df_feat.aroma = [aroma]
    df_feat.body = [body]
    df_feat.flavor = [flavor]
    df_feat.aftertaste = [aftertaste]
    df_feat.acidity = [acidity]

    overall = rfr_num.predict(df_feat)[0].round(2)
    st.write('With subcategory scores as shown above, I predict your coffee to be review overall as:',overall)

elif choice == 'Generated Reviews':
    st.title('Review Generator')
    st.write('Generate a rough draft review based on past reviews in the category')    

    first = st.checkbox("Coffee Type: Smooth, Citrus, Floral")
    second = st.checkbox("Coffee Type: Dark, Chocolate, Roast, Wood ")
    third = st.checkbox("Coffee Type: Sweet, Almond, Tart, Zest")
    fourth = st.checkbox("Coffee Type: Cacao, Nutty, Dried Fruit, Clean")
    fifth = st.checkbox("Coffee Type: Sweet, Hazelnut, Resinous")
    sixth = st.checkbox("Coffee Type: Juicy, Acidic, Cacao, Nectar")
    seventh = st.checkbox("Coffee Type: Red Berries")
    eighth = st.checkbox("Coffee Type: Caramel, Nutty, Woody")
    ninth = st.checkbox("Coffee Type: Cherry, Vinuous, Chocolate")

    def markov_chain(corpus):
        #tokenize text into words
        words = []
        for review in corpus:
            words += review.split(' ')

        # initialize a default dictionary to hold all of the words and next words
        word_dict = defaultdict(list)

        # create a zipped list of all of the word pairs and put them in word: list of next words format
        for first, second in list(zip(words,words[1:])):
            word_dict[first].append(second)

        return dict(word_dict)

    def generate_review(corpus, n_words=30):
        start = random.choice(list(corpus.keys())).capitalize()
        sentence = []
        sentence.append(start)
        for _ in range(n_words):
            next_word = random.choice(list(corpus[sentence[-1].lower()]))
            sentence.append(next_word)

        return ' '.join(sentence)+'.'  

    text_list = []
    if first:
        text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 0].review]
    elif second:
        text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 1].review]
    elif third:
        text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 2].review]
    elif fourth:
        text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 3].review]
    elif fifth:
        text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 4].review]
    elif sixth:
        text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 5].review]
    elif seventh:
        text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 6].review]
    elif eighth:
        text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 7].review]
    elif ninth:
        text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 8].review]

    if text_list == []:
        st.write('Pick a coffee type or combine a few to see a (rough) computer generated review!')
    else:
        st.write(generate_review(markov_chain(text_list)))

    


# In[ ]:


# st.title('Coffee Review Recommendations and Analysis')    

# # st.write("Using a radio button restricts selection to only one option at a time.")
# # neighborhood = st.radio("Neighborhood", df.neighbourhood_group.unique())
# # show_exp = st.checkbox("Include expensive listings")
# # Set up model
# doc_word = tfidf_blind

# nmf_model = nmf_tfidfblind
# doc_topic = blindtfidf_topic
# topic_word = nmf_model.components_

# words = blindtfidf.get_feature_names()
# t = nmf_model.components_.argsort(axis=1)[:,-1:-7:-1]

# topic_words = [[words[e] for e in l] for l in t]

# # First checkbox: coffee recommender

# if st.checkbox('Get a recommendation for coffee from a description'):
#     # Format inputs
#     user_coffee_description = st.text_input("Give a couple sentences here of how you describe your ideal coffee. Try to include as much as you can about your desired flavor profile.", '')
#     text = [user_coffee_description]
#     vt = blindtfidf.transform(text).todense()
#     tt1 = nmf_model.transform(vt)

#     #Find Recommendations
#     indices = pairwise_distances(tt1.reshape(1,-1),doc_topic,metric='cosine').argsort()
#     recs = list(indices[0][0:4])
#     # df_topic_breakdown.iloc[recs]
#     # st.write('The coffee you liked was described as:',t[0])
#     st.write('\n')
#     if user_coffee_description == '':
#         st.write('Excited to recommend a coffee for you!')
#     else:
#         st.write('Based on your input coffee, I recommend you try:','\n\n',ratings.iloc[recs[0]]['Roaster'],'who roast a bean from',ratings.iloc[recs[0]]['Coffee Origin'],'.','\n\n','It could be desribed as:','\n\n',coffee.iloc[recs[0]].Review)

# # Second checkbox: coffee scorer
# if st.checkbox('Predict overall and category score predictions for a coffee description'):
#     user_coffee_description = st.text_input("Provide a couple sentence descripton of the flavors, acid level, aroma, aftertaste, and body of your coffee.", '')
#     user_text = [user_coffee_description]
#     vt = blindtfidf.transform(user_text).todense()
#     tt1 = nmf_model.transform(vt)
    
#     word_count = pd.DataFrame()
#     word_count['text'] = user_text
#     word_count['length'] = word_count.text.str.replace(r'\d+','',regex=True).str.len()
#     word_count['word count'] = pd.DataFrame(blindvectorizer.transform(user_text).toarray()).sum(axis=1)[0]
#     word_count.drop(columns='text',inplace=True)
    
#     sid = SentimentIntensityAnalyzer()
#     sentiment_vector = pd.DataFrame()
#     sentiment_vector['text'] = user_text
#     sentiment_vector['scores'] = sentiment_vector.text.apply(lambda Text: sid.polarity_scores(Text))
#     sentiment_vector['pos']  = sentiment_vector['scores'].apply(lambda score_dict: score_dict['pos'])
#     sentiment_vector['neg']  = sentiment_vector['scores'].apply(lambda score_dict: score_dict['neg'])
#     sentiment_vector['compound']  = sentiment_vector['scores'].apply(lambda score_dict: score_dict['compound'])
#     sentiment_vector.drop(columns=['text','scores'],inplace=True)
    
#     attributes = pd.concat([sentiment_vector,word_count],axis=1)
#     attributes = pd.concat([attributes,pd.DataFrame(tt1)],axis=1)

    
#     overall = lm.predict(attributes)
#     aroma = lm_aroma.predict(attributes)
#     acidity = lm_acidity.predict(attributes)
#     aftertaste = lm_aftertaste.predict(attributes)
#     flavor = lm_flavor.predict(attributes)
#     body = lm_body.predict(attributes)

#     if user_coffee_description == '':
#         st.write('Excited to predict the score of your coffee!')
#     else:
#         st.write('Based on your input coffee, I predict it to receive a score of:',overall[0].round(2),'\n\n',
#                 'An aroma score of (out of 10):',aroma[0].round(2),'\n\n',
#                 'An acidity score of (out of 10):',acidity[0].round(2),'\n\n',
#                 'An aftertaste score of (out of 10):',aftertaste[0].round(2),'\n\n',
#                 'A flavor score of (out of 10):',flavor[0].round(2),'\n\n',
#                 'A body score of (out of 10):',body[0].round(2))

        

# # Third checkbox: coffee scorer if you know the underlying category scores...
# if st.checkbox('Predict overall score for a coffee based on its subcategory scores'):
#     aroma = st.slider('aroma',min_value=1,max_value=10,step=1)
#     body = st.slider('body',min_value=1,max_value=10,step=1)
#     flavor = st.slider('flavor',min_value=1,max_value=10,step=1)
#     aftertaste = st.slider('aftertaste',min_value=1,max_value=10,step=1)
#     acidity = st.slider('acidity',min_value=1,max_value=10,step=1)
    
#     features = ['aroma', 'body', 'flavor', 'aftertaste', 'acidity']
#     df_feat = pd.DataFrame(columns = features)
#     df_feat.aroma = [aroma]
#     df_feat.body = [body]
#     df_feat.flavor = [flavor]
#     df_feat.aftertaste = [aftertaste]
#     df_feat.acidity = [acidity]
    
#     overall = rfr_num.predict(df_feat)[0].round(2)
#     st.write('With subcategory scores as shown above, I predict your coffee to be review overall as:',overall)

# # Fourth checkbox: computer generated review for a coffee...
# if st.checkbox('Generate a review for a coffee based on which group it falls into'): 
    
#     first = st.checkbox("Coffee Type: 1")
#     second = st.checkbox("Coffee Type: 2")
#     third = st.checkbox("Coffee Type: 3")
#     fourth = st.checkbox("Coffee Type: 4")
#     fifth = st.checkbox("Coffee Type: 5")
#     sixth = st.checkbox("Coffee Type: 6")
#     seventh = st.checkbox("Coffee Type: 7")
#     eighth = st.checkbox("Coffee Type: 8")
#     ninth = st.checkbox("Coffee Type: 9")
    
#     def markov_chain(corpus):
#         #tokenize text into words
#         words = []
#         for review in corpus:
#             words += review.split(' ')

#         # initialize a default dictionary to hold all of the words and next words
#         word_dict = defaultdict(list)

#         # create a zipped list of all of the word pairs and put them in word: list of next words format
#         for first, second in list(zip(words,words[1:])):
#             word_dict[first].append(second)

#         return dict(word_dict)

#     def generate_review(corpus, n_words=50):
#         start = random.choice(list(corpus.keys())).capitalize()
#         sentence = []
#         sentence.append(start)
#         for _ in range(n_words):
#             next_word = random.choice(list(corpus[sentence[-1].lower()]))
#             sentence.append(next_word)

#         return ' '.join(sentence)+'.'  
    
#     text_list = []
#     if first:
#         text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 0].review]
#     elif second:
#         text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 1].review]
#     elif third:
#         text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 2].review]
#     elif fourth:
#         text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 3].review]
#     elif fifth:
#         text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 4].review]
#     elif sixth:
#         text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 5].review]
#     elif seventh:
#         text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 6].review]
#     elif eighth:
#         text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 7].review]
#     elif ninth:
#         text_list = text_list + [review.lower() for review in generating_reviews[generating_reviews.group == 8].review]

#     if text_list == []:
#         st.write('Pick a coffee type to see a computer generated review!')
#     else:
#         st.write(generate_review(markov_chain(text_list)))


# In[ ]:





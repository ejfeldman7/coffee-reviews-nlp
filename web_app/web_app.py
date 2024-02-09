import pickle
from collections import defaultdict
import random
import os.path
import os

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import plotly.graph_objects as go
from nltk import download
from nltk.sentiment.vader import SentimentIntensityAnalyzer


download('vader_lexicon')
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
pickle_jar = os.path.join(CUR_DIR, 'model_files')
data_path = os.path.join(CUR_DIR, 'data')
data = {'df_full': None, 'df_topic_breakdown': None, 'coffee_ratings': None}
pickles = {'coffee_words': None, 'lm_acidity': None, 'lm_aftertaste': None,
           'lm_flavor': None, 'lm_body': None, 'lm_aroma': None, 'words_to_score_linear': None,
           'generating_reviews': None, 'tfidfvec_202402': None,
           'nmf_202402': None, 'blindvectorizer': None,
           'tfidftopic_202402': None, 'num_to_score_RF': None}


for pkl in pickles.keys():
    try:
        with open(os.path.join(pickle_jar, f'{pkl}.pickle'), 'rb') as f:
            pickles[pkl] = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file {pkl}: {e}")
for datum in data.keys():
    try:
        with open(os.path.join(data_path, f'{datum}.pickle'), 'rb') as f:
            data[datum] = pickle.load(f)
    except Exception as e:
        print(f"Error loading data file {datum}: {e}")

# doc_topic = pickles['tfidftmat_202402']
blindtfidf = pickles['tfidfvec_202402']
blindtfidf_topic = pickles['tfidftopic_202402']
nmf_tfidfblind = pickles['nmf_202402']
blindvectorizer = pickles['blindvectorizer']
rfr_num = pickles['num_to_score_RF']
coffee = pickles['coffee_words']

lm = pickles['words_to_score_linear']
lm_aroma = pickles['lm_aroma']
lm_acidity = pickles['lm_acidity']
lm_aftertaste = pickles['lm_aftertaste']
lm_flavor = pickles['lm_flavor']
lm_body = pickles['lm_body']
generating_reviews = pickles['generating_reviews']

ratings = data['ratings_202402']
# df_full = data['df_full']
# df_topic_breakdown = data['df_topic_breakdown']

ratings = ratings.reset_index().rename(columns={'index': 'Roaster'})

# create a button in the side bar that will move to the next page/radio button choice
next = st.sidebar.button('Next on list')
new_choice = ['Home', 'Recommender', 'Score from Text', 'Score from Subscores', 'Generated Reviews']

if os.path.isfile('next.p'):
    next_clicked = pickle.load(open('next.p', 'rb'))
    if next_clicked == len(new_choice):
        next_clicked = 0
else:
    next_clicked = 0

if next:
    next_clicked = next_clicked + 1
    if next_clicked == len(new_choice):
        next_clicked = 0
choice = st.sidebar.radio("go to", ('Home', 'Recommender', 'Score from Text', 'Score from Subscores', 'Generated Reviews'), index=next_clicked)

st.sidebar.write(
    '''
    __About__ \n
    This project was built from just under 6000 reviews from www.coffeereview.com. The blind reviews were used to create nine-dimensional flavor vectors for comparisons between coffees.
    \n
    This site was created by Ethan Feldman. You can find him on [GitHub](https://github.com/ejfeldman7), [LinkedIn](https://www.linkedin.com/in/feldmanethan/), [Medium/TDS](https://ethan-feldman.medium.com/) and eventually on his website (link to come)!
    ''')
# pickle the index associated with the value, to keep track if the radio button has been used
pickle.dump(new_choice.index(choice), open('next.p', 'wb'))

# Define informatoin on each page
if choice == 'Home':
    st.title('Welcome to my data analysis app for coffee reviews!')
    '''
    This project was built from just under 6000 reviews from  www.coffeereview.com.
    The blind reviews were used to create nine-dimensional flavor vectors using non-negative matrix factorization on a TF-IDF encoding of each coffee's review. This enabled comparison between coffees by their difference or similarity across the derived flavor spectrum.
    These vectors and additional features were then used for recommendations of coffees with similar vectors, predicting scores, and more.  \r\n
    __On the side bar__ on the left you will find a few different application  \r\n
    __Below__ is a quick table of contents for the different pages of the site
    '''
    '''
    1. This is the __Home Page__
    2. Use the __Recommender__ app to get a coffee recommendation based on your flavor description
    3. Use the __Score from Text__ app to generate a prediction for overall and subcategory score based on a coffee's description
    4. Use the __ Score from Subscores__ app to generate an overall score prediction based on subcategory scores
    5. Use the __Generated Reviews__ app to create a computer generated review for a coffee depending on different flavor attributes
    \r\n
    This site was created by Ethan Feldman. You can find him on [GitHub](https://github.com/ejfeldman7), [LinkedIn](https://www.linkedin.com/in/feldmanethan/),
    [Medium/TDS](https://ethan-feldman.medium.com/) and on his [website](https://www.ejfeldman.com/)  \r\n
    '''

elif choice == 'Recommender':
    st.title('Coffee Recommender')
    st.write('Get a new coffee recommendation. Please keep in mind the reviews in this recommendation span across multiple years and the coffee recommended may not be currently available.')
    st.write('Please select from __one__ of the two options below.')

    # Radio button for input style
    rec_choice = st.radio('Choices', ('I want to enter a description of my own', "I'd like a list of adjectives to choose from"), index=0)
#     first = st.checkbox('I want to enter a description of my own')
#     second = st.checkbox("I'd like a list of adjectives to choose from")
    user_coffee_description = ''
    if rec_choice == 'I want to enter a description of my own':
        # User inputs their own description
        user_coffee_description = st.text_input("Give a couple sentences here of how you describe your ideal coffee. Try to include as much as you can about your desired flavor profile.", '')
        st.write('''
        If you are not sure of a description to input, feel free to copy and paste this example of an Ethiopia Suke Quto from Street Bean: \n
        Crisply sweet, citrusy-bright. Tangerine zest, apricot, almond, cocoa nib, freesia-like flowers in aroma and cup. Sweet-leaning structure with high-toned acidity; smooth, satiny mouthfeel. Notes of tangerine zest and almond characterize the crisp, long finish.
        ''')
    elif rec_choice == "I'd like a list of adjectives to choose from":
        # User selects from a list of adjective categories
        col1, col2, col3 = st.beta_columns(3)
        a, b, c, d, e, f = col1.checkbox('Berries'), col1.checkbox('Cherry'), col1.checkbox('Wine-y'), col1.checkbox('Floral'),  col1.checkbox('Citrus'), col1.checkbox('Tropical')

        g, h, i, j, k, l = col2.checkbox('Woody'), col2.checkbox('Caramel'), col2.checkbox('Honey'), col2.checkbox('Chocolate'), col2.checkbox('Very Light Roast'), col2.checkbox('Very Dark Roast')

        m, n, o, p, q, r = col3.checkbox('Tart/Acidic'), col3.checkbox('Complex'), col3.checkbox('Nutty'), col3.checkbox('Silky'), col3.checkbox('Crisp'), col3.checkbox('Rich')

        text_list = []
        if a:
            text_list = text_list + [random.choice(['blackberry', 'raspberry', 'currant'])]
        elif b:
            text_list = text_list + [random.choice(['cherry'])]
        elif c:
            text_list = text_list + [random.choice(['wine', 'brandy', 'vinous'])]
        elif d:
            text_list = text_list + [random.choice(['flowers', 'floral', 'honeysuckle'])]
        elif e:
            text_list = text_list + [random.choice(['zest', 'orange', 'lemon'])]
        elif f:
            text_list = text_list + [random.choice(['mango', 'papaya', 'lychee', 'apricot'])]
        elif g:
            text_list = text_list + [random.choice(['wood', 'cedar', 'fir', 'sandalwood'])]
        elif h:
            text_list = text_list + [random.choice(['caramel', 'sweet', 'sugar'])]
        elif i:
            text_list = text_list + [random.choice(['sweet', 'honey', 'sugar'])]
        elif j:
            text_list = text_list + [random.choice(['cacao', 'cocoa', 'chocolate', 'nib'])]
        elif k:
            text_list = text_list + [random.choice(['bright', 'vibrant', 'nuanced', 'lively'])]
        elif l:
            text_list = text_list + [random.choice(['dark', 'roast', 'scorched', 'rich'])]
        elif m:
            text_list = text_list + [random.choice(['bright', 'acidic', 'tart'])]
        elif n:
            text_list = text_list + [random.choice(['nuanced', 'complex', 'structured', 'layers', 'depth'])]
        elif o:
            text_list = text_list + [random.choice(['nutty', 'almond', 'hazelnut', 'nut'])]
        elif p:
            text_list = text_list + [random.choice(['silky', 'smooth', 'full', 'structured'])]
        elif q:
            text_list = text_list + [random.choice(['crisp', 'dry'])]
        elif r:
            text_list = text_list + [random.choice(['rich'])]
        user_coffee_description = ' '.join(text_list)

    # User text is converted to vector and given topic scores
    text = [user_coffee_description]
    doc_topic = blindtfidf_topic
    vt = np.asarray(blindtfidf.transform(text).todense())
    tt1 = nmf_tfidfblind.transform(vt)

    # Find Recommendations
    indices = pairwise_distances(tt1.reshape(1, -1), doc_topic, metric='cosine').argsort()
    recs = list(indices[0][0:4])

    st.write('\n')
    # Placeholder for recommendation
    if user_coffee_description == '':
        st.write('''
        Excited to recommend a coffee for you!''')
    # Give recommendations
    else:
        # Setting up polar plots for comparison of input and recommendation
        example_comps = [doc_topic[recs[0]], tt1[0]]
        names = [ratings.iloc[recs[0]]['Roaster'], 'Your Input Description']
        categories = ['tart_tea_juicy', 'fruit_floral_dry', 'chocolate_dark_rich', 'pastry_hazelnut_date', 'cacao_crisp_citrus', 'floral_nectar_acid', 'orange_grapefruit_lime', 'currant_cherry_berry', 'wood_nut_caramel']
        topics = ['Tart, Tea, Juicy', 'Fruit, Floral, Dry', 'Choc, Dark, Rich', 'Pastry, Hazelnut, Date', 'Cacao, Crisp, Citrus', 'Floral, Nectar, Acid', 'Citrus!!!', 'Currant Cherry Berry', 'Wood, Nutty, Caramel']
        fig = go.Figure()
        for i in range(0, 2):
            fig.add_trace(go.Scatterpolar(
                  r=example_comps[i],
                  theta=topics,
                  fill=None,
                  name=names[i],
                  opacity=.5,
            ))

        fig.update_layout(
                          title={
                                 'text': 'Visualizing a comparison',
                                 'y': .9,
                                 'x': .5,
                                 'xanchor': 'center',
                                 'yanchor': 'top'},
                          legend_title="Comparison Coffees",
                          polar=dict(
                                     radialaxis=dict(
                                                     visible=False,
                                                     range=[0, max(max(doc_topic[recs[0]]), max(tt1[0]))+.03]
                                                    )),
                          showlegend=True
            )

        if ratings.iloc[recs[0]]['Coffee Origin'] == 'Not disclosed' or not ratings.iloc[recs[0]]['Coffee Origin']:
            st.write('Based on your input coffee, I recommend you try a blend from:', '\n\n', ratings.iloc[recs[0]]['Roaster'], '\n\n', 'It could be desribed as:', '\n\n', coffee.iloc[recs[0]].Review)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write('Based on your input coffee, I recommend you try:', '\n\n', ratings.iloc[recs[0]]['Roaster'], 'who roast a great bean from', str(ratings.iloc[recs[0]]['Coffee Origin'])+'.', '\n\n', 'It could be desribed as:', '\n\n', coffee.iloc[recs[0]].Review)
            st.plotly_chart(fig, use_container_width=True)

elif choice == 'Score from Text':
    st.title('Score Predictor')
    st.write('Predict coffee scores from reviews. This output is a prediction of the score that might be assigned on a 0-100 scale as well as subscores on a 0-10 scale.')

    user_coffee_description = st.text_input("Provide a couple sentence descripton of the flavors, acid level, aroma, aftertaste, and body of your coffee.", '')
    user_text = [user_coffee_description]
    vt = np.asarray(blindtfidf.transform(user_text).todense())
    tt1 = nmf_tfidfblind.transform(vt)

    word_count = pd.DataFrame()
    word_count['text'] = user_text
    word_count['length'] = word_count.text.str.replace(r'\d+', '', regex=True).str.len()
    word_count['word count'] = pd.DataFrame(blindvectorizer.transform(user_text).toarray()).sum(axis=1)[0]
    word_count.drop(columns='text', inplace=True)

    sid = SentimentIntensityAnalyzer()
    sentiment_vector = pd.DataFrame()
    sentiment_vector['text'] = user_text
    sentiment_vector['scores'] = sentiment_vector.text.apply(lambda Text: sid.polarity_scores(Text))
    sentiment_vector['pos'] = sentiment_vector['scores'].apply(lambda score_dict: score_dict['pos'])
    sentiment_vector['neg'] = sentiment_vector['scores'].apply(lambda score_dict: score_dict['neg'])
    sentiment_vector['compound'] = sentiment_vector['scores'].apply(lambda score_dict: score_dict['compound'])
    sentiment_vector.drop(columns=['text', 'scores'], inplace=True)

    attributes = pd.concat([sentiment_vector, word_count], axis=1)
    attributes = pd.concat([attributes, pd.DataFrame(tt1)], axis=1)
    attributes.columns = attributes.columns.astype(str)

    overall = lm.predict(attributes)
    aroma = lm_aroma.predict(attributes)
    acidity = lm_acidity.predict(attributes)
    aftertaste = lm_aftertaste.predict(attributes)
    flavor = lm_flavor.predict(attributes)
    body = lm_body.predict(attributes)

    if user_coffee_description == '':
        st.write('''
        Excited to predict the score of your coffee! \n
        If you are not sure of a description to input, feel free to copy and paste this example of an Ethiopia Suke Quto from Street Bean which was not part of the reviews of this project. \n
        If you are interested, the real scores for this coffee were: \n
        Overall 93, Aroma 9, Acidity 9, Body 8, Flavor 9, Aftertaste 8 \n
        Crisply sweet, citrusy-bright. Tangerine zest, apricot, almond, cocoa nib, freesia-like flowers in aroma and cup. Sweet-leaning structure with high-toned acidity; smooth, satiny mouthfeel. Notes of tangerine zest and almond characterize the crisp, long finish.
        ''')
    else:
        st.write('Based on your input coffee, I predict it to receive a score of:', overall[0].round(2), '\n\n',
                 'An aroma score of (out of 10):', aroma[0].round(2), '\n\n',
                 'An acidity score of (out of 10):', acidity[0].round(2), '\n\n',
                 'An aftertaste score of (out of 10):', aftertaste[0].round(2), '\n\n',
                 'A flavor score of (out of 10):', flavor[0].round(2), '\n\n',
                 'A body score of (out of 10):', body[0].round(2))

elif choice == 'Score from Subscores':
    st.title('Overall Score Based on Subcategories')
    st.write('Use this tool to create a 0-100 rating based on subscores in the categories below.')

    aroma = st.slider('aroma', min_value=1, max_value=10, step=1)
    body = st.slider('body', min_value=1, max_value=10, step=1)
    flavor = st.slider('flavor', min_value=1, max_value=10, step=1)
    aftertaste = st.slider('aftertaste', min_value=1, max_value=10, step=1)
    acidity = st.slider('acidity', min_value=1, max_value=10, step=1)

    features = ['aroma', 'body', 'flavor', 'aftertaste', 'acidity']
    df_feat = pd.DataFrame(columns=features)
    df_feat.aroma = [aroma]
    df_feat.body = [body]
    df_feat.flavor = [flavor]
    df_feat.aftertaste = [aftertaste]
    df_feat.acidity = [acidity]

    overall = rfr_num.predict(df_feat)[0].round(2)
    st.write('With subcategory scores as shown above, I predict your coffee to be review overall as:', overall)

elif choice == 'Generated Reviews':
    st.title('Review Generator')
    st.write('Generate a "rough draft" review based on past reviews in the category.')
    cats = ['Tart, Tea, Juicy', 'Fruit, Floral, Dry', 'Choc, Dark, Rich', 'Pastry, Hazelnut, Date', 'Cacao, Crisp, Citrus', 'Floral, Nectar, Acid', 'Citrus!!!', 'Currant Cherry Berry', 'Wood, Nutty, Caramel']

    first = st.checkbox(f"Coffee Type: f{cats[0]}")
    second = st.checkbox(f"Coffee Type: f{cats[1]}")
    third = st.checkbox(f"Coffee Type: f{cats[2]}")
    fourth = st.checkbox(f"Coffee Type: f{cats[3]}")
    fifth = st.checkbox(f"Coffee Type: f{cats[4]}")
    sixth = st.checkbox(f"Coffee Type: f{cats[5]}")
    seventh = st.checkbox(f"Coffee Type: f{cats[6]}")
    eighth = st.checkbox(f"Coffee Type: f{cats[7]}")
    ninth = st.checkbox(f"Coffee Type: f{cats[8]}")

    def markov_chain(corpus):
        # tokenize text into words
        words = []
        for review in corpus:
            words += review.split(' ')

        # initialize a default dictionary to hold all of the words and next words
        word_dict = defaultdict(list)

        # create a zipped list of all of the word pairs and put them in word: list of next words format
        for first, second in list(zip(words, words[1:])):
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

# Coffee-Reviews-NLP
 Analysis of cofffee reviews using natural language processing to create predictions for review scores, recommend similar coffees, and generate coffee reviews

# Contents

## In the Notebooks folder you will find:

- scraping: Beautiful soup and my .py files to scrape the coffee reviews from CoffeeReview.com  
- first-nlp: Preprocessing, regex, lemmatizing, countvectorizing, and tf-idf  
- nmf: Creating clusters of reviews into topics based on countvectorized corpus
- coffee_ratings_models:Linear and random forest regression based on scores and nlp to determine overall rating of coffees
- visualizations and topic_explorations: Creating visualizations throughout the project
- sentiment_analysis: Analyzing reviews for positive, negative, and neutral sentiment using Vader
- recommendations: Building coffee recommendations based on cosine similarity from NMF topics
- coffee_app: Combining predictions, recommendations, and more into a Streamlit app
- lda: A notebook for creating cluster of reviews using Latent Dirichlet Allocation
- kmeans: A notebook for running KMeans clustering on ratings, convectorized corpus, and tf-idf corpus, includes visuals

## In the Visuals folder you will find:  

An assortment of visuals used for the presentation

# Summary of Work and Findings  

## Process

I scraped just under 6000 reviews from [Coffee Review](https://www.coffeereview.com/) that covers their entire history over the last twenty four years. Each review contains details on the roaster, coffee processing, origin, numerical scoring across subcategories and overall, as well as written text reviews. For the purpose of the project, I focused my effort on Natural Language Processing on the "Blind Assessment" portion of a review. This section provides a detailed, adjective heavy review of the coffee drinking experience. 

I implemented topic modeling using Non-negative Matrix Factorization on a TF-IDF embedding of the blind review corpus to turn each review into scores across, ultimately, a nine dimensional flavor vector. These topics give a quick description of the flavor and experience of drinking a coffee without respect to origin, price, or roaster. This is especially important as I was interested in identifying recommendations for coffee based on its flavor and not the history of coffee bean. 

## Recommender

Once every coffee review had been converted into a NMF vector, recommendations between coffees were made based on cosine similarity of their respective vectors. This approach can be applied to descriptions outside of the original corpus as well to find similar coffees to any description of a coffee that may be given as input. 

## Prediction Models

I was also interested to see if I could predice the score assignments to a coffee based solely on the blind assessment. To do this, I consider a number of potential features including: TF-IDF embedding, NMF vectors, sentiment analysis, length of review, and number of unique words in the review. I focused my work on linear models, for interpretability, and Random Forest Regressors, for improved performance, to create such a model.

## Text Generation

As a last step and an item for future consideration, I also created a small model to create a computer generated review based on a subset of coffee reviews. A user can select a subset of coffee reviews based on their majority assignment by the NMF topic model and a simple Markov model generates a short review.

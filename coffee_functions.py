#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import re
from IPython.core.display import display, HTML
from bs4 import BeautifulSoup
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.request import Request, urlopen
import pickle
from collections import defaultdict


# In[2]:


def get_urls():
    # URL for specific Coffee
    search = 'https://www.coffeereview.com/advanced-search/'
    req = Request(search, headers={'User-Agent': 'Mozilla/5.0'})

    # Read into a soup variable
    webpage = urlopen(req).read()
    searchsoup = BeautifulSoup(webpage, 'html')

    tables = searchsoup.find_all('div',attrs={'class':'review-template'})

    url_list =[]
    for i in range(len(tables)):
        url_list = url_list + ['https://www.coffeereview.com'+ tables[i].find('div',attrs={'class':'row row-3'}).find('a').get('href')]

    for i in range(2,299):
        temp_search = 'https://www.coffeereview.com/advanced-search/?pg='+str(i)
        temp_req = Request(temp_search, headers={'User-Agent': 'Mozilla/5.0'})

        # Read into a soup variable
        temp_webpage = urlopen(temp_req).read()
        temp_soup = BeautifulSoup(temp_webpage, 'html')

        temp_tables = temp_soup.find_all('div',attrs={'class':'review-template'})

        for j in range(len(temp_tables)):
            url_list = url_list + ['https://www.coffeereview.com'+ temp_tables[j].find('div',attrs={'class':'row row-3'}).find('a').get('href')]
    return url_list


# In[3]:


def get_coffee(url_list): 
    coffee = pd.DataFrame()
    for link in url_list:
        req = Request(link, headers={'User-Agent': 'Mozilla/5.0'})

        # Read into a soup variable
        webpage = urlopen(req).read()
        soup = BeautifulSoup(webpage, 'html')
        try:
            roaster = soup.find('div',attrs = {'class': 'review-template'}).find_all('p')[0].text
        except AttributeError:
            roaster = ''
        try:
            review = soup.find('div',attrs = {'class': 'review-template'}).find_all('p')[1].text.partition(': ')[2]
        except AttributeError:
            review = ''
        try:
            notes = soup.find('div',attrs = {'class': 'review-template'}).find_all('p')[2].text.partition(': ')[2]
        except AttributeError:
            notes = ''
        try:
            tldr = soup.find('div',attrs = {'class': 'review-template'}).find_all('p')[3].text.partition(': ')[2]
        except AttributeError:
            tldr = ''

        temp_dict = defaultdict(str)
        temp_dict[roaster] = [review,notes, tldr]

        coffee = pd.concat([coffee,pd.DataFrame.from_dict(temp_dict, orient='index',columns=['Review', 'Notes', 'TLDR'])])
    return coffee


# In[4]:


def get_ratings(url_list):
    ratings = pd.DataFrame()
    for link in url_list:
        req = Request(link, headers={'User-Agent': 'Mozilla/5.0'})

        # Read into a soup variable
        webpage = urlopen(req).read()
        soup = BeautifulSoup(webpage, 'html')
        try:
            roaster = soup.find('div',attrs = {'class': 'review-template'}).find_all('p')[0].text
        except AttributeError:
            roaster = ''

        # Scrape the categories and values for each
        # Overall score
        num = soup.find('div', attrs={'class':'column col-1'}).find('span', attrs={'class':'review-template-rating'}).text
        # Categories
        categories = soup.find_all('table',attrs = {'class':'review-template-table'})[0].find_all('td')[0::2]+soup.find_all('table',attrs = {'class':'review-template-table'})[1].find_all('td')[0::2]
        # Scores and values
        scores = soup.find_all('table',attrs = {'class':'review-template-table'})[0].find_all('td')[1::2]+soup.find_all('table',attrs = {'class':'review-template-table'})[1].find_all('td')[1::2]
        y = ['Overall']+[cat.get_text().partition(':')[0] for cat in categories]
        x = [num]+[score.get_text() for score in scores]

        temp_dict = defaultdict(str)
        temp_dict[roaster] = x

        ratings = pd.concat([ratings,pd.DataFrame.from_dict(temp_dict, orient='index',columns=y)])
    return ratings


# In[ ]:





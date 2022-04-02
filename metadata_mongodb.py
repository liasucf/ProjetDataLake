#!/usr/bin/env python
# coding: utf-8

# In[115]:


import os

import findspark

findspark.init()

import numpy as np
import pyspark
from pyspark.sql import SparkSession
from datetime import datetime
from tqdm import tqdm
from pymongo import MongoClient

from nltk.corpus import stopwords
import string
import re
import nltk 
from nltk import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import matplotlib.pyplot as plt

from wordcloud import WordCloud



# ----------------------- Metadata Intra-Donnés ---------------------

#Setting configuration variables
nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])



conf = pyspark.conf.SparkConf().setAll([
                                   ('spark.executorEnv.OMP_NUM_THREADS', '8'),
                                   ('spark.workerEnv.OMP_NUM_THREADS', '8'),
                                   ('spark.executorEnv.OPENBLAS_NUM_THREADS', '8'),
                                   ('spark.workerEnv.OPENBLAS_NUM_THREADS', '8'),
                                   ('spark.executorEnv.MKL_NUM_THREADS', '8'),
                                   ('spark.workerEnv.MKL_NUM_THREADS', '8'),
                                   ])

spark = SparkSession.builder.config(conf=conf).getOrCreate()

#MONGODB 

#Connecting with the MongoClient
url_mongo = "mongodb+srv://admin:admin@cluster0.wjk47.mongodb.net/WikepediaMetadatas?ssl=true&ssl_cert_reqs=CERT_NONE"
client = MongoClient(url_mongo)
#create database 
db = client["WikepediaMetadatas"]
   
#------------- RAW METADATA
#Create collection for raw document metadata mapping and adding the intra-metadata information inside the
#collection

Collection = db["intra_metadata_raw"]

path_to_json = 'enwiki-mini/'
index = 1
for pos_json in tqdm(os.listdir(path_to_json)):
    if pos_json.endswith('.json'):
        path = 'enwiki-mini/' + str(pos_json)
        df = spark.read.option("multiline","true").json(path).toPandas()
        docs = [[w.lower() for w in word_tokenize(text)] 
            for text in list(df['text'])]
        bag_of_words = [item for sublist in docs for item in sublist]
        word_fd = nltk.FreqDist(bag_of_words).most_common(10)
        word_list = [x[0] for x in word_fd]

        data = {
          "id": list(df['id']), 
            #Technical metadata
           "intra-dataset-metadata": [ {
            "properties": [
            {"file_name": pos_json,
            "file_size" : str(round(os.path.getsize(path)/(1024))) + " MB",
            "creation_date": datetime.fromtimestamp(os.path.getctime(path)).strftime('%Y-%m-%d %H:%M:%S') ,
            "sensitivity_level": "low",
             "title": df['title'][0], 
            "document_type" : "text", 
             "language": "english", 
             "number_of_words" : list(df['text'].apply(lambda x: len(x.split())))[0]
              }], 
             "previzualization": [ {
            #previsualization metadata
             "keywords" : word_list 
             }],
            "version": [ {
                "transformation" : "Original version",
                "presentation" : "raw format"
            }]
               #tfid-cleaned
           }]
          
        }
        Collection.insert_one(data)
        index = index + 1


#------------- CLeaned Metadata

#Loading the stopwords
stop_words = stopwords.words('english')
stopwords_en = set(stop_words)

#creating function to remove stopwords, pontuation, digits ...
def cleanup_text(msg):
    #removing pontuation
    No_Punctuation = [char if char not in string.punctuation else ' ' for char in msg ]
    sentence = ''.join(No_Punctuation)
    #remove all non latin caracters
    sentence = re.sub(r'[^\x00-\x7f]',r'', sentence)
    #removing digits
    sentence = re.sub("\S*\d+\S*", "", sentence)
    #### Word tokenization is the process of splitting up “sentences” into “words”
    #sentence = nltk.word_tokenize(sentence)
    #Lemmatizing the words
    sentence = nlp(sentence)
    #lemmetazer = WordNetLemmatizer()
    return " ".join([token.lemma_ for token in sentence if token not in stopwords_en])

#Cleaning the raw documents and creating another set of data with the treated texts
#Save in the Windows Filesystem the documents transformed with the cleaned texts

path_to_json = 'enwiki-mini/'
for pos_json in tqdm(os.listdir(path_to_json)[4:]):
    if pos_json.endswith('.json'):
        path = 'enwiki-mini/' + str(pos_json)
        df = spark.read.option("multiline","true").json(path).toPandas()
        df['text_clean'] = df['text'].apply(lambda x:cleanup_text(x))
        df_clean = df[['id', 'text_clean', 'title' ]]
        df_clean.to_json(r'enwiki-mini-clean/' + str(pos_json.replace('.json', ''))+'-clean.json', orient='records')

#Create collection for cleaned document metadata mapping and 
#adding the intra-metadata information inside the collection

Collection = db["intra_metadata_cleaned"]

path_to_json = 'enwiki-mini-clean/'
for pos_json in tqdm(os.listdir(path_to_json)):
    if pos_json.endswith('.json'):
        path = 'enwiki-mini-clean/' + str(pos_json)
        df = spark.read.option("multiline","true").json(path).toPandas()
        docs = [[w.lower() for w in word_tokenize(text) if w.lower() not in stopwords_en] 
            for text in list(df['text_clean'])]
        bag_of_words = [item for sublist in docs for item in sublist]
        word_fd = nltk.FreqDist(bag_of_words).most_common(10)
        word_list = [x[0] for x in word_fd]

        data = {
          "id": list(df['id']), 
            #Technical metadata
           "intra-dataset-metadata": [ {
            "properties": [
            {"file_name": pos_json,
            "file_size" : str(round(os.path.getsize(path)/(1024))) + " MB",
            "creation_date": datetime.fromtimestamp(os.path.getctime(path)).strftime('%Y-%m-%d %H:%M:%S') ,
            "sensitivity_level": "low",
             "title": df['title'][0], 
            "document_type" : "text", 
             "language": "english", 
             "number_of_words" : list(df['text_clean'].apply(lambda x: len(x.split())))[0]
              }], 
             "previzualization": [ {
            #previsualization metadata
             "keywords" : word_list 
             }],
            "version": [ {
                "transformation" : "Lemmatized version",
                "presentation" : "raw format"
            }]
               #tfid-cleaned
           }]
          
        }
        Collection.insert_one(data)


#------------- tfidf Metadata

#Agregate all the words in the documents and create a tfidf index 


texts = []
path_to_json = 'enwiki-mini-clean/'
for pos_json in tqdm(os.listdir(path_to_json)):
    if pos_json.endswith('.json'):
        path = 'enwiki-mini-clean/' + str(pos_json)
        df = spark.read.option("multiline","true").json(path).toPandas()
        texts.append(list(df['text_clean']))
        
docs = [[w.lower() for w in text if w.lower() not in stopwords_en] 
            for text in texts]
bag_of_docs = [item for sublist in docs for item in sublist]  
#Train TFIDF
vectorizer = TfidfVectorizer(max_features=300)
vectorizer.fit(bag_of_docs)        

#Save in the Windows Filesystem the documents transformed with the tfidf vector representations

path_to_json = 'enwiki-mini-clean/'
for pos_json in tqdm(os.listdir(path_to_json)):
    if pos_json.endswith('.json'):
        path = 'enwiki-mini-clean/' + str(pos_json)
        df = spark.read.option("multiline","true").json(path).toPandas()
        vectors = vectorizer.transform(list(df['text_clean']))
        dense = vectors.todense()
        df['vectors'] = [dense]
        df_tfidf = df[['id', 'vectors']]
        df_tfidf.to_json(r'enwiki-mini-tfidf/' + str(pos_json.replace('.json', ''))+'-tfidf.json', orient='records')
        
#Create collection for tfidf document metadata mapping and 
#adding the intra-metadata information inside the collection

Collection = db["intra_metadata_tfidf"]

path_to_json = 'enwiki-mini-tfidf/'
index = 1
for pos_json in tqdm(os.listdir(path_to_json)):
    if pos_json.endswith('.json'):
        path = 'enwiki-mini-tfidf/' + str(pos_json)
        tfid = spark.read.option("multiline","true").json(path).toPandas()

        data = {
          "id": list(df['id']), 
            #Technical metadata
           "intra-dataset-metadata": [ {
            "properties": [
            {"file_name": pos_json,
            "file_size" : str(round(os.path.getsize(path)/(1024))) + " MB",
            "creation_date": datetime.fromtimestamp(os.path.getctime(path)).strftime('%Y-%m-%d %H:%M:%S') ,
            "sensitivity_level": "low",
            "document_type" : "numeric", 
             "language": "english", 
              }], 
             "previzualization": [ {
            #previsualization metadata
             "keywords" : None 
             }],
            "version": [ {
                "transformation" : "Lemmatized version",
                "presentation" : "tf-idf vector format"
            }]
               #tfid-cleaned
           }]
          
        }
        Collection.insert_one(data)
        index = index + 1


# ----------------------- Metadata Globale ---------------------


#get the tokenizer dictionary and save inside a collection in MongoDB
dictionary = dict((k, int(str(v))) for k,v in vectorizer.vocabulary_.items())


#Create collection for global metadata mapping and 
#adding the vocabulary inside the collection

Collection = db["global_metadata"]

data = {
    #Technical metadata
   "global-dataset-metadata": 
    [{ 
    "tokenizer_dictionary":
    [
        dictionary,
     
    ] 
   }]

}
Collection.insert_one(data)


# ## ---------Textual Analysis

#get the bag of words from all the documents
words =  [[w.lower() for w in nltk.word_tokenize(sentence) if w.lower() not in stopwords_en] 
            for sentence in bag_of_docs]
bag_of_words = [item for sublist in words for item in sublist]  

#Plot the wordCloud
plt.figure(figsize=(9,18))
word_cloud = WordCloud(background_color='black',
                          max_font_size = 80
                         ).generate(" ".join(bag_of_words))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()

#Bar plot for the top 30 most common words
x=[]
y=[]

counter = Counter(bag_of_words)
most = counter.most_common()

for word,count in most[:30]:
        x.append(word)
        y.append(count)
plt.figure(figsize=(8,4))
plt.xticks(fontsize=18, rotation=90)

my_cmap = plt.get_cmap("viridis")
rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

plt.bar(x, y, color=my_cmap(rescale(y)))
plt.show()


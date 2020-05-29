

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import re
import numpy as np 
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import os
import spacy
from collections import Counter
from nltk.tokenize import word_tokenize
from cleaning import load_data
trainFile, testFile = "../cache/corpus.tache1.learn.utf8", "../cache/corpus.tache1.test.utf8"
train, test = load_data(trainFile, testFile)
# generate the lemmatized datasets
nlp = spacy.load("fr_core_news_sm")
free_stopwords, lemmatized, free_stopwords_lemmatized = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
lemmatized["label"] = train["label"].copy()
free_stopwords["label"] = train["label"].copy()
free_stopwords_lemmatized["label"] = train["label"].copy()
lemmatized["text"] = " "
free_stopwords["text"] = ""
free_stopwords_lemmatized["text"] = ""

for i, row in train.iterrows(): # do it all at once
  doc = nlp(row["text"])
  lemmatized.at[i, "text"] = " ".join([i.lemma_ for i in doc])
  free_stopwords.at[i, "text"] =" ".join([ i.text for i in doc if not i.is_stop])
  free_stopwords_lemmatized.at[i, "text"] =" ".join([ i.lemma_ for i in doc if not i.is_stop])
  if i % 1000 ==0:
    print("at row : " + str(i))

free_stopwords_lemmatized.to_csv (r'../cache/lemmatized_free_stopwords.csv', index = False, header=True)
free_stopwords.to_csv (r'../cache/free_stopwords.csv', index = False, header=True)
lemmatized.to_csv (r'../cache/lemmatized.csv', index = False, header=True)

#Stemmed data

stemmer = FrenchStemmer()
stemmed = train["text"].apply(lambda x : " ".join([stemmer.stem(i) for i in word_tokenize(x)]))
stemmed.to_csv ("../cache/stemmed.csv", index = False, header=True) # saving it

#Stemmed data set without the stop words

stemmer = FrenchStemmer()
stemmed_free_stopWords = free_stopwords["text"].apply(lambda x : " ".join([stemmer.stem(i) for i in word_tokenize(x)]))

stemmed_free_stopWords.to_csv ("../cache/stemmed_free_stopWords.csv", index = False, header=True)

#pos_tags 

#the possible pos_ values in spacy
POS_LIST = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
pos_tag = pd.DataFrame()
pos_tag["label"] = train["label"]
for k in POS_LIST:
  pos_tag[k] = 0
pos_tag["length"] = train["text"].apply(lambda x : len(x.split())) # adding a column for the length of the sentences
for i, row in train.iterrows():
  d = dict(Counter([i.pos_ for i in nlp(row["text"])])) # counting the number of each type
  for k, v in d.items():
    pos_tag.at[i, k] = v
pos_tag.to_csv ("../cache/pos_tags.csv", index = False, header=True) # save it in cache


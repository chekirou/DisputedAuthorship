
import numpy as np 
import pandas as pd
from nltk.corpus import stopwords
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from Dataset_Preparation import *
class DataSet:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.nlp = spacy.load("fr_core_new_sm")
        pass
    def CountVectoriser(self):
        count_vect = CountVectorizer(analyzer='word', tokenizer = self.tokenizer, stop_words=stopwords.words("french"), ngram_range= (1,2), strip_accents='unicode' )
        count_vect.fit(self.train['text'])
        # transform the training and validation data using count vectorizer object
        train_count =  count_vect.transform(self.train["text"])
        test_count =  count_vect.transform(self.test["text"])
        return train_count, test_count
    def tf_idf(self):
        # word level tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', preprocessor=clean, stop_words=stopwords.words("french"), ngram_range= (1,2), strip_accents='unicode' ,max_features=5000)
        tfidf_vect.fit(self.train['text'])
        train_tfidf =  tfidf_vect.transform(self.train["text"])
        test_tfidf =  tfidf_vect.transform(self.test["text"])
        return train_tfidf, test_tfidf
    def Lematize(self):
        
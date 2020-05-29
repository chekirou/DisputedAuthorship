#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
from Feature_engineering import *
from cleaning import * 
from sklearn.metrics import f1_score, make_scorer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# In[5]:


trainFile, testFile = "../data/corpus.tache1.learn.utf8", "../data/corpus.tache1.test.utf8"


# In[32]:


train, test = load_data(trainFile, testFile)


# In[33]:


dataClass = DataSet()
df = train#dataClass.lemmatisation()
df.fillna("", inplace = True)


# In[19]:


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
class Classifier(BaseEstimator):

    def __init__(self):
        self.naiveBayes = MultinomialNB()
        pass

    def fit(self, X, y):

        # Check that X and y have correct shape
        # Return the classifier
        self.naiveBayes.fit(X, y)
        return self

    def predict(self, X):
        
        prediction = self.naiveBayes.predict(X)
        results = pd.DataFrame()
        results["prediction_raw"]= prediction
        #closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        results["prediction_1"] = 0
        for i, row in results.iterrows():
            results.at[i, "prediction_1"] = 1 if (results.loc[i-5: i+5, "prediction_raw"]).mean() >= 0.5 else 0
        results["prediction_2"] = 0
        for i , row in results.iterrows():
            results.at[i, "prediction_2"] = 1 if (results.loc[i-1: i+1, "prediction_1"]).mean() >= 0.5 else 0
        return results["prediction_2"].to_numpy()




clf = MultinomialNB()
count_vect = CountVectorizer(analyzer='word',lowercase = False ,ngram_range= (1,2))
count_vect.fit(df["text"])
train_count =  count_vect.transform(df["text"])
Scores = cross_val_score( clf, train_count, df["label"], cv=5, scoring=make_scorer(f1_score,pos_label=1))





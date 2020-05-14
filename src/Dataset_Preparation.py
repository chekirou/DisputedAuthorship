from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import re
import numpy as np 
import pandas as pd
from nltk.corpus import stopwords
#import xgboost ,textblob, string
#from keras.preprocessing import text, sequence
#from keras import layers, models, optimizers

def clean(sentence):
    #remplace punctuation by blanck
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = {c: " " for c in filters}
    translate_map = str.maketrans(translate_dict)
    return sentence.translate(translate_map)


def load_data(train_file, test_file):
    train = pd.DataFrame(columns=["text", "label"])
    train_texts, train_labels = [], []
    print("Loading : train data")
    with open(train_file, "r") as f:
        for line in f:
            line = line.strip()
            m = re.match("<\d+:\d+:(\w)> (.+)", line)
            train_labels.append(1 if m.group(1)=="M" else -1 )
            train_texts.append(m.group(2))
    test = pd.DataFrame(columns=["text"])
    test_texts = []
    print("Loading : test data")
    with open(test_file, "r") as f:
        for line in f:
            line = line.strip()
            m = re.match("<.+> (.+)", line)
            test_texts.append(m.group(1))
    train["text"] = train_texts
    train["label"] = train_labels
    test["text"] = test_texts
    encoder = preprocessing.LabelEncoder()
    train["label"] = encoder.fit_transform(train["label"])
    return train, test

class DataSet:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        pass
    def CountVectoriser(self):
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', preprocessor=clean, stop_words=stopwords.words("french"), ngram_range= (1,2), strip_accents='unicode' )
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




























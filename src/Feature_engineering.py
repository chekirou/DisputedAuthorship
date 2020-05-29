from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas as pd

class DataSet:
    def __init__(self):
        pass
    def stemming(self):
        stemmed = pd.DataFrame()
        try:
            stemmed = pd.read_csv("../cache/stemmed.csv")
        except:
            print("stemmed file not generated, execute Feature_generation.py")
        return stemmed
    def lemmatisation(self):
        lemmatized = pd.DataFrame()
        try:
            lemmatized = pd.read_csv("../cache/lemmatized.csv")
        except:
            print("lemmatized file not generated, execute Feature_generation.py")
        return lemmatized
    def CountVectorizer(self, df,  maxFeatures= None,NGram= (1,1)):
        count_vect = CountVectorizer(analyzer='word',lowercase = False, ngram_range= NGram, max_features= maxFeatures )
        count_vect.fit(df)
        train_count =  count_vect.transform(df)
        return train_count
    def tf_idf(self, df, maxFeatures= None,NGram= (1,1)):
        tfidf_vect = TfidfVectorizer(analyzer='word',ngram_range= NGram, max_features= maxFeatures)
        tfidf_vect.fit(df)
        train_tfidf =  tfidf_vect.transform(df)
        return train_tfidf
    def clean(sentence):
        #remplacer la ponctuation par des vides
        filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        translate_dict = {c: "" for c in filters}
        translate_map = str.maketrans(translate_dict)
        return sentence.translate(translate_map)
    def stop_words(self,mode = "lemmatized"):
        if mode == "lemmatized":
            lemmatized_free_stopWords = pd.DataFrame()
            try:
                lemmatized_free_stopWords = pd.read_csv("../cache/lemmatized_free_stopwords.csv")
            except:
                print("lemmatized free of stopwords file not generated, execute Feature_generation.py")
            return lemmatized_free_stopWords
        elif mode == "stemmed":
            stemmed_free_stopWords = pd.DataFrame()
            try:
                stemmed_free_stopWords = pd.read_csv("../cache/stemmed_free_stopWords.csv")
            except:
                print("stemmed free of stopwords file not generated, execute Feature_generation.py")
            return stemmed_free_stopWords
        else:
            free_stopWords = pd.DataFrame()
            try:
                free_stopWords = pd.read_csv("../cache/free_stopwords.csv")
            except:
                print("train free of stop words not generated, execute Feature_generation.py")
            return free_stopWords
    def pos_tag(self):
        df = pd.DataFrame()
        try:
        	df = pd.read_csv("../cache/PosTags.csv")
        except:
        	print("pos tags file not generated, execute Feature_generation.py")
        return df


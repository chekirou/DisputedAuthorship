# -*- coding: utf-8 -*-
"""author : Hakim Chekirou
	used this script to generate the files stored in cache
"""

import re
import numpy as np 
import pandas as pd
from sklearn import preprocessing

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








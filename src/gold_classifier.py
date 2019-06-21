import os
import pandas as pd
import numpy as np
# machine learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from chinese_tokenizer.tokenizer import Tokenizer

import stanfordnlp
from stanfordnlp.utils.resources import DEFAULT_MODEL_DIR

# load data
datpath = os.path.join("..", "dat")
fname = "gold_standard.xlsx"
df = pd.read_excel(os.path.join(datpath, fname))
colname_content = "微博正文"
colname_targets = ["愤怒","厌恶","悲伤","恐惧","快乐","惊讶","类别其他"]
targets = list()
for colname in colname_targets:
    targets.append(df[colname].values)
posts = df[colname_content].values



## Target variable
# 0: no emotional content; 1: simple emotional content; 2: complex emotional content
targets = np.array(targets).T

#Y = np.zeros((targets.shape[0], ))
Y = list()
for i, v in enumerate(targets):
    if np.sum(v) == 0:
        Y.append("null")
    elif np.sum(v) == 1:
        Y.append("simple")
    elif np.sum(v) > 1:
        Y.append("blended")

data = pd.DataFrame()
data["texts"] = posts
data["class"] = Y

# split data
ratio = 0.8
mask = np.random.rand(len(data)) <= ratio


train = data[mask]
test = data[~mask]

## training set
X_train = train["texts"].values
y_train = train["class"].values
## test set
X_test = test["texts"].values
y_test = test["class"].values


#############################
# instantiate
jie_ba_tokenizer = Tokenizer().jie_ba_tokenizer
vectorizer = CountVectorizer(tokenizer=jie_ba_tokenizer, lowercase=True, min_df=.01, max_features=1000)



feat_train = vectorizer.fit_transform(X_train)
feat_test =  vectorizer.transform(X_test)
feat_names = vectorizer.get_feature_names()

classifier = MultinomialNB()
classifier.fit(feat_train, y_train)


# EVALUATION
pred = classifier.predict(feat_test)
confmat = metrics.confusion_matrix(y_test, pred)# horizontal: predicted label; vertical: true label
# obeserved accuracy
print("Accurracy: {}".format(round(metrics.accuracy_score(y_test, pred),2)))

# cohen's kappa
print("K: {}".format(metrics.cohen_kappa_score(y_test, pred)))
# model summary
print(metrics.classification_report(y_test, pred))

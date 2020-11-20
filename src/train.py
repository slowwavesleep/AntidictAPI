import pandas as pd
import numpy as np
from string import ascii_lowercase
from sklearn.linear_model import SGDClassifier
import gensim
import pickle


loanwords_1 = pd.read_csv('../data/slovar_edited.csv')
loanwords_1.columns = ['word']
loanwords_1.word = loanwords_1.word.apply(lambda x: x.lower().
                                          lstrip('(').rstrip(')').
                                          rstrip('»').lstrip('«').
                                          strip(' '))
loanwords_1 = loanwords_1.loc[loanwords_1.word.str.len() > 2]
loanwords_1 = loanwords_1.drop_duplicates()
loanwords_1 = loanwords_1.loc[~loanwords_1.word.apply(lambda x: bool(set(x).intersection(set(ascii_lowercase))))]
loanwords_1['is_loanword'] = 1
loanwords_2 = pd.read_csv('../data/forms.csv').sample(10_000)
loanwords_2.columns = ['word']
loanwords_2['is_loanword'] = 0
loanwords = data = pd.concat([loanwords_1, loanwords_2])

model = gensim.models.KeyedVectors.load('../models/fasttext/araneum_none_fasttextcbow_300_5_2018.model')

X = data['word']
y = data.iloc[:, 1:].values.ravel()

clf = SGDClassifier()

X_vec = X.apply(lambda x: model[x])
X_vec = np.vstack(X_vec.values)

clf.fit(X_vec, y)

with open('../models/classifier.pkl', 'wb') as file:
    pickle.dump(clf, file)




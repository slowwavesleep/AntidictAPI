import pickle
import gensim
from razdel import tokenize

with open('models/classifier.pkl', 'rb') as file:
    clf = pickle.load(file)

model = gensim.models.KeyedVectors.load('models/fasttext/araneum_none_fasttextcbow_300_5_2018.model')


def predict(text: str):
    tokens = tokenize(text)
    tokens = [token.text.lower() for token in tokens]
    tokens = [(token, clf.predict([model[token]]).tolist()[0]) for token in tokens]
    tokens = [{'word': token, 'is_expressive': pred} for token, pred in tokens]
    return tokens

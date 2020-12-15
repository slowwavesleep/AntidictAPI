import pickle
import gensim
from razdel import tokenize
import regex
import stopwordsiso
from typing import List, Union, Dict, Any, Set

stops = set("""чей свой из-за вполне вообще вроде сюда аж той
россия россии россию россией путин путина путину путиным путине
даю даешь дает даем даете дают""".split())
stops = stops | stopwordsiso.stopwords("ru")

with open("models/classifier.pkl", "rb") as file:
    loanword_clf = pickle.load(file)
with open("models/cb_classifier.pkl", "rb") as file:
    obscene_clf = pickle.load(file)
with open("models/expressive_classifier.pkl", "rb") as file:
    expressive_clf = pickle.load(file)

model = gensim.models.KeyedVectors.load("models/fasttext/araneum_none_fasttextcbow_300_5_2018.model")


def statistics(analysis: List[dict]) -> dict:
    total = len(analysis)
    loanword = len([t for t in analysis if t["is_loanword"]])
    obscene = len([t for t in analysis if t["is_obscene"]])
    expressive = len([t for t in analysis
                      if (t["is_obscene"] or t["is_expressive"])])
    stats = {"loanword_ratio": loanword,
             "obscene_ratio": obscene,
             "expressive_ratio": expressive}
    return {k: round(v / total, 2) for k, v in stats.items()}


def is_word(token: str, min_len: int, max_len: int, s_words: Set[str]) -> bool:
    t = token.lower()
    min_max = str(min_len) + ',' + str(max_len)
    return regex.fullmatch(r"[а-яё\-]{" + min_max + "}", t) and (t not in s_words)


def predict(text: str) -> Dict[str, Union[List[Dict[str, Any]], dict]]:
    tokens = [t.text for t in tokenize(text)]
    cache = {t: {"is_loanword": 0, "is_obscene": 0, "is_expressive": 0}
             for t in set(t.lower() for t in tokens)}
    for t in cache:
        if is_word(t, min_len=3, max_len=30, s_words=stops):
            cache[t]["emb"] = model[t]
            cache[t]["is_loanword"] = loanword_clf.predict([cache[t]["emb"]]).item()
            cache[t]["is_obscene"] = obscene_clf.predict([cache[t]["emb"]]).item()
            cache[t]["is_expressive"] = expressive_clf.predict([cache[t]["emb"]]).item()
    analysis = [{'word': t,
                 "is_loanword": cache[t.lower()]["is_loanword"],
                 "is_obscene": cache[t.lower()]["is_obscene"],
                 "is_expressive": cache[t.lower()]["is_expressive"]
                 } for t in tokens]
    return {"analysis": analysis, "statistics": statistics(analysis)}

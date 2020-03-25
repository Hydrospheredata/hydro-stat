import pickle
import re, string, random
import gensim.downloader as api
from fse import IndexedList
from fse.models import SIF
import numpy as np
from typing import List
import nltk
from utils.utils import fix_path
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
fix_path()

f = open('models/sentiment_analyzer.provectus', 'rb')
classifier = pickle.load(f)
f.close()


def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


class Preprocessor:

    def __init__(self):
        nltk.download('punkt')
        self.stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is',
                           'it', 'its',
                           'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}
        self.ps = nltk.stem.PorterStemmer()

    def tokenize(self, text):
        # TODO word tokenize text using nltk lib
        return word_tokenize(text)

    def stem(self, word, stemmer):
        # TODO stem word using provided stemmer
        return stemmer.stem(word)

    def is_apt_word(self, word):
        # TODO check if word is appropriate - not a stop word and isalpha,
        # i.e consists of letters, not punctuation, numbers, dates
        word = str(word).lower()
        for letter in word:
            if not (letter >= 'a' and letter <= 'z'):
                return False
        # punctiations = ['.','?',',','!',',',':',';','[',']','{','}','(',')','']
        return word not in self.stop_words

    def preprocess(self, text):
        # TODO combine all previous methods together: tokenize lowercased text
        # and stem it, ignoring not appropriate words
        result = []
        for word in self.tokenize(text):
            word = prep.stem(word, self.ps)
            if self.is_apt_word(word):
                result.append(word)
        return result


prep = Preprocessor()


class SentenceEmbed:
    def __init__(self, data):
        # data = api.load("quora-duplicate-questions")
        glove = api.load("glove-wiki-gigaword-100")
        s = IndexedList(data)
        self.model = SIF(glove, workers=8)
        self.model.train(s)

    def infer(self, setences):
        return self.model.infer(setences)


def pos_tagging(sentence):
    res = pos_tag(prep.preprocess(sentence))
    result = []
    for r in res:
        result.append(r[2])
    return result


def sentence_avg_word_lengths(sentences: List[str]):
    def sentence_to_len(sentence):
        res = np.array(list(map(len, sentence)), dtype=np.int32)
        return res.mean()

    res = list(map(prep.preprocess, sentences))
    print(res)
    # return res
    return np.array(list(map(len, res)), dtype=np.int32), np.array(list(map(sentence_to_len, res)), dtype=np.int32)


def sentiment(sentence):
    custom_tokens = remove_noise(word_tokenize(sentence))

    return classifier.classify(dict([token, True] for token in custom_tokens))


if __name__ == '__main__':
    sentence = 'you are a good person :)'
    print(sentiment(sentence))

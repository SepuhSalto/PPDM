import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tokenization import tokenize_word
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stemmed_words = [stemmer.stem(word) for word in tokenize_word]

print("Stemmed Words:", stemmed_words)



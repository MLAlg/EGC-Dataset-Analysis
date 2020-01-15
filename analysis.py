# Prepare Environment
import sys
colab = 'google.colab' in sys.modules
# Download the dataset from my drive(fixed format issue)
if colab:
  !wget 'https://drive.google.com/uc?authuser=0&id=1rseU8HjF16lq87CjVtVCLbhrUCqt_lzi&export=download' -O "EGC_dataset.csv"

#imports
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import FrenchStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF, LatentDirichletAllocation
from math import floor
import pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import keras
from sklearn.decomposition import NMF, LatentDirichletAllocation

nltk.download('stopwords')
nltk.download('punkt')
stopword = set(stopwords.words('french'))
porter = PorterStemmer()
snowball_stemmer = FrenchStemmer()
print("french stop words: ", stopword)

# Read Data
path = "/content/EGC_dataset.csv"
df = pd.read_csv(path)
df.head()

# Find Top Authors
df.authors
df.authors[0]

# prepare authors field
authors = df.authors.str.split(',')
result =  [list(map(str.strip, sublist)) for sublist in authors]
flattened_authors = [item for sublist in result for item in sublist]
for i in range(1269):
  for j in range(len(authors[i])):
    authors[i][j] = authors[i][j].lower()
authors

# mapper
dict_aut = {}
for aut in flattened_authors:
  aut = aut.lower()
  dict_aut[aut] = dict_aut.get(aut,0) + 1
len(dict_aut) # number of authors: 2007
dict_aut # dictionary of authors and their contributions

# reducer
import operator
sorted_aut = sorted(dict_aut.items(), key=operator.itemgetter(1), reverse=True)
sorted_aut[0:11] # top authors

map_aut = []
for aut in flattened_authors:
  map_aut.append((aut, 1))
#map_aut

# Titles of articles for every author
dict_art = {}
for i in range(1269):
  temp = dict.fromkeys(authors[i], df.title[i])
  x = temp.keys()
  for a in x:
    if not dict_art.get(a):
      temp[a] = temp[a]
    else:
      s = dict_art.get(a)
      temp[a] += ", " + s
  dict_art.update(temp)
dict_art



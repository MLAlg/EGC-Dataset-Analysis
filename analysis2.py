import sys
colab = 'google.colab' in sys.modules

# Download the dataset from my drive(fixed format issue)
if colab:
  !wget 'https://drive.google.com/uc?authuser=0&id=1rseU8HjF16lq87CjVtVCLbhrUCqt_lzi&export=download' -O "EGC_dataset.csv"

# English
if colab:
  !wget 'https://drive.google.com/uc?authuser=0&id=1G_9zLZG9e0nn6sLxMzt7fy440J2BOuJh&export=download' -O "en_EGC_dataset.csv"

# French
if colab:
  !wget 'https://drive.google.com/uc?authuser=0&id=1-pC4rfLhk7nYLijAVj68ShZpaHarxnqJ&export=download' -O "fr_EGC_dataset.csv"

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
import keras

from sklearn.decomposition import NMF, LatentDirichletAllocation

nltk.download('stopwords')
nltk.download('punkt')
stopword = set(stopwords.words('french'))
porter = PorterStemmer()
snowball_stemmer = FrenchStemmer()
print("french stop words: ", stopword)

path = "/content/EGC_dataset.csv"
df = pd.read_csv(path)
df.head()


def tokenize(text):
    # print('abstract: ', text)
    # print('abstract: ', type(text))
    # clean false data
    if not isinstance(text, str):
        # print('continue')
        return
    words = word_tokenize(text)  # split words

    # words = text.split() #split words

    words = [w.lower() for w in words if w.isalpha()]
    words = [w for w in words if not w in stopword]
    stemmed = [snowball_stemmer.stem(w) for w in words]
    # # stemmed = [porter.stem(w) for w in words]

    # # stemmed = words
    # # try lemmatizing instead?
    stemmed = ' '.join(w for w in stemmed)
    return stemmed
    # return text


def filtered_token(text):
    # print('abstract: ', text)
    # print('abstract: ', type(text))
    # clean false data
    if not isinstance(text, str):
        # print('continue')
        return
    words = word_tokenize(text)  # split words

    # words = text.split() #split words

    words = [w.lower() for w in words if w.isalpha()]
    words = [w for w in words if not w in stopword]

    return words

df['abstract_cleaned'] = df.abstract.map(tokenize)
df.head()

df = df.dropna()


def get_mat(df, n_topics=10, n_words=20, stopwords='english'):
    tfidf = TfidfVectorizer(max_features=1000, stop_words=stopwords)
    result = tfidf.fit_transform(df)
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method='online',
                                    learning_offset=50.)
    lda.fit(result)

    return result

mat = get_mat(df.abstract_cleaned)
print(mat.shape)
df.shape

from sklearn.cluster import KMeans
n_clusters = 5
km = KMeans(n_clusters=n_clusters)
km.fit(mat)
clusters = km.labels_.tolist()

from sklearn.externals import joblib
#joblib.dump(km,  'doc_cluster.pkl') # Save model
km = joblib.load('doc_cluster.pkl') # Load model
clusters = km.labels_.tolist()

# Add cluster number to a new column
df['cluster'] = clusters
print(df['cluster'].value_counts()) #number of films per cluster (clusters from 0 to 4)

df.head()

# Top terms in abstract per cluster
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def get_topics(df, n_topics=10, n_words=20):
    tfidf = TfidfVectorizer(max_features=1000, stop_words=stopword)

    list_text = df.to_list()
    result = tfidf.fit_transform(list_text)
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method='online',
                                    learning_offset=50.)
    lda.fit(result)
    tf_feature_names = tfidf.get_feature_names()
    print_top_words(lda, tf_feature_names, n_words)
    return result

i = 0
while i < n_clusters:
  get_topics(df[df.cluster==i].abstract_cleaned, n_topics = 2, n_words= 3)
  i+=1

df[df.cluster==4].head()

print(np.unique(df.series))
print(np.unique(df.booktitle))

# Detect language of abstracts
#!pip install langdetect
from langdetect import detect
language = []
for i in range(df.shape[0]):
  language.append(detect(df.abstract.iloc[i]))
languageset = set(language)
print(languageset)

idx_en = []
for i in range(df.shape[0]):
  if language[i] == 'en':
    idx_en.append(i)

idx_fr = []
for i in range(df.shape[0]):
  if language[i] == 'fr':
    idx_fr.append(i)

print(len(idx_en))
print(idx_en)

df_en = df.iloc[idx_en]
df_fr = df.iloc[idx_fr]
print(df_en.iloc[0])

# Save english ones to csv
df_en.to_csv('en_EGC_dataset.csv', index=False)

# Save french ones to csv
df_fr.to_csv('fr_EGC_dataset.csv', index=False)

# Hierarchial clustering
np.unique(df.year)
# For years 2017,2018
from scipy.cluster.hierarchy import ward, dendrogram, linkage
from sklearn.metrics.pairwise import cosine_similarity

df_year = df[df.year > 2016]
mat = get_mat(df_year.abstract_cleaned)
print(mat.shape)
dist = 1 - cosine_similarity(mat)
linkage_matrix = linkage(dist, 'complete') # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
fig, ax = plt.subplots(figsize=(15, 20))
ax = dendrogram(linkage_matrix, orientation="right", above_threshold_color='#bcbddc')

plt.tick_params(axis= 'x',which='both',bottom='off',top='off',labelbottom='off')

plt.show()

linkage_matrix.shape
mat.shape





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

# Prepare Dataset
path = "/content/EGC_dataset.csv"
df = pd.read_csv(path)
df.head()


def tokenize(text):
  # clean false data
  if not isinstance(text, str):
    return
  words = word_tokenize(text)  # split words

  words = [w.lower() for w in words if w.isalpha()]
  words = [w for w in words if not w in stopword]
  # stemmed = [snowball_stemmer.stem(w) for w in words]
  # # stemmed = [porter.stem(w) for w in words]

  stemmed = words
  # # try lemmatizing instead?
  stemmed = ' '.join(w for w in stemmed)
  return stemmed
  # return text

df['abstract_cleaned'] = df.abstract.map(tokenize)
df.head()

# Exploratory Data Analysis
# How many authors we have ?
def get_unique_authors(df):
  authors = df.authors.str.split(',')
  result =  [list(map(str.strip, sublist)) for sublist in authors for item in sublist]
  flattened_authors = [item for sublist in result for item in sublist]
  unique_authors = np.unique(np.array(flattened_authors))
  return unique_authors, authors

# In all years
unique_authors, authors = get_unique_authors(df)
print(unique_authors)
num_classes = len(unique_authors)
print("# of unique authors in all years: ", num_classes)

# Unique authors per year:
years = np.unique(df.year)
frequency_dic = {}
to_be_averaged = 0
for year in years:
  df_year = df.loc[df.year==year]
  uni_auth, auth = get_unique_authors(df_year)
  len_unique_authors = len(uni_auth)
  print("year: ", year, " authors: ", len_unique_authors)
  to_be_averaged += len_unique_authors
  frequency_dic[year] = len_unique_authors

plt.bar(list(frequency_dic.keys()), list(frequency_dic.values()), align='center')
# plt.show()
plt.savefig("authors_dis.png")
print("average Autors: ", to_be_averaged / len(years))

# Are there authors that usually publish together?
def one_hot_authors(row, encoder, num_classes):
  row = list(map(str.strip, row))
  encoding = encoder.transform(row)
  one_hot =  keras.utils.to_categorical(encoding, num_classes= num_classes)
  result = np.sum(one_hot,axis=0)
  # result = pd.DataFrame(result)
  return result

le = preprocessing.LabelEncoder()
le.fit(unique_authors)
LabelEncoder()

encoding_matrx = []
for items in authors.iteritems():
    res =  one_hot_authors(items[1], le, num_classes)
    encoding_matrx.append(res)

encoding_matrx = np.array(encoding_matrx)

columns_name = [i for i in range(num_classes)]
columns_name = le.inverse_transform(columns_name)
df_encoded_as_columns = pd.DataFrame(encoding_matrx, columns= columns_name)
encoded_dataset = pd.concat([df, df_encoded_as_columns], axis=1)
encoded_dataset.to_csv("encoded_dataset.csv",  encoding='utf-8')

from matplotlib import pyplot as plt
from matplotlib import cm as cm

threshold_frequency_dic = {}
# fig = plt.figure()
fig = plt.figure(figsize=(20, 20))
for i in range (0, 9):
  df_authors_filtered = df_encoded_as_columns.loc[:, (df_encoded_as_columns.sum(axis=0) >= ((i+1)*2) )]
  print(df_authors_filtered.shape)
  index = ((i+1)*2)
  threshold_frequency_dic[index] =df_authors_filtered.shape[1]
  c = df_authors_filtered.corr().abs()
  # ax1 = subplot(4,1,i)
  ax1 = fig.add_subplot(3,3,i+1)
  ax1.title.set_text("more than " + str(((i+1)*2)) + " articles")
  cmap = cm.get_cmap('jet', 30)
  cax = ax1.imshow(c, interpolation="nearest", cmap=cmap)
  ax1.grid(True)
  # Add colorbar, make sure to specify tick locations to match desired ticklabels
  fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
  # plt.show()

plt.savefig("coorelation.png")

from matplotlib import pyplot as plt
from matplotlib import cm as cm

fig = plt.figure()
threshold_frequency_dic = {}
for i in range (0, 20):
  df_authors_filtered = df_encoded_as_columns.loc[:, (df_encoded_as_columns.sum(axis=0) >= i+1)]
  threshold_frequency_dic[i] =df_authors_filtered.shape[1]

plt.bar(list(threshold_frequency_dic.keys()), list(threshold_frequency_dic.values()),)
# plt.show()
fig.suptitle('Cumulative Sum of # Authors', fontsize=15)
plt.xlabel('threshold articles', fontsize=18)
plt.ylabel('# of authors', fontsize=16)
plt.savefig("cumsum_authors.png")

threshold_frequency_dic

df_authors_filtered = df_encoded_as_columns.loc[:, (df_encoded_as_columns.sum(axis=0) == 8 )]
c = df_authors_filtered.corr().abs()

import seaborn as sns

sns.set(rc={'figure.figsize':(14,14)})
corr = c
sns_plot = sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
fig = sns_plot.get_figure()
fig.savefig("output.png")


def get_duplications(df):
  to_delete = set()
  cols = df.columns
  for i in range(0, df.shape[1]):
    for j in range(0, i + 1):
      to_delete.add((cols[i], cols[j]))
  return to_delete


def correlation_ordered(df):
  authors_corr = df.corr().abs().unstack()
  labels_to_drop = get_duplications(df)
  authors_corr = authors_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
  return authors_corr


print("Top Correlation")
top_correlation = correlation_ordered(df_encoded_as_columns)

top_correlation[:4]

top_correlation.to_csv("correlation.csv")

# df_top_correlation = pd.DataFrame(top_correlation)
df_top_correlation = top_correlation.reset_index()
df_top_correlation = pd.DataFrame(df_top_correlation)
df_top_correlation.columns = ['Author_1', 'Author_2', 'Corr']
print(df_top_correlation.loc[df_top_correlation['Corr'] < 1] [:50])

df.groupby('year').count()['booktitle'].plot(kind='bar')

grouped_articles = df.groupby('year').count()['booktitle']
grouped_articles = grouped_articles.reset_index()
plt.bar(grouped_articles.year.values, grouped_articles.booktitle.values, align='center')
plt.savefig("articles_dist.png")
print("average articles per year: ",  grouped_articles.booktitle.values.mean())

# What are the top topics in all dataset, per authors, per year ?
# Year 2018
def print_top_words(model, feature_names, n_top_words):
  topic_words = ""
  for topic_idx, topic in enumerate(model.components_):
    message = "Topic #%d: " % topic_idx
    message += " ".join([feature_names[i]
                         for i in topic.argsort()[:-n_top_words - 1:-1]])
    topic_words += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
    print(message)
  print()
  return topic_words


def get_topics(df, n_topics=10, n_words=20, stopwords='english'):
  tfidf = TfidfVectorizer(max_features=1000, stop_words=stopwords)
  df = df.dropna()
  list_text = df.to_list()
  result = tfidf.fit_transform(list_text)
  lda = LatentDirichletAllocation(n_components=n_topics, learning_method='online',
                                  learning_offset=50.)
  lda.fit(result)
  tf_feature_names = tfidf.get_feature_names()
  topic_words = print_top_words(lda, tf_feature_names, n_words)
  return topic_words

df_2018 = df[df.year == 2018]
topic_text = get_topics(df_2018.abstract, n_topics = 5, n_words= 3, stopwords =stopword)

df_2018 = df[df.year == 2018]
# df_2018 = df
topic_text = get_topics(df_2018.abstract_cleaned, n_topics = 3, n_words=400 , stopwords =stopword)
topic_text

df_2018 = df_2018.dropna()

wordcloud = WordCloud(width=1600, height=800, max_words=100, stopwords=['contrainte', 'variable', 'outil', 'type', 'domaine', 'cc', 'document', 'rôle', 'utilisateur','pase', 'algorithme', 'variante', 'nécessaire', 'of', 'ainsi','textes', 'ensemble', 'plus','requête', 'problème', 'deux', 'carte', 'basé', 'nouvelle', 'critère', 'résultats', 'résultat', 'celle', 'basée', 'base', 'article', 'données', 'donnée','méthode', 'approche','déjà', 'cependant', 'cet', 'cette', 'système','performance', 'particulier', 'spécifique','spécifiques', 'papier', 'recherche', 'autres', 'entre', 'proposée', 'proposons', 'autre', 'être','modèle', 'moyen'], background_color="white").generate(topic_text)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file("year_2018_wc.png")

# Frequent groups of authos
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

auth_list = df.authors.to_list()
auth_list = [ group.split(',') for group in auth_list]
dataset = [  [s.strip() for s in group] for group in auth_list]
len(dataset)

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df_apriori = pd.DataFrame(te_ary, columns=te.columns_)
df_apriori

frequent_authors = apriori(df_apriori, min_support=0.001, use_colnames=True)
frequent_authors['length'] = frequent_authors['itemsets'].apply(lambda x: len(x))
frequent_authors

frequent_authors[ (frequent_authors['length'] >= 2)  &
                   (frequent_authors['support'] >= 0.004)]












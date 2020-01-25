import sys
colab = 'google.colab' in sys.modules

# Download the dataset from my drive (fixed format issue)
if colab:
  !wget 'https://drive.google.com/uc?authuser=0&id=1rseU8HjF16lq87CjVtVCLbhrUCqt_lzi&export=download' -O "EGC_dataset.csv"

import pandas as pd
path = "/content/EGC_dataset.csv"
df = pd.read_csv(path)
df.head()

# Create a copy of the data [to leave original data unchanged.]
df1 = df.copy()

# Since we found in ExploratoryAnalysis.ipynb that there were missing abstracts,
# we shall drop those missing entries
# Check for empty data
import numpy as np

cols_to_check = ['title','abstract','authors']
for i in cols_to_check:
  index_missing = np.where(df.isnull()[i])
  print('# samples with no', i, ':', len(index_missing[0]))

get_missing_index = np.where(df.isnull()['abstract'])[0]
df1 = df1.drop(get_missing_index)

print('Sample size:', df1.shape)

# Create TFIDF vectors
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 5,max_df = 0.95,max_features = 8000)
tfidf.fit(df1.abstract)
text = tfidf.transform(df1.abstract)

# 5 clusters
from sklearn.cluster import KMeans
n_clusters = 5
km = KMeans(n_clusters=n_clusters)
km.fit(text)
clusters = km.labels_.tolist()

# Add clusters to df2 (copy of df1)
df2 = df1.copy()
df2['cluster'] = clusters
print(df2['cluster'].value_counts())

# Fit an LDA and get top topics per cluster
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

from sklearn.decomposition import LatentDirichletAllocation

def get_topics(df, n_topics = 10, n_words= 20):
  tfidf = TfidfVectorizer(max_features = 1000)
  
  list_text = df.to_list()
  result = tfidf.fit_transform(list_text)
  lda = LatentDirichletAllocation(n_components= n_topics, learning_method='online',
                                learning_offset=50.)
  lda.fit(result)
  tf_feature_names = tfidf.get_feature_names()
  print_top_words(lda, tf_feature_names, n_words)
  return result

i = 0
while i < n_clusters:
  get_topics(df2[df2.cluster==i].abstract, n_topics = 2, n_words= 3)
  i+=1

!pip install langdetect
from langdetect import detect
language = []
for i in range(df1.shape[0]):
  language.append(detect(df1.abstract.iloc[i]))
languageset = set(language)
print(languageset)

# Get indices of documents in either language
idx_en = []
for i in range(df1.shape[0]):
  if language[i] == 'en':
    idx_en.append(i)

idx_fr = []
for i in range(df1.shape[0]):
  if language[i] == 'fr':
    idx_fr.append(i)

print("Number of English papers", len(idx_en))
print("Their indices", idx_en)

# Seperate these
df_en = df1.iloc[idx_en]
df_fr = df1.iloc[idx_fr]
print(df_en.iloc[0]) #

# Save english ones to csv
df_en = df_en.reset_index()
df_en.to_csv('en_EGC_dataset.csv', index=False)

# Save french ones to csv
df_fr = df_fr.reset_index()
df_fr.to_csv('fr_EGC_dataset.csv', index=False)

df_en.head()

df_fr.head()


from wordcloud import WordCloud
wordcloud = WordCloud(max_font_size=100, max_words=800, background_color="white").generate(' '.join(df_fr['abstract']))

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# French stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import FrenchStemmer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
stopword = set(stopwords.words('french'))
porter = PorterStemmer()
snowball_stemmer = FrenchStemmer()
print("french stop words: ", stopword)
print("Number of french stop words: ", len(stopword))

print("partir" in stopword)
print("article" in stopword)

!npm install stopwords-fr

stopword_file = './node_modules/stopwords-fr/stopwords-fr.json'
import json
with open(stopword_file) as f:
  stopword = json.load(f)

stopword=stopword+['donner', 'variable', 'variables', 'data', 'sciences', 'méthode', 'méthodes', 'technique', 'techniques', 'donnes', 'partir', 'article', 'articles', 
                 'algorithme', 'algorithmes', 'approche', 'approches','système', 'papier', 'contribution', 'recherche', 'utiliser', 'données', 'nouvelle', 'proposer',
                 'rechercher', 'méthode', 'utilisér']

print("Updated french stop words: ", stopword)
print("Number of updated french stop words: ", len(stopword))

!python -m spacy download fr
!pip install spacy
!pip install spacy_lefff
import spacy
from spacy_lefff import LefffLemmatizer, POSTagger

nlp = spacy.load('fr')
pos = POSTagger()
french_lemmatizer = LefffLemmatizer(after_melt=True, default=True)
nlp.add_pipe(pos, name='pos', after='parser')
nlp.add_pipe(french_lemmatizer, name='lefff', after='pos')

def tokenize(text):
    if not isinstance(text, str):
      # print('continue')
      return
    #words = word_tokenize(text) #split words
    #words = [w.lower() for w in words if w.isalpha()] 
    #words =[w for w in words if  not w in stopword]
    
    #stemmed = [snowball_stemmer.stem(w) for w in words]
    #lemmatized = [nlp(unicode(w, "utf-8")).french_lemmatizer for w in stemmed]
    words = nlp(text)
    lemmatized = [w._.lefff_lemma for w in words]
    
    return lemmatized
    # return text

def filtered_token(text):

    words = text.split() #split words
    words = [w.lower() for w in words if w.isalpha()] 
    words = [w for w in words if  not w in stopword]

    return words

# Make lowercase and remove stopwords
tmp = df_fr.abstract.map(filtered_token)
df_fr['abstract_cleaned'] = tmp
for i in range(df_fr.shape[0]):
  tmp = ' '.join(map(str, df_fr['abstract_cleaned'][i]))
  df_fr['abstract_cleaned'][i] = tmp

# Lemmatize and again check for stopwords
tmp = df_fr.abstract_cleaned.map(tokenize)
df_fr['abstract_cleaned'] = tmp
for i in range(df_fr.shape[0]):
  tmp = ' '.join(map(str, df_fr['abstract_cleaned'][i]))
  df_fr['abstract_cleaned'][i] = tmp

tmp = df_fr.abstract_cleaned.map(filtered_token)
df_fr['abstract_cleaned'] = tmp

for i in range(df_fr.shape[0]):
  tmp = ' '.join(map(str, df_fr['abstract_cleaned'][i]))
  df_fr['abstract_cleaned'][i] = tmp

df_fr.head()

def get_mat(df, n_topics = 10, n_words= 20):
  tfidf = TfidfVectorizer(max_features = 1000)
  result = tfidf.fit_transform(df)
  
  return result

mat = get_mat(df_fr.abstract_cleaned)
print(mat.shape)

Sum_of_squared_distances = []
K = range(1,30)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(mat)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

from sklearn.cluster import MiniBatchKMeans
def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    sse = []
    
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=2).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
    
    f, ax = plt.subplots(1, 1, figsize=(20,20))
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    
find_optimal_clusters(mat, 90)

# Choose 30 clusters
km = KMeans(n_clusters=30)
km = km.fit(mat)
clusters = km.labels_.tolist()

# Add clusters to df2 (copy of df1)
df_fr['kmean_cluster'] = clusters
print(df_fr['kmean_cluster'].value_counts())

clusters = np.array(clusters, dtype='int')

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm

def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=900, replace=False)
    
    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))
    
    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    
plot_tsne_pca(mat, clusters)

!pip install pyLDAvis
!pip install gensim
from gensim import corpora, models
import gensim
import pyLDAvis.gensim

df_fr.head()

tmp = df_fr.abstract_cleaned.map(filtered_token)
df_fr['abstract_cleaned'] = tmp

from nltk.tokenize import RegexpTokenizer
def docs_preprocessor(docs):
    tokenizer = RegexpTokenizer(r'\w+')

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isdigit()] for doc in docs]
    
    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 3] for doc in docs]
    
    return docs

docs = docs_preprocessor(df_fr.abstract_cleaned)
docs = docs_preprocessor(docs)
#Create Biagram & Trigram Models 
from gensim.models import Phrases
# Add bigrams and trigrams to docs,minimum count 10 means only that appear 10 times or more.
bigram = Phrases(docs, min_count=10)
trigram = Phrases(bigram[docs])

for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)
    for token in trigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

dictionary = corpora.Dictionary(docs)
corpus = [dictionary.doc2bow(text) for text in docs]

for idx, word in dictionary.iteritems():
  print(idx, word)

first_bow_doc = corpus[0]
display(", ".join(df_fr.abstract_cleaned[0]))
# bow representation
for i in range(len(first_bow_doc)):
    print("Word {} (\"{}\") appears {} time.".format(first_bow_doc[i][0],
                                               dictionary[first_bow_doc[i][0]],
                                               first_bow_doc[i][1]))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=15, id2word = dictionary, passes=20)
import pprint
pprint.pprint(ldamodel.top_topics(corpus,topn=5))

from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cloud = WordCloud(stopwords=stopword,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=1000)

topics = ldamodel.show_topics(formatted=False)

fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

from gensim.models.coherencemodel import CoherenceModel

# Compute Coherence Score
coherence_model_lda_npmi = CoherenceModel(model=ldamodel, texts=docs, dictionary=dictionary, coherence='c_npmi')
coherence_model_lda_cv = CoherenceModel(model=ldamodel, texts=docs, dictionary=dictionary, coherence='c_v')
c_npmi_score = coherence_model_lda_npmi.get_coherence()
c_v_score = coherence_model_lda_cv.get_coherence()

print('\n c_npmi: ', c_npmi_score)
print('\n c_v: ', c_v_score)

len(corpus)

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word = dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=docs, start=2, limit=100, step=6)

limit=100; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Number of topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=25, id2word = dictionary, passes=20)
import pprint
pprint.pprint(ldamodel.top_topics(corpus,topn=5))

# Compute Coherence Score
coherence_model_lda_cv = CoherenceModel(model=ldamodel, texts=docs, dictionary=dictionary, coherence='c_v')
c_v_score = coherence_model_lda_cv.get_coherence()
print('\n c_v: ', c_v_score)

# Save model 
ldamodel.save('lda.gensim')

import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary=ldamodel.id2word)
vis

pyLDAvis.save_html(vis, 'ldaVis.html')

pyLDAvis.save_html(vis, './drive/My Drive/mallet/ldaVis.html')

year_list = np.unique(df_fr.year)
yearly_articles = []
for i in year_list:
  yearly_articles.append(df_fr[df_fr.year == i].shape[0])
print('No. of articles per year:', yearly_articles)

dictionary[0]

# Dynamic LDA
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

ldaseq = gensim.models.ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=dictionary, time_slice=yearly_articles, num_topics=25, em_max_iter=3)

# Save model
ldaseq.save('ldaseq.gensim')

topics = ldaseq.dtm_coherence(time=14) #2018
cm_wrapper = CoherenceModel(topics=topics, texts=docs, dictionary=dictionary, coherence='c_v')
print ("C_v topic coherence")
print ("Coherence for 2018 is ", cm_wrapper.get_coherence())

slices = list(range(0,len(year_list)))

def topic_evolution(topic_number,n_terms: int):
  """
  Returns top topics along with probabilities for the specified topic.
  Input:
  topic_number = Between 0 and 24 [total 25 topics]
  n_terms = How many top words to return for the topic

  Output:
  dataframe containing yearly top terms and their probabilities
  """
  out = ldaseq.print_topic_times(topic_number, top_terms=n_terms)
  columns = [None]*2*len(year_list)
  for i in range(len(year_list)):
    columns[i*2] = str(year_list[i])
    columns[i+1] = 'prob_'+str(year_list[i])
  data = pd.DataFrame(index=list(range(n_terms)), columns=year_list)
  j=1
  for i in slices:
    yearly_top_terms = np.array((out[i]))[:,0]
    data[str(year_list[i])] = yearly_top_terms
    yearly_probs = np.array((out[i]))[:,1]
    data['prob_'+str(year_list[i])] = yearly_probs
    j += 2

  return data.dropna(axis=1)

topic_evolution(21, 30)

for i in range(25):
  df = topic_evolution(i, 30)
  csvname = 'topic'+str(i)+'Evolution.csv'
  df.to_csv(csvname, sep=',')

np.where(topic20)

topic10 = topic_evolution(10, 30)

topic10 = topic_evolution(10, 30)
topic10_arbredecision = []
for i in range(len(year_list)):
  col = str(year_list[i])
  idx = np.where(topic10[col]=='arbre_décision')[0]
  tmp = np.array(topic10['prob_'+str(year_list[i])], dtype='float32')
  topic10_arbredecision.append(tmp[idx])

topic10_semantique = []
for i in range(len(year_list)):
  col = str(year_list[i])
  idx = np.where(topic20[col]=='sémantique')[0]
  tmp = np.array(topic20['prob_'+str(year_list[i])], dtype='float32')
  topic10_semantique.append(tmp[idx])

topic1 = topic_evolution(1, 30)
topic1_ensemble = []
for i in range(len(year_list)):
  col = str(year_list[i])
  idx = np.where(topic1[col]=='ensemble')[0]
  tmp = np.array(topic1['prob_'+str(year_list[i])], dtype='float32')
  topic1_ensemble.append(tmp[idx])

topic1_classification = []
for i in range(len(year_list)):
  col = str(year_list[i])
  idx = np.where(topic1[col]=='classification')[0]
  tmp = np.array(topic1['prob_'+str(year_list[i])], dtype='float32')
  topic1_classification.append(tmp[idx])

topic1_carte_cognitif = []
for i in range(len(year_list)):
  col = str(year_list[i])
  idx = np.where(topic1[col]=='carte_cognitif')[0]
  tmp = np.array(topic1['prob_'+str(year_list[i])], dtype='float32')
  topic1_carte_cognitif.append(tmp[idx])

topic5 = topic_evolution(5, 30)
topic5_visualisation = []
for i in range(len(year_list)):
  col = str(year_list[i])
  idx = np.where(topic5[col]=='visualisation')[0]
  tmp = np.array(topic5['prob_'+str(year_list[i])], dtype='float32')
  topic5_visualisation.append(tmp[idx])

topic21 = topic_evolution(21, 30)
topic21_reseausocial = []
for i in range(len(year_list)):
  col = str(year_list[i])
  idx = np.where(topic21[col]=='réseau_social')[0]
  tmp = np.array(topic21['prob_'+str(year_list[i])], dtype='float32')
  topic21_reseausocial.append(tmp[idx])

plt.figure(figsize=(10,9))
plt.plot(year_list, topic10_arbredecision)
plt.plot(year_list, topic10_semantique)
plt.plot(year_list, topic1_ensemble)
plt.plot(year_list, topic1_classification)
plt.plot(year_list, topic1_carte_cognitif)
plt.plot(year_list, topic5_visualisation)
plt.legend(['arbre_décision', 'Semantique', 'Ensemble', 'Classification', 'Carte Cognitif', 
            'Visualisation'], loc='upper left')
plt.title('Term-Evolution')

plt.plot(year_list, topic21_reseausocial)
plt.title('Term-Evolution (Term - réseau_social in Topic 21)')



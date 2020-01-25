import sys
colab = 'google.colab' in sys.modules
# Download the dataset from my drive (fixed format issue)
if colab:
  !wget 'https://drive.google.com/uc?authuser=0&id=1rseU8HjF16lq87CjVtVCLbhrUCqt_lzi&export=download' -O "EGC_dataset.csv"

# Load data
path = "/content/EGC_dataset.csv"

import pandas as pd
df = pd.read_csv(path)
df.head()

list_col = df.columns
print('Data size:', df.shape, '\n')
print('Columns:',list_col)

# Check for empty data
import numpy as np

cols_to_check = ['title','abstract','authors']
for i in cols_to_check:
  index_missing = np.where(df.isnull()[i])
  print('# samples with no', i, ':', len(index_missing[0]))

# Create a seperate dataset removing these rows
df1 = df.copy()
get_missing_index = np.where(df.isnull()['abstract'])[0]
df1 = df1.drop(get_missing_index)

print('Sample size:', df1.shape)

years = np.sort(np.unique(df1.year))
print('Years:', years)

year_dist = df.groupby([df['year']]).agg('count')['title']

sorted_table = np.c_[years,year_dist]
sorted_table = sorted_table[sorted_table[:,1].argsort()]

# Plot the distribution
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.bar(years, year_dist)
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Number of articles per year')

mean_pub = np.mean(year_dist)
print('Mean publications per year:', np.round(mean_pub,2))

def get_unique_authors(df):
  authors = df.authors.str.split(',')
  result =  [list(map(str.strip, sublist)) for sublist in authors for item in sublist] 
  flattened_authors = [item for sublist in result for item in sublist]
  unique_authors = np.unique(np.array(flattened_authors))
  return unique_authors, authors

unique_authors, authors = get_unique_authors(df)
print(unique_authors)
num_classes = len(unique_authors)
print("# of unique authors in all years: ", num_classes)

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
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Unique authors per year')
plt.savefig("authors_dis.png")
print("Average number of authors per year: ", to_be_averaged / len(years))

# Number of authors per paper
def get_authors_pp(df):
  count = []

  for i in range(df.shape[0]):
    authors = len(df.authors[i].split(','))
    count.append(authors)
  return count

authors_pp = get_authors_per_paper(df)
print('Authors per paper (mean over all years):', np.mean(authors_pp))

# Mean authors per paper by year
def get_authors_ppy(df,years):
  count_ = []
  mean_ = []
  for k in years:
    df_year = df[df['year']==k]
    count = 0
    for i in range(df_year.shape[0]):
      authors = len(df_year.authors.iloc[i].split(','))
      count += authors
    mean_authors_pp_k = count/df_year.shape[0]
    count_.append(mean_authors_pp_k)

  return count_

authors_ppy = get_authors_ppy(df,years)
print('Authors per paper (Count per year):', np.round(authors_ppy,2))

plt.scatter(authors_ppy,years)
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Unique authors per year')
plt.savefig("authors_dis.png")


import keras
def one_hot_authors(row, encoder, num_classes):
  row = list(map(str.strip, row))
  encoding = encoder.transform(row)
  one_hot =  keras.utils.to_categorical(encoding, num_classes= num_classes)
  result = np.sum(one_hot,axis=0)
  # result = pd.DataFrame(result)
  return result

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(unique_authors)

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

sns.set(rc={'figure.figsize':(10,5)})
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
        for j in range(0, i+1):
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





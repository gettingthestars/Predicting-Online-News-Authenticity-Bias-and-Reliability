from wordcloud import WordCloud, ImageColorGenerator
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer

#show python pathway
import sys
print(sys.executable)
#/Users/shenzhou/opt/anaconda3/envs/szp3.8/bin/python -m pip install wordcloud

from sklearn.decomposition import LatentDirichletAllocation as LDA
from pyLDAvis import sklearn as sklearn_lda
import pickle
import pyLDAvis
import numpy as np
def plot_10_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
        count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words in real news text ')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()






mostcommon = FreqDist(clean[clean['bias']=='L']['title_clean']).most_common(100)

mostcommon = FreqDist(clean[clean['bias']=='R']['title_clean']).most_common(100)

mostcommon = FreqDist(clean[clean['bias']=='L']['text_clean']).most_common(100)

mostcommon = FreqDist(clean[clean['bias']=='R']['text_clean']).most_common(100)
wordcloud = WordCloud(width=500, height=300, background_color='white').generate(str(mostcommon))
fig = plt.figure(figsize=(6,8), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
# plt.title('Top 100 Most common word in fake news', fontsize=20)
plt.tight_layout(pad=0)
plt.show()

mostcommon = FreqDist(clean[clean['source']=='CNN']['title_clean']).most_common(100)
wordcloud = WordCloud(width=500, height=300, background_color='white').generate(str(mostcommon))
fig = plt.figure(figsize=(6,8), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
# plt.title('Top 100 Most common word in fake news', fontsize=20)
plt.tight_layout(pad=0)
plt.show()

mostcommon = FreqDist(clean[clean['source']=='Fox News']['title_clean']).most_common(100)
wordcloud = WordCloud(width=500, height=300, background_color='white').generate(str(mostcommon))
fig = plt.figure(figsize=(6,8), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
# plt.title('Top 100 Most common word in fake news', fontsize=20)
plt.tight_layout(pad=0)
plt.show()

mostcommon = FreqDist(clean[clean['source']=='Guardian']['title_clean']).most_common(100)
wordcloud = WordCloud(width=500, height=300, background_color='white').generate(str(mostcommon))
fig = plt.figure(figsize=(6,8), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
# plt.title('Top 100 Most common word in fake news', fontsize=20)
plt.tight_layout(pad=0)
plt.show()

mostcommon = FreqDist(clean[clean['source']=='National Review']['title_clean']).most_common(100)
wordcloud = WordCloud(width=500, height=300, background_color='white').generate(str(mostcommon))
fig = plt.figure(figsize=(6,8), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
# plt.title('Top 100 Most common word in fake news', fontsize=20)
plt.tight_layout(pad=0)
plt.show()

mostcommon = FreqDist(clean[clean['source']=='New York Post']['title_clean']).most_common(100)
wordcloud = WordCloud(width=500, height=300, background_color='white').generate(str(mostcommon))
fig = plt.figure(figsize=(6,8), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
# plt.title('Top 100 Most common word in fake news', fontsize=20)
plt.tight_layout(pad=0)
plt.show()

mostcommon = FreqDist(clean[clean['source']=='New York Times']['title_clean']).most_common(100)
wordcloud = WordCloud(width=500, height=300, background_color='white').generate(str(mostcommon))
fig = plt.figure(figsize=(6,8), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
# plt.title('Top 100 Most common word in fake news', fontsize=20)
plt.tight_layout(pad=0)
plt.show()










data_pie = clean['bias'].value_counts().rename_axis('class_id').reset_index(name='class_num')
plt.pie(data_pie.class_num, labels=['Right','Left'], autopct='%1.0f%%',startangle=90)
plt.title('Distribution of news')
plt.show()

data_pie = clean['source'].value_counts().rename_axis('class_id').reset_index(name='class_num')
plt.pie(data_pie.class_num, labels=['Guardian','CNN','New York Post','New York Times','National Review ','Fox News '], autopct='%1.0f%%',startangle=90)
plt.title('Distribution of news')
plt.show()

data_pie = clean['reliability'].value_counts().rename_axis('class_id').reset_index(name='class_num')
plt.pie(data_pie.class_num, labels=['A','C','B'], autopct='%1.0f%%',startangle=90)
plt.title('Distribution of realiability')
plt.show()


fix=pd.read_csv('All_cleaned_fixed.csv')
data_pie = fix['authenticity'].value_counts().rename_axis('class_id').reset_index(name='class_num')
plt.pie(data_pie.class_num, labels=['real','fake'], autopct='%1.0f%%',startangle=90)
plt.title('Distribution of authenticity')
plt.show()



clean=pd.read_csv('Allnews_cleaned.csv')
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
#count_data = count_vectorizer.fit_transform(clean[clean['bias']=='L']['text_clean'])
count_data = count_vectorizer.fit_transform(clean[clean['source']=='CNN']['title_clean'])
# Visualise the 10 most common words
#count_data = count_vectorizer.fit_transform(clean[clean['bias']=='R']['title_clean'])
plot_10_most_common_words(count_data, count_vectorizer)

count_data = count_vectorizer.fit_transform(clean[clean['source']=='Fox News']['title_clean'])
# Visualise the 10 most common words
#count_data = count_vectorizer.fit_transform(clean[clean['bias']=='R']['title_clean'])
plot_10_most_common_words(count_data, count_vectorizer)

count_data = count_vectorizer.fit_transform(clean[clean['source']=='Guardian']['title_clean'])
# Visualise the 10 most common words
#count_data = count_vectorizer.fit_transform(clean[clean['bias']=='R']['title_clean'])
plot_10_most_common_words(count_data, count_vectorizer)

count_data = count_vectorizer.fit_transform(clean[clean['source']=='National Review']['title_clean'])
# Visualise the 10 most common words
#count_data = count_vectorizer.fit_transform(clean[clean['bias']=='R']['title_clean'])
plot_10_most_common_words(count_data, count_vectorizer)

count_data = count_vectorizer.fit_transform(clean[clean['source']=='New York Post']['title_clean'])
# Visualise the 10 most common words
#count_data = count_vectorizer.fit_transform(clean[clean['bias']=='R']['title_clean'])
plot_10_most_common_words(count_data, count_vectorizer)

count_data = count_vectorizer.fit_transform(clean[clean['source']=='New York Times']['title_clean'])
# Visualise the 10 most common words
#count_data = count_vectorizer.fit_transform(clean[clean['bias']=='R']['title_clean'])
plot_10_most_common_words(count_data, count_vectorizer)









count_data = count_vectorizer.fit_transform(clean[clean['source']=='CNN']['text_clean'])
# Visualise the 10 most common words
#count_data = count_vectorizer.fit_transform(clean[clean['bias']=='R']['title_clean'])
plot_10_most_common_words(count_data, count_vectorizer)

count_data = count_vectorizer.fit_transform(clean[clean['source']=='Fox News']['text_clean'])
# Visualise the 10 most common words
#count_data = count_vectorizer.fit_transform(clean[clean['bias']=='R']['title_clean'])
plot_10_most_common_words(count_data, count_vectorizer)

count_data = count_vectorizer.fit_transform(clean[clean['source']=='Guardian']['text_clean'])
# Visualise the 10 most common words
#count_data = count_vectorizer.fit_transform(clean[clean['bias']=='R']['title_clean'])
plot_10_most_common_words(count_data, count_vectorizer)

count_data = count_vectorizer.fit_transform(clean[clean['source']=='National Review']['text_clean'])
# Visualise the 10 most common words
#count_data = count_vectorizer.fit_transform(clean[clean['bias']=='R']['title_clean'])
plot_10_most_common_words(count_data, count_vectorizer)

count_data = count_vectorizer.fit_transform(clean[clean['source']=='New York Post']['text_clean'])
# Visualise the 10 most common words
#count_data = count_vectorizer.fit_transform(clean[clean['bias']=='R']['title_clean'])
plot_10_most_common_words(count_data, count_vectorizer)

count_data = count_vectorizer.fit_transform(clean[clean['source']=='New York Times']['text_clean'])
# Visualise the 10 most common words
#count_data = count_vectorizer.fit_transform(clean[clean['bias']=='R']['title_clean'])
plot_10_most_common_words(count_data, count_vectorizer)


import spacy

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

nlp = spacy.load("en_core_web_sm")

single_achievenment_csv_path = "single_achievement.csv"
single_achievenment_df = pd.read_csv(single_achievenment_csv_path)

print(single_achievenment_df.head())

def lemmatize(text):
    doc = nlp(text)

    tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct)]
    return ' '.join(tokens)

single_achievenment_df['processed_text'] = single_achievenment_df['cleaned_hm'].apply(lambda txt:lemmatize(txt))

text = ''
for row in single_achievenment_df.iterrows():
    text += row[1]['processed_text'] + ' '

wordcloud = WordCloud(stopwords=STOPWORDS,collocations=True).generate(text)
plt.imshow(wordcloud,interpolation='bilInear')
plt.axis('off')
plt.show()


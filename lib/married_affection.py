import pandas as pd
import spacy

import matplotlib.pyplot as plt
import matplotlib as mpl

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

nlp = spacy.load("en_core_web_sm")

married_affection_csv_path = "married_affection.csv"
married_affection_df = pd.read_csv(married_affection_csv_path)

def lemmatize(text):
    doc = nlp(text)

    tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct)]
    return ' '.join(tokens)

married_affection_df['processed_text'] = married_affection_df['cleaned_hm'].apply(lambda txt:lemmatize(txt))

text = ''
for row in married_affection_df.iterrows():
    text += row[1]['processed_text'] + ' '

wordcloud = WordCloud(stopwords=STOPWORDS,collocations=True).generate(text)
plt.imshow(wordcloud,interpolation='bilInear')
plt.axis('off')
plt.show()

# married_affection_df.head()

# X = cv.fit_transform(married_affection_df.processed_text)
# print(type(X))
# print(X)
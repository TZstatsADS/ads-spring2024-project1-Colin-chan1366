import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


cleaned_hm_csv_file = "HappyDB/happydb/data/cleaned_hm.csv"
cleaned_hm_df = pd.read_csv(cleaned_hm_csv_file)

def lemmatize(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc]
    return tokens


text = "went to movies with my friends it was fun"
res = lemmatize(text)
print(res)

def lemmatize(text):
    doc = nlp(text)

    tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct)]
    return ' '.join(tokens)

cleaned_hm_df['processed_text'] = cleaned_hm_df['cleaned_hm'].apply(lambda txt:lemmatize(txt))


X = cv.fit_transform(cleaned_hm_df.processed_text)
print(X)


df_2 = cleaned_hm_df[cleaned_hm_df['ground_truth_category'] == 'achievement']
df_2 = cleaned_hm_df[cleaned_hm_df['ground_truth_category'] == 'affection']

# word2vec


df_2.loc[df_2.ground_truth_category == 'achievement' ,'label'] =0
df_2.loc[df_2.ground_truth_category == 'affection' ,'label'] =1

y = df_2.ground_truth_category.astype(int)

print(len(X))
print(len(y))

clf = MultinomialNB()
clf.fit(X,y)
yhat = clf.predict(X)

print("Accuracy",accuracy_score(y,yhat))

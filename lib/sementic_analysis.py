import spacy
import pandas as pd

from tqdm import tqdm

import eng_spacysentiment
nlp = eng_spacysentiment.load()

chm_csv_path = "./happydb/data/cleaned_hm.csv"

chm_df = pd.read_csv(chm_csv_path)

pos = []
neg = []
neu = []
for row in tqdm(chm_df.iterrows()):
    text = row[1]['cleaned_hm']
    doc = nlp(text)
    pos.append(doc.cats['positive'])
    neg.append(doc.cats['negative'])
    neu.append(doc.cats['neutral'])

chm_df['positive'] = pos
chm_df['negative'] = neg
chm_df['neutral'] = neu


chm_df.to_csv("sementic_cleaned_hm.csv")

# text = "Welcome to Arsenals official YouTube channel Watch as we take you closer and show you the personality of the club"


# 
# print(doc.cats)
# import spacy
import pandas as pd
import eng_spacysentiment

# nlp = spacy.load("en_core_web_sm")

nlp = eng_spacysentiment.load()


cleaned_hm_csv_file = "HappyDB/happydb/data/cleaned_hm.csv"
cleaned_hm_df = pd.read_csv(cleaned_hm_csv_file)

cleaned_hm_df = cleaned_hm_df.dropna(subset=['ground_truth_category'])

for text in list(cleaned_hm_df['cleaned_hm'].values):
    doc = nlp(text)
    print(doc.cats)
    break

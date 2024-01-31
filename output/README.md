# ADS Project 1: What made you happy today?
### Output folder

The output directory contains analysis output, processed datasets, logs, or other processed things.

## urther data mining Happiness
- To delve deeper, semantic models are needed, which may require time to select the appropriate one.
- A simpler and intuitive approach is to analyze word frequency.
## Married

```python
nouns_list = []
for row in married_affection_df.iterrows():
    text = row[1]['cleaned_hm']
    doc = nlp(text)
    
    nouns = [token.text
         for token in doc
         if (not token.is_stop and
             not token.is_punct and
             token.pos_ == "NOUN")]
    nouns_list.extend(nouns)

counter = Counter(nouns_list)
dictionary=dict(counter)
k=20
res=counter.most_common(k)
print(res)
```

```
[('wife', 339), ('husband', 331), ('daughter', 289), ('son', 278), ('time', 204), ('family', 173), ('day', 168), ('dinner', 137), ('night', 103), ('kids', 94), ('baby', 89), ('birthday', 82), ('work', 80), ('year', 77), ('dog', 70), ('mother', 65), ('morning', 65), ('school', 64), ('sister', 63), ('week', 57)]
```
- The top 5 words are all family members, indicating that family is a key source of happiness.
- "Dinner" and "night" suggest that sharing meals and spending evenings together are sources of happiness and ways to enhance it.
- "Birthday" being a high-frequency word indicates the joy of preparing for birthday parties.
- Time-related words 
- "dog" suggest that adding pets can also enhance family happiness (further analysis needed).

## Single

```python
for row in single_achievenment_df.iterrows():
    text = row[1]['cleaned_hm']
    doc = nlp(text)
    
    nouns = [token.text
         for token in doc
         if (not token.is_stop and
             not token.is_punct and
             token.pos_ == "NOUN")]
    nouns_list.extend(nouns)

counter = Counter(nouns_list)
dictionary=dict(counter)
k=20
res=counter.most_common(k)
print(res)
```
```python
[('work', 272), ('job', 181), ('time', 117), ('money', 97), ('car', 94), ('today', 85), ('day', 80), 
('months', 58), ('week', 58), ('game', 55), ('school', 51), ('project', 47), ('lot', 45), ('goal', 45), 
('event', 42), ('month', 39), ('house', 39), ('years', 38), ('college', 36), ('bonus', 35)]
```
- "Money" and "bonus" are indicators of work recognition and sources of happiness for singles, highlighting the importance of work achievements.
- The appearance of "work" and "job" as the top words emphasizes the significance of work-related achievements.
- Time-related words suggest a focus on time management.
- "College" and terms like "project" 
- and "goal" indicate the importance of setting and achieving goals (further exploration needed).


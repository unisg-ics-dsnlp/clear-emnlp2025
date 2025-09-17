# script to remove the essays dataset texts from files, to comply with the license
import json
import os

import pandas as pd

FILE_DIR = os.path.dirname(__file__)

annotations1 = os.listdir(os.path.join(FILE_DIR, 'preference_manual_scale', 'annotations', 'annotator1'))
annotations1 = [f for f in annotations1 if f.startswith('essay')]
print(annotations1)

for af1 in annotations1:
    with open(os.path.join(FILE_DIR, 'preference_manual_scale', 'annotations', 'annotator1', af1), 'r') as f:
        data = json.load(f)
    data['text'] = 'removed to comply with license'
    with open(os.path.join(FILE_DIR, 'preference_manual_scale', 'annotations', 'annotator1', af1), 'w') as f:
        json.dump(data, f)

annotations2 = os.listdir(os.path.join(FILE_DIR, 'preference_manual_scale', 'annotations', 'annotator2'))
annotations2 = [f for f in annotations2 if f.startswith('essay')]
print(annotations2)

for af2 in annotations2:
    with open(os.path.join(FILE_DIR, 'preference_manual_scale', 'annotations', 'annotator2', af2), 'r') as f:
        data = json.load(f)
    data['text'] = 'removed to comply with license'
    with open(os.path.join(FILE_DIR, 'preference_manual_scale', 'annotations', 'annotator2', af2), 'w') as f:
        json.dump(data, f)

df = pd.read_csv(os.path.join(FILE_DIR, 'preference_manual_scale', 'essay_t1t2_df.csv'))
# remove text column
df = df.drop(columns=['text1', 'text2'])
df.to_csv(os.path.join(FILE_DIR, 'preference_manual_scale', 'essay_t1t2_df_clean.csv'), index=False)

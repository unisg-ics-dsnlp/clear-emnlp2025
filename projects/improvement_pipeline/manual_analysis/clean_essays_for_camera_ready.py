# script to remove the essays dataset texts from files, to comply with the license
import os

import pandas as pd

FILE_DIR = os.path.dirname(__file__)

df = pd.read_csv(os.path.join(FILE_DIR, 'preference_manual', 'essay_t1t2_df.csv'))

# remove text1 and text2 columns
df = df.drop(columns=['text1', 'text2'])
df.to_csv(os.path.join(FILE_DIR, 'preference_manual', 'essay_t1t2_df_clean.csv'), index=False)

df = pd.read_csv('sampled_data_manual_analysis.csv')
# remove all rows where graph_id starts with essay
df = df[~df['graph_id'].str.startswith('essay')]
df.to_csv('sampled_data_manual_analysis_clean.csv', index=False)
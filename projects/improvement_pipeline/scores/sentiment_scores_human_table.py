import json
import os

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

sentiment_scores_path = os.path.join(CURRENT_FILE_DIR, 'scores_out', 'SENTIMENT_SCORES')
essays_sentiment_path = os.path.join(sentiment_scores_path, 'HUMAN_SENTIMENT__essays_sentiment_sentiment.json')
rev1_sentiment_path = os.path.join(sentiment_scores_path,
                                   'HUMAN_SENTIMENT__human_draft1_original_sentiment_sentiment.json')
rev2_sentiment_path = os.path.join(sentiment_scores_path,
                                   'HUMAN_SENTIMENT__human_draft2_improved_sentiment_sentiment.json')
rev3_sentiment_path = os.path.join(sentiment_scores_path,
                                   'HUMAN_SENTIMENT__human_draft3_improved_sentiment_sentiment.json')
microtexts_sentiment_path = os.path.join(sentiment_scores_path, 'HUMAN_SENTIMENT__microtexts_sentiment_sentiment.json')

with open(essays_sentiment_path, 'r') as f:
    essays_sentiment = json.load(f)

with open(rev1_sentiment_path, 'r') as f:
    rev1_sentiment = json.load(f)

with open(rev2_sentiment_path, 'r') as f:
    rev2_sentiment = json.load(f)

with open(rev3_sentiment_path, 'r') as f:
    rev3_sentiment = json.load(f)

with open(microtexts_sentiment_path, 'r') as f:
    microtexts_sentiment = json.load(f)

print('foo')


polarity_avg_rev1 = round(sum(rev1_sentiment['polarity']) / len(rev1_sentiment['polarity']), 2)
polarity_avg_rev2 = round(sum(rev2_sentiment['polarity']) / len(rev2_sentiment['polarity']), 2)
polarity_avg_rev3 = round(sum(rev3_sentiment['polarity']) / len(rev3_sentiment['polarity']), 2)

polarity_essays = round(sum(essays_sentiment['polarity']) / len(essays_sentiment['polarity']), 2)
polarity_microtexts_en = round(sum(s for s in microtexts_sentiment['polarity'] if s is not None) / len([s for s in microtexts_sentiment['polarity'] if s is not None]), 2)
polarity_microtexts_proba_pos = round(sum(s for s in microtexts_sentiment['german_proba_positive'] if s is not None) / len([s for s in microtexts_sentiment['german_proba_positive'] if s is not None]), 2)
polarity_microtexts_proba_neg = round(sum(s for s in microtexts_sentiment['german_proba_negative'] if s is not None) / len([s for s in microtexts_sentiment['german_proba_negative'] if s is not None]), 2)
polarity_microtexts_proba_neutral = round(sum(s for s in microtexts_sentiment['german_proba_neutral'] if s is not None) / len([s for s in microtexts_sentiment['german_proba_neutral'] if s is not None]), 2)

print('Polarity averages:')


# make df
import pandas as pd

data = {
    'Rev1': [polarity_avg_rev1],
    'Rev2': [polarity_avg_rev2],
    'Rev3': [polarity_avg_rev3],
    'Essays': [polarity_essays],
    'MT (EN)': [polarity_microtexts_en],
    ('MT (DE)', 'Neg'): [polarity_microtexts_proba_neg],
    ('MT (DE)', 'Neutral'): [polarity_microtexts_proba_neutral],
    ('MT (DE)', 'Pos'): [polarity_microtexts_proba_pos],
}

# Create dataframe
df = pd.DataFrame(data)

# To make sure the columns are a MultiIndex where needed
df.columns = pd.MultiIndex.from_tuples(
    [(col if isinstance(col, tuple) else ('', col)) for col in df.columns]
)

print(df.to_latex(index=False, float_format='%.2f'))

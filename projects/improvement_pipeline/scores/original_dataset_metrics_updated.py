import pandas as pd

from projects.improvement_pipeline.improve_arguments import load_microtext_df
from projects.improvement_pipeline.improvement_pipeline import prepare_essays_df, load_arg_rewrite_df, \
    prepare_microtexts_both_df
from projects.text_generation_scores.linguaf_scores import linguaf_scores_single

microtext_df = prepare_microtexts_both_df()
microtext_df['is_german'] = microtext_df['topic_id'].str.endswith('(de)')
microtext_df_german = microtext_df[microtext_df['is_german']]
microtext_df_english = microtext_df[~microtext_df['is_german']]
essays_df = prepare_essays_df()
revisions_df = load_arg_rewrite_df()

avg_len_microtexts_english = microtext_df_english['argument'].apply(len).mean()
avg_len_microtexts_de = microtext_df_english['argument'].apply(len).mean()
avg_len_essays = essays_df['argument'].apply(len).mean()
avg_len_rev1 = revisions_df['draft1'].apply(len).mean()
avg_len_rev2 = revisions_df['draft2'].apply(len).mean()
avg_len_rev3 = revisions_df['draft3'].apply(len).mean()

count_microtexts_english = len(microtext_df_english)
count_microtexts_de = len(microtext_df_german)
count_essays = len(essays_df)
count_rev1 = len(revisions_df['draft1'])
count_rev2 = len(revisions_df['draft2'])
count_rev3 = len(revisions_df['draft3'])

# interesting: sentence_count (averaged by len), avg_sentence_length, avg_words_per_sentence

scores_mt_en = linguaf_scores_single(microtext_df_english['argument'].tolist(), lang='en')
scores_mt_de = linguaf_scores_single(microtext_df_german['argument'].tolist(), lang='de')
scores_essays = linguaf_scores_single(essays_df['argument'].tolist(), lang='en')
scores_rev1 = linguaf_scores_single(revisions_df['draft1'].tolist(), lang='en')
scores_rev2 = linguaf_scores_single(revisions_df['draft2'].tolist(), lang='en')
scores_rev3 = linguaf_scores_single(revisions_df['draft3'].tolist(), lang='en')

avg_sent_count_mt_en = scores_mt_en['sentence_count'] / len(microtext_df_english)
avg_sent_count_mt_de = scores_mt_de['sentence_count'] / len(microtext_df_german)
avg_sent_count_essays = scores_essays['sentence_count'] / len(essays_df)
avg_sent_count_rev1 = scores_rev1['sentence_count'] / len(revisions_df['draft1'])
avg_sent_count_rev2 = scores_rev2['sentence_count'] / len(revisions_df['draft2'])
avg_sent_count_rev3 = scores_rev3['sentence_count']/ len(revisions_df['draft3'])

avg_sent_len_mt_en = scores_mt_en['avg_sentence_length']
avg_sent_len_mt_de = scores_mt_de['avg_sentence_length']
avg_sent_len_essays = scores_essays['avg_sentence_length']
avg_sent_len_rev1 = scores_rev1['avg_sentence_length']
avg_sent_len_rev2 = scores_rev2['avg_sentence_length']
avg_sent_len_rev3 = scores_rev3['avg_sentence_length']

avg_words_per_sent_mt_en = scores_mt_en['avg_words_per_sentence']
avg_words_per_sent_mt_de = scores_mt_de['avg_words_per_sentence']
avg_words_per_sent_essays = scores_essays['avg_words_per_sentence']
avg_words_per_sent_rev1 = scores_rev1['avg_words_per_sentence']
avg_words_per_sent_rev2 = scores_rev2['avg_words_per_sentence']
avg_words_per_sent_rev3 = scores_rev3['avg_words_per_sentence']

data = {
    'Rev1': [avg_len_rev1, count_rev1, avg_sent_count_rev1, avg_sent_len_rev1, avg_words_per_sent_rev1],
    'Rev2': [avg_len_rev2, count_rev2, avg_sent_count_rev2, avg_sent_len_rev2, avg_words_per_sent_rev2],
    'Rev3': [avg_len_rev3, count_rev3, avg_sent_count_rev3, avg_sent_len_rev3, avg_words_per_sent_rev3],
    'Essays': [avg_len_essays, count_essays, avg_sent_count_essays, avg_sent_len_essays, avg_words_per_sent_essays],
    'MT (EN)': [avg_len_microtexts_english, count_microtexts_english, avg_sent_count_mt_en, avg_sent_len_mt_en, avg_words_per_sent_mt_en],
    'MT (DE)': [avg_len_microtexts_de, count_microtexts_de, avg_sent_count_mt_de, avg_sent_len_mt_de, avg_words_per_sent_mt_de],
}

# make df
# row indices should be ['Avg. Length', 'Count']

df = pd.DataFrame(data, index=['Avg. Length', 'Count', 'Avg. Sentence Count', 'Avg. Sentence Length', 'Avg. Words per Sentence'])
print(df.to_latex(index=True, float_format='%.2f'))

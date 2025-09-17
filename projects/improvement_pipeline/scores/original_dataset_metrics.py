import pandas as pd

from projects.improvement_pipeline.improve_arguments import load_microtext_df
from projects.improvement_pipeline.improvement_pipeline import prepare_essays_df, load_arg_rewrite_df, \
    prepare_microtexts_both_df

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

data = {
    'Rev1': [avg_len_rev1, count_rev1],
    'Rev2': [avg_len_rev2, count_rev2],
    'Rev3': [avg_len_rev3, count_rev3],
    'Essays': [avg_len_essays, count_essays],
    'MT (EN)': [avg_len_microtexts_english, count_microtexts_english],
    'MT (DE)': [avg_len_microtexts_de, count_microtexts_de],
}

# make df
# row indices should be ['Avg. Length', 'Count']

df = pd.DataFrame(data, index=['Avg. Length', 'Count'])
print(df.to_latex(index=True, float_format='%.2f'))

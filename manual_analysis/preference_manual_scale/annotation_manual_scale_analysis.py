import json
import os

import numpy as np
import pandas as pd

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATIONS_DIR = os.path.join(FILE_DIR, 'annotations')

if __name__ == "__main__":
    annotator1_files = os.listdir(os.path.join(ANNOTATIONS_DIR, 'annotator1'))
    annotator2_files = os.listdir(os.path.join(ANNOTATIONS_DIR, 'annotator2'))

    annotations_annotator1 = []
    annotations_annotator2 = []
    for f in annotator1_files:
        file_path = os.path.join(ANNOTATIONS_DIR, 'annotator1', f)
        with open(file_path, 'r') as f:
            content = json.load(f)
        source = content['source']
        topic_id = content['topic_id']
        graph_id = content['graph_id']
        text_type = content['text_type']
        _text = content['text']
        rating = content['rating']
        annotations_annotator1.append({
            'source': source,
            'topic_id': topic_id,
            'graph_id': graph_id,
            'text_type': text_type,
            'text': _text,
            'rating': rating
        })
    for f in annotator2_files:
        file_path = os.path.join(ANNOTATIONS_DIR, 'annotator2', f)
        with open(file_path, 'r') as f:
            content = json.load(f)
        source = content['source']
        topic_id = content['topic_id']
        graph_id = content['graph_id']
        text_type = content['text_type']
        _text = content['text']
        rating = content['rating']
        annotations_annotator2.append({
            'source': source,
            'topic_id': topic_id,
            'graph_id': graph_id,
            'text_type': text_type,
            'text': _text,
            'rating': rating
        })
    df_annotator1 = pd.DataFrame(annotations_annotator1)
    df_annotator2 = pd.DataFrame(annotations_annotator2)
    df_annotator2 = df_annotator2.copy()
    df_annotator2['topic_id'] = df_annotator2['topic_id'].str.replace('__', '/')

    # merge dfs on 'source', 'topic_id', 'graph_id', 'text_type', 'text'
    df_merged = pd.merge(df_annotator1, df_annotator2, on=['source', 'topic_id', 'graph_id', 'text_type', 'text'],
                         suffixes=('_annotator1', '_annotator2'))
    # save df_merged to csv
    df_merged.to_csv(os.path.join(FILE_DIR, 'annotations', 'merged_annotations.csv'), index=False)

    # get those in both dfs that DO NOT have a partner/merge
    # df_annotator1_only = df_annotator1[~df_annotator1.set_index(['source', 'topic_id', 'graph_id', 'text_type', 'text']).index.isin(df_merged.set_index(['source', 'topic_id', 'graph_id', 'text_type', 'text']).index)]
    # df_annotator2_only = df_annotator2[~df_annotator2.set_index(['source', 'topic_id', 'graph_id', 'text_type', 'text']).index.isin(df_merged.set_index(['source', 'topic_id', 'graph_id', 'text_type', 'text']).index)]

    # calculate cohens kappa between rating_annotator1 and rating_annotator2
    from sklearn.metrics import cohen_kappa_score

    kappa = cohen_kappa_score(
        df_merged['rating_annotator1'],
        df_merged['rating_annotator2'],
        weights='quadratic'
    )
    print(f"Cohen's kappa: {kappa:.4f}")

    # Get the rating vectors
    r1 = df_merged['rating_annotator1'].values
    r2 = df_merged['rating_annotator2'].values

    # Build a confusion matrix (5x5) for ratings 1-5
    conf_mat = np.zeros((5, 5), dtype=int)
    for a, b in zip(r1, r2):
        conf_mat[a - 1, b - 1] += 1  # 0-indexed

    # Create a custom weight matrix where:
    # Difference ≤ 1 -> perfect agreement (weight 1)
    # Difference > 1 -> disagreement (weight 0)
    weights = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if abs(i - j) <= 1:
                weights[i, j] = 1  # Considered equivalent

    # Compute observed agreement (O)
    total = conf_mat.sum()
    observed_agreement = (conf_mat * weights).sum() / total

    # Compute expected agreement (E)
    row_marginals = conf_mat.sum(axis=1) / total
    col_marginals = conf_mat.sum(axis=0) / total
    expected_agreement = sum(
        weights[i, j] * row_marginals[i] * col_marginals[j]
        for i in range(5) for j in range(5)
    )

    # Tolerant Kappa
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    print(f"Tolerant Kappa (±1 counts as agreement): {kappa:.4f}")

    # do tolerant per source
    tolerant_kappas = {}
    for source in df_merged['source'].unique():
        source_df = df_merged[df_merged['source'] == source]
        r1 = source_df['rating_annotator1'].values
        r2 = source_df['rating_annotator2'].values

        conf_mat = np.zeros((5, 5), dtype=int)
        for a, b in zip(r1, r2):
            conf_mat[a - 1, b - 1] += 1  # 0-indexed

        weights = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                if abs(i - j) <= 1:
                    weights[i, j] = 1  # Considered equivalent

        total = conf_mat.sum()
        observed_agreement = (conf_mat * weights).sum() / total

        row_marginals = conf_mat.sum(axis=1) / total
        col_marginals = conf_mat.sum(axis=0) / total
        expected_agreement = sum(
            weights[i, j] * row_marginals[i] * col_marginals[j]
            for i in range(5) for j in range(5)
        )

        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        tolerant_kappas[source] = kappa
    print(tolerant_kappas)

    # count how often annotator1_rating is higher than annotator2_rating, lower, or equal
    annotator1_higher = (df_merged['rating_annotator1'] > df_merged['rating_annotator2']).sum()
    annotator1_lower = (df_merged['rating_annotator1'] < df_merged['rating_annotator2']).sum()
    annotator1_equal = (df_merged['rating_annotator1'] == df_merged['rating_annotator2']).sum()

    print(f"Annotator 1 higher: {annotator1_higher}")
    print(f"Annotator 1 lower: {annotator1_lower}")
    print(f"Annotator 1 equal: {annotator1_equal}")

    # get those where difference is 2 or bigger
    df_disagreement = df_merged[
        abs(df_merged['rating_annotator1'] - df_merged['rating_annotator2']) >= 2
    ]
    print(df_disagreement)
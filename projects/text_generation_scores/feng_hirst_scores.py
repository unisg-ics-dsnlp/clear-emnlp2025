import json
import os

import pandas as pd
from datasets import tqdm
from feng_hirst_parser.parse import DiscourseParser
from feng_hirst_parser.trees.extract_metrics import extract_metrics, extract_relation_ngrams


def feng_hirst_scores(
        original_texts: list[str],
        improved_texts: list[str],
        output_dir: str,
        identifier: str,
        src_lng: str = 'en',
        tgt_lng: str = 'en',
):
    df = pd.DataFrame()
    for i, (original, improved) in enumerate(zip(original_texts, improved_texts)):
        # workaround to force the parser to clean up
        parser = DiscourseParser(
            False,
            False,
            False,
            False,
            output_dir=output_dir
        )
        try:
            print('> RST parsing original text')
            pt1 = parser.parse_from_text(original, identifier + '_original_' + str(i))
            print('> RST parsing improved text')
            pt2 = parser.parse_from_text(improved, identifier + '_improved_' + str(i))
            print('> Both texts parsed, proceeding to extract metrics')
        except ValueError:
            # add a row to df, where all values are nan
            df.loc[len(df)] = [float('nan')] * len(df.columns)
            continue
        G1 = pt1.to_networkx()
        G2 = pt2.to_networkx()
        metrics_original = extract_metrics(G1, relation_ngrams=[(2, 5)])
        metrics_improved = extract_metrics(G2, relation_ngrams=[(2, 5)])
        row_data = {
            'depth_original': metrics_original['depth'],
            'depth_improved': metrics_improved['depth']
        }

        for concept, count in metrics_original['concept_counts'].items():
            row_data['concept_count_original_' + concept] = count
            row_data['concept_count_improved_' + concept] = metrics_improved['concept_counts'].get(concept, 0)

        for relation, count in metrics_original['relation_counts'].items():
            row_data['relation_count_original_' + relation] = count
            row_data['relation_count_improved_' + relation] = metrics_improved['relation_counts'].get(relation, 0)

        for ngram_size, counts in metrics_original['relation_ngram_counts'].items():
            for ngram, count in counts.items():
                ngram_joined = ';'.join(ngram).replace(' ', '_')
                row_data['relation_ngram_original_' + ngram_joined] = count
                row_data['relation_ngram_improved_' + ngram_joined] = metrics_improved['relation_ngram_counts'][
                    ngram_size].get(ngram, 0)

        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    df.fillna(0, inplace=True)
    return df

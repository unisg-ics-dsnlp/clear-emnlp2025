import json
import os

import pandas as pd

from projects.text_generation_scores.text_mining_classifier_scores import predict_english, predict_german

MODELS = [
    'bloomz-3b-cleaned',
]

APPROACHES = [
    'direct',
]

DATASETS = [
    'ESSAYS',
    'MICROTEXTS',
    'REVISIONS1',
    'REVISIONS2',
    'REVISIONS3',
    'REVISIONS_FEEDBACK1',
    'REVISIONS_FEEDBACK2',
    'REVISIONS_FEEDBACK3'
]

current_file_dir = os.path.dirname(os.path.abspath(__file__))

for dataset in DATASETS:
    comparison_delta_dfs = []
    corpus_dfs = []
    for model in MODELS:
        out_file_name = f'HUMAN__'
        if dataset == 'REVISIONS1':
            file_path = os.path.join(current_file_dir, '..', 'improved_out', 'REVISIONS', f'{model}_revision1.json')
            out_file_name = f'HUMAN_revision1__REVISIONS'
        elif dataset == 'REVISIONS2':
            file_path = os.path.join(current_file_dir, '..', 'improved_out', 'REVISIONS', f'{model}_revision2.json')
            out_file_name = f'HUMAN_revision2__REVISIONS'
        elif dataset == 'REVISIONS3':
            file_path = os.path.join(current_file_dir, '..', 'improved_out', 'REVISIONS', f'{model}_revision3.json')
            out_file_name = f'HUMAN_revision3__REVISIONS'
        elif dataset == 'REVISIONS_FEEDBACK1':
            file_path = os.path.join(current_file_dir, '..', 'improved_out', 'REVISIONS_FEEDBACK', f'{model}_revision1.json')
            out_file_name = f'HUMAN_revision1__REVISIONS_FEEDBACK'
        elif dataset == 'REVISIONS_FEEDBACK2':
            file_path = os.path.join(current_file_dir, '..', 'improved_out', 'REVISIONS_FEEDBACK', f'{model}_revision2.json')
            out_file_name = f'HUMAN_revision2__REVISIONS_FEEDBACK'
        elif dataset == 'REVISIONS_FEEDBACK3':
            file_path = os.path.join(current_file_dir, '..', 'improved_out', 'REVISIONS_FEEDBACK', f'{model}_revision3.json')
            out_file_name = f'HUMAN_revision3__REVISIONS_FEEDBACK'
        else:
            file_path = os.path.join(current_file_dir, '..', 'improved_out', dataset, f'{model}.json')
            out_file_name = f'HUMAN__{dataset}'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.loads(f.read())
            # also intentional - only using first one, its enough for original texts
            for e in data[:1]:
                out_file_name_full = f'{out_file_name}__comparison.csv'
                if os.path.exists(os.path.join(current_file_dir, 'scores_out', 'ARG_MINING_SCORES', out_file_name_full)):
                    print(f'> Skipping {out_file_name_full}')
                    continue
                # using original - intentional for human scores!!
                improved = e['original_arguments']
                out = []
                if dataset == 'MICROTEXTS':
                    english = [ia for ia in improved[:89]]
                    german = [ia for ia in improved[:89]]
                    out.extend(predict_english(english))
                    out.extend(predict_german(german))
                    keys_all = set()
                    for o in out:
                        keys_all.update(o.keys())
                    for o in out:
                        for k in keys_all:
                            if k not in o:
                                o[k] = 0
                else:
                    out.extend(predict_english(improved))
                    keys_all = set()
                    for o in out:
                        keys_all.update(o.keys())
                    for o in out:
                        for k in keys_all:
                            if k not in o:
                                o[k] = 0
                    # make dataframe
                df = pd.DataFrame(out)
                df.to_csv(os.path.join(current_file_dir, 'scores_out', 'ARG_MINING_SCORES', out_file_name_full))
        else:
            print('??')

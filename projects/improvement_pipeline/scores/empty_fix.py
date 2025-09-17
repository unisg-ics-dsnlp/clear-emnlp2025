import json
import os

folders = ['MICROTEXTS', 'ESSAYS', 'REVISIONS', 'REVISIONS', 'REVISIONS']
file_names = ['Phi-3-medium-4k-instruct-cleaned.json', 'Phi-3-medium-4k-instruct-cleaned.json', 'Phi-3-medium-4k-instruct-cleaned_revision1.json', 'Phi-3-medium-4k-instruct-cleaned_revision2.json', 'Phi-3-medium-4k-instruct-cleaned_revision3.json']
model_name = 'microsoft/Phi-3-medium-4k-instruct'

current_file_dir = os.path.dirname(os.path.abspath(__file__))
for folder, file_name in zip(folders, file_names):
    out_name = file_name[:len(file_name) - len('.json')]
    _test_file = os.path.join('../improvement_pipeline/improved_out', folder, file_name)
    with open(_test_file, 'r') as file:
        data = json.load(file)
    for approach in data:
        approach_name = approach['approach']
        original = approach['original_arguments']
        improved = approach['improved_arguments']
        print(approach_name, folder, file_name, len(original), len(improved))

        # get indicies where improved.strip() is empty
        empty_improved = [i for i, x in enumerate(improved) if not x.strip()]
        # exclude those from original and improved
        original = [x for i, x in enumerate(original) if i not in empty_improved]
        improved = [x for i, x in enumerate(improved) if i not in empty_improved]

        out_dir = os.path.join(current_file_dir, 'empty_fix')
        out_file_name = f'{out_name}__{folder}__{approach_name}_empty_json'

        # save empty ones
        with open(os.path.join(out_dir, out_file_name), 'w') as f:
            json.dump(empty_improved, f, indent=4)

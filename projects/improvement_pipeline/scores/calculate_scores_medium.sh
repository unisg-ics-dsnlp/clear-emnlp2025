#!/bin/bash

export PYTHONPATH=/data/protefrastuff
PYTHON_SCRIPT_PATH="/data/protefrastuff/projects/improvement_pipeline/scores/calculate_scores2.py"
export LLAMA_OPENAI_ENDPOINT='SET_THIS'
# remember to echo the llama api key!!

FOLDERS=(
  'MICROTEXTS'
  'MICROTEXTS'
  'MICROTEXTS'
  'MICROTEXTS'
  'MICROTEXTS'
  'MICROTEXTS'
  'ESSAYS'
  'ESSAYS'
  'ESSAYS'
  'ESSAYS'
  'ESSAYS'
  'ESSAYS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
  'REVISIONS'
)
FILE_NAMES=(
  'bloomz-3b-cleaned.json'
  'bloomz-560m-cleaned.json'
  'llama3_nemotron_cleaned.json'
  'OLMo-7B-0724-Instruct-hf-cleaned.json'
  'Phi-3-medium-4k-instruct-cleaned.json'
  'Phi-3-mini-4k-instruct-cleaned.json'
  'bloomz-3b-cleaned.json'
  'bloomz-560m-cleaned.json'
  'llama3_nemotron_cleaned.json'
  'OLMo-7B-0724-Instruct-hf-cleaned.json'
  'Phi-3-medium-4k-instruct-cleaned.json'
  'Phi-3-mini-4k-instruct-cleaned.json'
  'bloomz-3b-cleaned_revision1.json'
  'bloomz-3b-cleaned_revision2.json'
  'bloomz-3b-cleaned_revision3.json'
  'bloomz-560m-cleaned_revision1.json'
  'bloomz-560m-cleaned_revision2.json'
  'bloomz-560m-cleaned_revision3.json'
  'llama3_nemotron_cleaned_revision1.json'
  'llama3_nemotron_cleaned_revision2.json'
  'llama3_nemotron_cleaned_revision3.json'
  'OLMo-7B-0724-Instruct-hf-cleaned_revision1.json'
  'OLMo-7B-0724-Instruct-hf-cleaned_revision2.json'
  'OLMo-7B-0724-Instruct-hf-cleaned_revision3.json'
  'Phi-3-medium-4k-instruct-cleaned_revision1.json'
  'Phi-3-medium-4k-instruct-cleaned_revision2.json'
  'Phi-3-medium-4k-instruct-cleaned_revision3.json'
  'Phi-3-mini-4k-instruct-cleaned_revision1.json'
  'Phi-3-mini-4k-instruct-cleaned_revision2.json'
  'Phi-3-mini-4k-instruct-cleaned_revision3.json'
)
MODEL_NAMES=(
  'bigscience/bloomz-3b'
  'bigscience/bloomz-560m'
  'llama3'
  'allenai/OLMo-7B-0724-Instruct-hf'
  'microsoft/Phi-3-medium-4k-instruct'
  'microsoft/Phi-3-mini-4k-instruct'
  'bigscience/bloomz-3b'
  'bigscience/bloomz-560m'
  'llama3'
  'allenai/OLMo-7B-0724-Instruct-hf'
  'microsoft/Phi-3-medium-4k-instruct'
  'microsoft/Phi-3-mini-4k-instruct'
  'bigscience/bloomz-3b'
  'bigscience/bloomz-3b'
  'bigscience/bloomz-3b'
  'bigscience/bloomz-560m'
  'bigscience/bloomz-560m'
  'bigscience/bloomz-560m'
  'llama3'
  'llama3'
  'llama3'
  'allenai/OLMo-7B-0724-Instruct-hf'
  'allenai/OLMo-7B-0724-Instruct-hf'
  'allenai/OLMo-7B-0724-Instruct-hf'
  'microsoft/Phi-3-medium-4k-instruct'
  'microsoft/Phi-3-medium-4k-instruct'
  'microsoft/Phi-3-medium-4k-instruct'
  'microsoft/Phi-3-mini-4k-instruct'
  'microsoft/Phi-3-mini-4k-instruct'
  'microsoft/Phi-3-mini-4k-instruct'

)
FORMAT_FUNCS=(
  4
  4
  0
  4
  1
  1
  4
  4
  0
  4
  1
  1
  4
  4
  4
  4
  4
  4
  0
  0
  0
  4
  4
  4
  1
  1
  1
  1
  1
  1
)

# Loop through the models and format functions
for i in "${!FOLDERS[@]}"; do
    FOLDER="${FOLDERS[$i]}"
    FILE_NAME="${FILE_NAMES[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    FORMAT_FUNC="${FORMAT_FUNCS[$i]}"

    # Run the Python script with the configurations for this iteration
    python "$PYTHON_SCRIPT_PATH" \
        --folder "$FOLDER" \
        --file-name "$FILE_NAME" \
        --model-name "$MODEL_NAME" \
        --format-func "$FORMAT_FUNC"

    echo "Completed iteration $((i+1)) with model: $MODEL"
done

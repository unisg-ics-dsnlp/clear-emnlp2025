#!/bin/bash

export PYTHONPATH=/data/protefrastuff
PYTHON_SCRIPT_PATH="/data/protefrastuff/projects/improvement_pipeline/improvement_pipeline.py"
export LLAMA_OPENAI_ENDPOINT='SET_THIS'
# remember to echo the llama api key!!

# Define a list of models you want to test
MODELS=(
  'microsoft/Phi-3-small-8k-instruct'
  "microsoft/Phi-3-mini-4k-instruct"
  'microsoft/Phi-3.5-mini-instruct'
  'microsoft/Phi-3-medium-4k-instruct'
  'allenai/OLMo-7B-0724-Instruct-hf'
  "google/gemma-2b-it"
  "google/gemma-7b-it"
  'tiiuae/falcon-7b'
  'tiiuae/falcon-40b-instruct'
  'bigscience/bloomz-560m'
  'bigscience/bloomz-3b'
)

# Define some placeholder paths for the output files
IMPROVED_OUT_FILES=(
  'improved_out/MICROTEXTS/Phi-3-small-8k-instruct-improved.json'
  'improved_out/MICROTEXTS/Phi-3-mini-4k-instruct-improved.json'
  'improved_out/MICROTEXTS/Phi-3.5-mini-instruct-improved.json'
  'improved_out/MICROTEXTS/Phi-3-medium-4k-instruct-improved.json'
  'improved_out/MICROTEXTS/OLMo-7B-0724-Instruct-hf-improved.json'
  'improved_out/MICROTEXTS/gemma-2b-it-improved.json'
  'improved_out/MICROTEXTS/gemma-7b-it-improved.json'
  'improved_out/MICROTEXTS/falcon-7b-improved.json'
  'improved_out/MICROTEXTS/falcon-40b-instruct-improved.json'
  'improved_out/MICROTEXTS/bloomz-560m-improved.json'
  'improved_out/MICROTEXTS/bloomz-3b-improved.json'
)
CLEANED_OUT_FILES=(
  'improved_out/MICROTEXTS/Phi-3-small-8k-instruct-cleaned.json'
  'improved_out/MICROTEXTS/Phi-3-mini-4k-instruct-cleaned.json'
  'improved_out/MICROTEXTS/Phi-3.5-mini-instruct-cleaned.json'
  'improved_out/MICROTEXTS/Phi-3-medium-4k-instruct-cleaned.json'
  'improved_out/MICROTEXTS/OLMo-7B-0724-Instruct-hf-cleaned.json'
  'improved_out/MICROTEXTS/gemma-2b-it-cleaned.json'
  'improved_out/MICROTEXTS/gemma-7b-it-cleaned.json'
  'improved_out/MICROTEXTS/falcon-7b-cleaned.json'
  'improved_out/MICROTEXTS/falcon-40b-instruct-cleaned.json'
  'improved_out/MICROTEXTS/bloomz-560m-cleaned.json'
  'improved_out/MICROTEXTS/bloomz-3b-cleaned.json'
)

# Define format functions to test
FORMAT_FUNCS=(
  1
  1
  1
  1
  4
  3
  3
  4
  4
  4
  4
)

# Optional flags: These are booleans, so you could specify them for each iteration if needed
USE_VLLM_VALS=(
  true
  true
  true
  true
  true
  true
  true
  true
  true
  true
  true
)
FORCE_VLLM_VALS=(
  true
  true
  true
  true
  true
  true
  true
  true
  true
  true
  true
)
TRUST_REMOTE_CODE_VALS=(
  true
  true
  true
  true
  true
  true
  true
  true
  true
  true
  true
)
# Loop through the models and format functions
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    IMPROVED_OUT_FILE="${IMPROVED_OUT_FILES[$i]}"
    CLEANED_OUT_FILE="${CLEANED_OUT_FILES[$i]}"
    FORMAT_FUNC="${FORMAT_FUNCS[$i]}"
    USE_VLLM="${USE_VLLM_VALS[$i]}"
    FORCE_VLLM_PRELOAD="${FORCE_VLLM_VALS[$i]}"
    TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE_VALS[$i]}"

    # Run the Python script with the configurations for this iteration
    python "$PYTHON_SCRIPT_PATH" \
        --model "$MODEL" \
        --improve-out-file-path "$IMPROVED_OUT_FILE" \
        --cleaned-out-file-path "$CLEANED_OUT_FILE" \
        --format-func "$FORMAT_FUNC" \
        $( [ "$USE_VLLM" == true ] && echo "--use-vllm" ) \
        $( [ "$FORCE_VLLM_PRELOAD" == true ] && echo "--force-vllm-preload" ) \
        $( [ "$TRUST_REMOTE_CODE" == true ] && echo "--trust-remote-code" ) \
        --dataset '3'

    echo "Completed iteration $((i+1)) with model: $MODEL"
done

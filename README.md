# CLEAR: A Comprehensive Linguistic Evaluation of Argument Rewriting by Large Language Models

Repository for our paper "CLEAR: A Comprehensive Linguistic Evaluation of Argument Rewriting by Large Language Models", accepted at EMNLP 2025 Findings.

## Abstract

While LLMs have been extensively studied on general text generation tasks, there is less research on text rewriting, a task related to general text generation, and particularly on the behavior of models on this task. In this paper we analyze what changes LLMs make in a text rewriting setting. We focus specifically on argumentative texts and their improvement, a task named Argument Improvement (ArgImp). We present CLEAR: an evaluation pipeline consisting of 57 metrics mapped to four linguistic levels: lexical, syntactic, semantic and pragmatic. This pipeline is used to examine the qualities of LLM-rewritten arguments on a broad set of argumentation corpora and compare the behavior of different LLMs on this task and analyze the behavior of different LLMs on this task in terms of linguistic levels. By taking all four linguistic levels into consideration, we find that the models perform ArgImp by shortening the texts while simultaneously increasing average word length and merging sentences. Overall we note an increase in the persuasion and coherence dimensions.

## Overview

### Manual Analysis

The code for the manual analysis, Section 5.2 of the paper, can be found in the `manual_analysis` folder.

For all annotations for the Argument Annotated Essays 2.0 corpus we do **not** distribute the texts, as per the license. The dataset is available [here](https://tudatalib.ulb.tu-darmstadt.de/items/9177c48c-8bd5-4881-9cb4-0632b5941464). If you want to use this dataset, you need to respect the license. We include our annotations without the texts of the essays.

### Improvement Pipeline

The code for the pipeline is included in `projects/improvement_pipeline`. The scoring scripts for the different metrics are included in `projects/text_generation_scores`.

## License

The Argument Annotated Essays 2.0 is available [here](https://tudatalib.ulb.tu-darmstadt.de/items/9177c48c-8bd5-4881-9cb4-0632b5941464) and may only be used for academic and research purposes. We do not include the dataset in this repository.

The ArgRewrite V.2 corpus is available under the GNU General Public License. The license text is included in the `gpl-3.0.txt` file. We include raw texts in our annotation files. The dataset can be downloaded [here](https://argrewrite.cs.pitt.edu/).

The Microtexts corpus is available under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. The license is included in the `microtexts-LICENSE.txt` file. We do not include the dataset in this repository. Our annotations include the texts.

All other code is licensed under the MIT license. See the `LICENSE` file for details.
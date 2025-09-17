from projects.improvement_pipeline.improvement_pipeline import ArgumentImprovementPipeline, prepare_essays_df, \
    prepare_microtexts_both_df, prepare_revisions_df, load_arg_rewrite_df
from util.template import llama3_format_func

# essay_df = prepare_essays_df()
# pipeline = ArgumentImprovementPipeline(
#     argument_df=essay_df,
#     model='llama3',
#     force_vllm_preload=False,
#     trust_remote_code=False,
#     use_vllm=False,
#     improve_out_file_path='improved_out/llama3_nemotron_improved.json',
#     cleaned_out_file_path='improved_out/llama3_nemotron_cleaned.json',
#     format_func=llama3_format_func,
# )
# pipeline.improve_arguments()
# pipeline.clean_improved_arguments()

# microtexts_df = prepare_microtexts_both_df()
# pipeline = ArgumentImprovementPipeline(
#     argument_df=microtexts_df,
#     model='llama3',
#     force_vllm_preload=False,
#     trust_remote_code=False,
#     use_vllm=False,
#     improve_out_file_path='improved_out/MICROTEXTS/llama3_nemotron_improved.json',
#     cleaned_out_file_path='improved_out/MICROTEXTS/llama3_nemotron_cleaned.json',
#     format_func=llama3_format_func,
# )
# pipeline.improve_arguments()
# pipeline.clean_improved_arguments()

revisions_df = load_arg_rewrite_df()
revisions1_df, revisions2_df, revisions3_df = prepare_revisions_df(revisions_df)
# pipeline = ArgumentImprovementPipeline(
#     argument_df=revisions1_df,
#     model='llama3',
#     force_vllm_preload=False,
#     trust_remote_code=False,
#     use_vllm=False,
#     improve_out_file_path='improved_out/REVISIONS/llama3_nemotron_improved_revision1.json',
#     cleaned_out_file_path='improved_out/REVISIONS/llama3_nemotron_cleaned_revision1.json',
#     format_func=llama3_format_func,
# )
# pipeline.improve_arguments()
# pipeline.clean_improved_arguments()
#
# pipeline = ArgumentImprovementPipeline(
#     argument_df=revisions2_df,
#     model='llama3',
#     force_vllm_preload=False,
#     trust_remote_code=False,
#     use_vllm=False,
#     improve_out_file_path='improved_out/REVISIONS/llama3_nemotron_improved_revision2.json',
#     cleaned_out_file_path='improved_out/REVISIONS/llama3_nemotron_cleaned_revision2.json',
#     format_func=llama3_format_func,
# )
# pipeline.improve_arguments()
# pipeline.clean_improved_arguments()

pipeline = ArgumentImprovementPipeline(
    argument_df=revisions3_df,
    model='llama3',
    force_vllm_preload=False,
    trust_remote_code=False,
    use_vllm=False,
    improve_out_file_path='improved_out/REVISIONS/llama3_nemotron_improved_revision3.json',
    cleaned_out_file_path='improved_out/REVISIONS/llama3_nemotron_cleaned_revision3.json',
    format_func=llama3_format_func,
)
pipeline.improve_arguments()
pipeline.clean_improved_arguments()

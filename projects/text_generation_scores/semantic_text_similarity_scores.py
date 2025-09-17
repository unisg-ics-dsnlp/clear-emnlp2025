import json

from semantic_text_similarity.models import WebBertSimilarity


def semantic_text_similarity_scores(
        original_texts: list[str],
        improved_texts: list[str],
):
    web_model = WebBertSimilarity(device='cuda', batch_size=10)
    web_scores = web_model.predict([(original, improved) for original, improved in zip(original_texts, improved_texts)])
    return web_scores.tolist()

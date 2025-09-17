import os
import pickle
from collections import Counter

from projects.text_generation_scores.english_model.classifiers_components.predictions.predict_linear_SVC import \
    predict as predict_english_svc

from projects.text_generation_scores.german_model.classifiers_components.predictions.predict_linear_SVC2 import \
    predict as predict_german_svc

current_file_dir = os.path.dirname(os.path.abspath(__file__))


def predict_english(
        texts: list[str]
):
    out = []
    with open(os.path.join(current_file_dir, 'english_model', 'models', 'linear_SVC', 'linear_SVC.pkl'), 'rb') as f:
        loaded_model = pickle.load(f)

    # loaded_model = pickle.load(open("../../../english_model/models/linear_SVC/linear_SVC.pkl", 'rb'))
    with open(
            os.path.join(current_file_dir, 'english_model', 'models', 'linear_SVC', 'count_vectorizer_linear_SVC.pkl'),
            'rb') as f:
        loaded_count_vectorizer = pickle.load(f)
    for text in texts:
        try:
            preds = predict_english_svc(text, loaded_model, loaded_count_vectorizer)
        except ZeroDivisionError:
            preds = []
        preds = [x for x in preds if x is not None]
        out.append(dict(Counter(preds)))
    return out


def predict_german(
        texts: list[str]
):
    with open(os.path.join(current_file_dir, 'german_model', 'models', 'linear_SVC', 'linear_SVC2.pkl'), 'rb') as f:
        loaded_model = pickle.load(f)

    # loaded_model = pickle.load(open("../../../english_model/models/linear_SVC/linear_SVC.pkl", 'rb'))
    with open(
            os.path.join(current_file_dir, 'german_model', 'models', 'linear_SVC', 'count_vectorizer_linear_SVC2.pkl'),
            'rb') as f:
        loaded_count_vectorizer = pickle.load(f)
    out = []
    for text in texts:
        try:
            preds = predict_german_svc(text, loaded_model, loaded_count_vectorizer)
        except ZeroDivisionError:
            preds = []
        preds = [x for x in preds if x is not None]
        out.append(dict(Counter(preds)))
    return out

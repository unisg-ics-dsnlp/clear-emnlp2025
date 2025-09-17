"""
Bertalign initialization
"""

__author__ = 'Jason (bfsujason@163.com)'
__version__ = '1.1.0'

from projects.text_generation_scores.bertalign_scores.bertalign.encoder import Encoder

# See other cross-lingual embedding models at
# https://www.sbert.net/docs/pretrained_models.html

model_name = 'LaBSE'
model = Encoder(model_name)

from projects.text_generation_scores.bertalign_scores.bertalign.aligner import Bertalign
# from bertalign.aligner import Bertalign

import os

from matplotlib import pyplot as plt
from wordcloud import WordCloud

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(CURRENT_FILE_DIR, 'llama3_nemotron_direct_summary.json'), 'r') as f:
    data = f.read()

import json

data = json.loads(data)

wordcloud = WordCloud(width=1260, height=800)
wordcloud.generate('\n'.join(x['analysis'] for x in data))

plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

wordcloud.to_file('llama3_nemotron_direct_summary_wordcloud.png')

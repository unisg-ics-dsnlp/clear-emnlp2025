import os
import xml.etree.ElementTree as ET

import pandas as pd


def parse_microtexts(
        language: str = 'en'
):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    microtexts_path = os.path.join(current_file_dir, 'arg-microtexts', 'corpus', language)
    arguments_out = []
    for file in sorted(os.listdir(microtexts_path)):
        if file.endswith('.xml'):
            file_path = os.path.join(microtexts_path, file)
        else:
            continue

        tree = ET.parse(file_path)  # replace 'path_to_your_file.xml' with the actual file path
        root = tree.getroot()

        topic_id = root.attrib.get('topic_id')
        graph_id = root.attrib.get('id')
        stance = root.attrib.get('stance')

        arguments = []
        for edu in root.findall('edu'):
            edu_id = edu.attrib.get('id')
            edu_text = edu.text.strip()
            arguments.append((edu_id, edu_text))

        argument = ' '.join([text for _, text in arguments])
        arguments_out.append((topic_id, graph_id, stance, argument))

    df = pd.DataFrame(arguments_out, columns=['topic_id', 'graph_id', 'stance', 'argument'])
    df.to_csv(os.path.join(current_file_dir, f'microtexts_{language}.csv'), index=False)


if __name__ == '__main__':
    parse_microtexts('de')

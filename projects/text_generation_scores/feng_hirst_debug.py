import os

from feng_hirst_parser.parse import DiscourseParser

current_file_dir = os.path.dirname(os.path.abspath(__file__))
parser = DiscourseParser(
    False,
    False,
    False,
    False,
    output_dir=os.path.join(current_file_dir, 'DEBUG')
)
problem_text = '3. Hundehaufen auf Gehwegen sind eine echte Gefahr. Immer wieder rutschen Kinder oder ältere Leute darauf aus. Eine Geldstrafe für Haufenhinterlassung mutet zugegeben ziemlich obrigkeitsstaatlich an, aber ähnliche Regelungen gibt es schließlich oft auch für Fahrräder auf Parkwegen. Die Geldstrafen zu erhöhen ist also der richtige Weg. Eine hundehaufenfreie Stadt wird nicht nur für die Bewohner, sondern auch für Touristen viel attraktiver sein!'
pt2 = parser.parse_from_text(problem_text, 'DEBUG_TEXT' + '_improved_')
print(pt2)

import json
from sklearn.metrics import cohen_kappa_score


def calc_pref_ratio(mask, reviewer):
    # if mask is True, then text2 is improved
    # if mask is False, then text1 is improved
    preferred_improved = 0
    total = len(mask)

    for i in range(total):
        improved = 2 if mask[i] else 1  # Which text is improved
        if reviewer[i] == improved:
            preferred_improved += 1

    ratio = preferred_improved / total
    return ratio


def calc_percent_agreement(r1, r2):
    assert len(r1) == len(r2), 'Reviewer lists must be the same length'
    agree = sum(1 for a, b in zip(r1, r2) if a == b)
    return agree / len(r1)


def calculate_pref_ratio():
    with open('essay_mask.json', 'r') as f:
        essay_mask = json.load(f)
    with open('microtext_de_mask.json', 'r') as f:
        microtext_de_mask = json.load(f)
    with open('microtext_en_mask.json', 'r') as f:
        microtext_en_mask = json.load(f)
    with open('revision1_mask.json', 'r') as f:
        revision1_mask = json.load(f)
    with open('revision2_mask.json', 'r') as f:
        revision2_mask = json.load(f)
    with open('revision3_mask.json', 'r') as f:
        revision3_mask = json.load(f)

    # if mask is True, then text2 is improved
    # if mask is False, then text1 is improved
    with open('essay_reviewer1.txt', 'r') as f:
        essay_reviewer1 = [int(line.strip().split(':')[-1]) for line in f if line.strip()]
    with open('mt_de_reviewer1.txt', 'r') as f:
        mt_de_reviewer1 = [int(line.strip().split(':')[-1]) for line in f if line.strip()]
    with open('mt_en_reviewer1.txt', 'r') as f:
        mt_en_reviewer1 = [int(line.strip().split(':')[-1]) for line in f if line.strip()]
    with open('rev1_reviewer1.txt', 'r') as f:
        rev1_reviewer1 = [int(line.strip().split(':')[-1]) for line in f if line.strip()]
    with open('rev2_reviewer1.txt', 'r') as f:
        rev2_reviewer1 = [int(line.strip().split(':')[-1]) for line in f if line.strip()]
    with open('rev3_reviewer1.txt', 'r') as f:
        rev3_reviewer1 = [int(line.strip().split(':')[-1]) for line in f if line.strip()]

    essay_pref_reviewer1 = calc_pref_ratio(essay_mask, essay_reviewer1)
    print(f'Essay preference ratio for reviewer 1: {essay_pref_reviewer1:.2f}')
    mt_de_pref_reviewer1 = calc_pref_ratio(microtext_de_mask, mt_de_reviewer1)
    print(f'Microtext DE preference ratio for reviewer 1: {mt_de_pref_reviewer1:.2f}')
    mt_en_pref_reviewer1 = calc_pref_ratio(microtext_en_mask, mt_en_reviewer1)
    print(f'Microtext EN preference ratio for reviewer 1: {mt_en_pref_reviewer1:.2f}')
    rev1_pref_reviewer1 = calc_pref_ratio(revision1_mask, rev1_reviewer1)
    print(f'Revision 1 preference ratio for reviewer 1: {rev1_pref_reviewer1:.2f}')
    rev2_pref_reviewer1 = calc_pref_ratio(revision2_mask, rev2_reviewer1)
    print(f'Revision 2 preference ratio for reviewer 1: {rev2_pref_reviewer1:.2f}')
    rev3_pref_reviewer1 = calc_pref_ratio(revision3_mask, rev3_reviewer1)
    print(f'Revision 3 preference ratio for reviewer 1: {rev3_pref_reviewer1:.2f}')

    # do total preference, across all
    essay_mask_all = essay_mask + microtext_de_mask + microtext_en_mask + revision1_mask + revision2_mask + revision3_mask
    essay_reviewer1_all = essay_reviewer1 + mt_de_reviewer1 + mt_en_reviewer1 + rev1_reviewer1 + rev2_reviewer1 + rev3_reviewer1
    essay_pref_reviewer1_all = calc_pref_ratio(essay_mask_all, essay_reviewer1_all)
    print(f'All preference ratio for reviewer 1: {essay_pref_reviewer1_all:.2f}')

    # repeat for reviewer 2
    with open('essay_reviewer2.txt', 'r') as f:
        essay_reviewer2 = [int(x) for x in f.read().splitlines()[0].split(',')]
    with open('mt_de_reviewer2.txt', 'r') as f:
        mt_de_reviewer2 = [int(x) for x in f.read().splitlines()[0].split(',')]
    with open('mt_en_reviewer2.txt', 'r') as f:
        mt_en_reviewer2 = [int(x) for x in f.read().splitlines()[0].split(',')]
    with open('rev1_reviewer2.txt', 'r') as f:
        rev1_reviewer2 = [int(x) for x in f.read().splitlines()[0].split(',')]
    with open('rev2_reviewer2.txt', 'r') as f:
        rev2_reviewer2 = [int(x) for x in f.read().splitlines()[0].split(',')]
    with open('rev3_reviewer2.txt', 'r') as f:
        rev3_reviewer2 = [int(x) for x in f.read().splitlines()[0].split(',')]
    essay_pref_reviewer2 = calc_pref_ratio(essay_mask, essay_reviewer2)
    print(f'Essay preference ratio for reviewer 2: {essay_pref_reviewer2:.2f}')
    mt_de_pref_reviewer2 = calc_pref_ratio(microtext_de_mask, mt_de_reviewer2)
    print(f'Microtext DE preference ratio for reviewer 2: {mt_de_pref_reviewer2:.2f}')
    mt_en_pref_reviewer2 = calc_pref_ratio(microtext_en_mask, mt_en_reviewer2)
    print(f'Microtext EN preference ratio for reviewer 2: {mt_en_pref_reviewer2:.2f}')
    rev1_pref_reviewer2 = calc_pref_ratio(revision1_mask, rev1_reviewer2)
    print(f'Revision 1 preference ratio for reviewer 2: {rev1_pref_reviewer2:.2f}')
    rev2_pref_reviewer2 = calc_pref_ratio(revision2_mask, rev2_reviewer2)
    print(f'Revision 2 preference ratio for reviewer 2: {rev2_pref_reviewer2:.2f}')
    rev3_pref_reviewer2 = calc_pref_ratio(revision3_mask, rev3_reviewer2)
    print(f'Revision 3 preference ratio for reviewer 2: {rev3_pref_reviewer2:.2f}')
    # do total preference, across all
    essay_mask_all = essay_mask + microtext_de_mask + microtext_en_mask + revision1_mask + revision2_mask + revision3_mask
    essay_reviewer2_all = essay_reviewer2 + mt_de_reviewer2 + mt_en_reviewer2 + rev1_reviewer2 + rev2_reviewer2 + rev3_reviewer2
    essay_pref_reviewer2_all = calc_pref_ratio(essay_mask_all, essay_reviewer2_all)
    print(f'All preference ratio for reviewer 2: {essay_pref_reviewer2_all:.2f}')

    all_reviewer1 = essay_reviewer1 + mt_de_reviewer1 + mt_en_reviewer1 + rev1_reviewer1 + rev2_reviewer1 + rev3_reviewer1
    all_reviewer2 = essay_reviewer2 + mt_de_reviewer2 + mt_en_reviewer2 + rev1_reviewer2 + rev2_reviewer2 + rev3_reviewer2

    # interrater agreement between reviewers
    # essay_kappa = cohen_kappa_score(essay_reviewer1, essay_reviewer2)
    # print(f'Cohen\'s Kappa for Essay: {essay_kappa:.2f}')
    #
    # mt_de_kappa = cohen_kappa_score(mt_de_reviewer1, mt_de_reviewer2)
    # print(f'Cohen\'s Kappa for Microtext DE: {mt_de_kappa:.2f}')
    #
    # mt_en_kappa = cohen_kappa_score(mt_en_reviewer1, mt_en_reviewer2)
    # print(f'Cohen\'s Kappa for Microtext EN: {mt_en_kappa:.2f}')
    #
    # rev1_kappa = cohen_kappa_score(rev1_reviewer1, rev1_reviewer2)
    # print(f'Cohen\'s Kappa for Revision 1: {rev1_kappa:.2f}')
    #
    # rev2_kappa = cohen_kappa_score(rev2_reviewer1, rev2_reviewer2)
    # print(f'Cohen\'s Kappa for Revision 2: {rev2_kappa:.2f}')
    #
    # rev3_kappa = cohen_kappa_score(rev3_reviewer1, rev3_reviewer2)
    # print(f'Cohen\'s Kappa for Revision 3: {rev3_kappa:.2f}')
    #
    # all_kappa = cohen_kappa_score(all_reviewer1, all_reviewer2)
    # print(f'Cohen\'s Kappa for ALL datasets: {all_kappa:.2f}')

    print('\n--- Percent Agreement ---')
    essay_agreement = calc_percent_agreement(essay_reviewer1, essay_reviewer2)
    print(f'Percent agreement for Essay: {essay_agreement:.2%}')

    mt_de_agreement = calc_percent_agreement(mt_de_reviewer1, mt_de_reviewer2)
    print(f'Percent agreement for Microtext DE: {mt_de_agreement:.2%}')

    mt_en_agreement = calc_percent_agreement(mt_en_reviewer1, mt_en_reviewer2)
    print(f'Percent agreement for Microtext EN: {mt_en_agreement:.2%}')

    rev1_agreement = calc_percent_agreement(rev1_reviewer1, rev1_reviewer2)
    print(f'Percent agreement for Revision 1: {rev1_agreement:.2%}')

    rev2_agreement = calc_percent_agreement(rev2_reviewer1, rev2_reviewer2)
    print(f'Percent agreement for Revision 2: {rev2_agreement:.2%}')

    rev3_agreement = calc_percent_agreement(rev3_reviewer1, rev3_reviewer2)
    print(f'Percent agreement for Revision 3: {rev3_agreement:.2%}')

    all_agreement = calc_percent_agreement(all_reviewer1, all_reviewer2)
    print(f'Percent agreement for ALL datasets: {all_agreement:.2%}')


if __name__ == '__main__':
    calculate_pref_ratio()

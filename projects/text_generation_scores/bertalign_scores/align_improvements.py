from dataclasses import dataclass

from projects.text_generation_scores.bertalign_scores.bertalign.aligner import Bertalign


@dataclass
class BERTAlignScores:
    delete: int
    add: int
    copy: int
    transformation: int
    other: int
    fusion: int
    merge: int
    total: int
    sources: list
    targets: list

    def to_dict(self):
        return {
            'delete': self.delete,
            'add': self.add,
            'copy': self.copy,
            'transformation': self.transformation,
            'other': self.other,
            'fusion': self.fusion,
            'merge': self.merge,
            'total': self.total,
            'sources': self.sources,
            'targets': self.targets
        }


def do_bertalign_metrics(
        original: str,
        improved: str,
        src_lng: str = 'en',
        tgt_lng: str = 'en',
):
    delete = 0
    add = 0
    copy = 0
    transformation = 0
    other = 0
    fusion = 0
    merge = 0
    total = 0
    sources = []
    targets = []
    aligner = Bertalign(original, improved, is_split=False, src_lang=src_lng, tgt_lang=tgt_lng)
    aligner.align_sents()
    for elem in aligner.result:
        len_src = len(elem[0])
        len_tgt = len(elem[1])
        total += 1
        if len_src == len_tgt == 1:
            if elem[0] == elem[1]:
                copy += 1
            else:
                transformation += 1
        elif len_src == len_tgt:
            test = True
            for number in range(0, len(elem[0])):
                if elem[0][number] != elem[1][number]:
                    transformation += 1
                    test = False
                    break
            if test:
                copy += 1
        elif len_src == 0:
            add += 1
        elif len_tgt == 0:
            delete += 1
        elif len_src > len_tgt:
            merge += 1
        elif len_src < len_tgt:
            fusion += 1
        else:
            other += 1

    lines = aligner.result
    # TODO: check if this is correct - it's the same in christinas script
    for l in lines:
        for alignment in lines:
            src = alignment[0]
            tgt = alignment[1]

            so = ''
            ta1 = ''
            for s1 in src:
                so = so + ' ' + aligner.src_sents[s1]
            for t in tgt:
                ta1 = ta1 + ' ' + aligner.tgt_sents[t]

            sources.append(so)
            targets.append(ta1)
    return BERTAlignScores(
        delete=delete,
        add=add,
        copy=copy,
        transformation=transformation,
        other=other,
        fusion=fusion,
        merge=merge,
        total=total,
        sources=sources,
        targets=targets
    )


if __name__ == '__main__':
    orig_text = '''It is always said that competition can effectively promote the development of economy. In order to survive in the competition, companies continue to improve their products and service, and as a result, the whole society prospers. However, when we discuss the issue of competition or cooperation, what we are concerned about is not the whole society, but the development of an individual's whole life. From this point of view, I firmly believe that we should attach more importance to cooperation during primary education.
First of all, through cooperation, children can learn about interpersonal skills which are significant in the future life of all students. What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others. During the process of cooperation, children can learn about how to listen to opinions of others, how to communicate with others, how to think comprehensively, and even how to compromise with other team members when conflicts occurred. All of these skills help them to get on well with other people and will benefit them for the whole life.
On the other hand, the significance of competition is that how to become more excellence to gain the victory. Hence it is always said that competition makes the society more effective. However, when we consider about the question that how to win the game, we always find that we need the cooperation. The greater our goal is, the more competition we need. Take Olympic games which is a form of competition for instance, it is hard to imagine how an athlete could win the game without the training of his or her coach, and the help of other professional staffs such as the people who take care of his diet, and those who are in charge of the medical care. The winner is the athlete but the success belongs to the whole team. Therefore without the cooperation, there would be no victory of competition.
Consequently, no matter from the view of individual development or the relationship between competition and cooperation we can receive the same conclusion that a more cooperative attitudes towards life is more profitable in one's success.'''

    improved_text = '''

Competition and cooperation are two fundamental aspects of life that have a profound impact on an individual's success. While competition can foster excellence and drive societal progress, it is cooperation that cultivates essential interpersonal skills and empowers individuals to achieve shared goals.

Through cooperation, children develop crucial abilities such as teamwork, listening, communication, and compromise. By working together towards a common objective, they learn to appreciate diverse perspectives, navigate conflicts effectively, and build strong relationships. These skills not only benefit their academic pursuits but also equip them with tools for navigating the complexities of life beyond the classroom.

On the other hand, competition has its merits. It instills a spirit of ambition, drive, and determination. By setting challenging goals and striving to attain them, individuals cultivate resilience, self-motivation, and a willingness to push beyond their limits. However, it is important to recognize that competition often requires cooperation. To win a game, an athlete must rely on their coach, trainers, and support staff. Similarly, in business, success seldom occurs in isolation. Cooperative efforts are essential for reaching ambitious goals and driving organizational growth.

Therefore, it is clear that regardless of the perspective, competition and cooperation are intertwined. They complement each other and work synergistically to unlock individual and collective potential. To maximize success, it is imperative to cultivate a spirit of cooperation and embrace its transformative power.'''

    metrics = do_bertalign_metrics(orig_text, improved_text)
    print(metrics)

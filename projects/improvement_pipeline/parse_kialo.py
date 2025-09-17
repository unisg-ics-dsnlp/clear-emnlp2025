import json
import os

import pandas as pd
import regex
import uuid
from enum import Enum

from networkx import NetworkXNoPath
from tqdm import tqdm

import networkx as nx


class Stance(Enum):
    PRO = 'Pro'
    CONTRA = 'Con'
    ROOT = 'Root'


class ArgumentType(Enum):
    SUPPORTIVE = ('Supportive', '(?P<Prompt>[PC]\d+(?:P\d+)*)(?P<Response>(?:P\d+)+)')
    CONTRADICTING = ('Contradicting', '(?P<Prompt>[PC]\d+(?:P\d+)*)(?P<Response>C\d+(?:P\d+)*)')
    COMPLEX = ('Complex', '(?P<Prompt>[PC]\d+(?:P\d+)*)(?P<Response>[PC]\d+(?:P\d+)*)')
    MULTITURN = ('Multi-Turn', '(?P<Prompt>[PC]\d+(?:P\d+)*)(?P<Prompt>(?:C\d+(?:P\d+)*)*C\d+(?:P\d+)*)')
    PURE_PRO = ('Pure Pro', '(?P<Prompt>P\d+)(?P<Response>(?:P\d+)*)')
    PURE_CON = ('Pure Con', '(?P<Prompt>C\d+)(?P<Response>(?:P\d+)*)')

    def __new__(cls, atype, rex):
        entry = object.__new__(cls)
        entry.atype = entry._value_ = atype
        entry.regex = rex
        return entry


def parse_kialo_tree(lines, target_stance=Stance.PRO, opposing_stance=Stance.CONTRA):
    g = nx.DiGraph()
    # title is always first line, always ends with newline
    title = lines[0][len('Discussion Title: '):-1]
    premise_prev = ''  # lines[2]

    # count how many lines to skip (because they were merged into previous lines)
    skip = 0
    for i, premise in enumerate(lines[2:]):
        # skip over merged lines
        if skip:
            skip -= 1
            continue
        # get the next line - must start at 3, because we skip the initial lines (title + empty line)
        try:
            next_line = lines[2 + i + 1]
        except IndexError:
            next_line = '1.'

        # count the next lines (starting from lines[i]) that dont start with a level (like 1.1.1.1.)
        next_counter = 0

        # premises start with a number, count all those that dont: they need to merged into the previous line(s)
        while not next_line.startswith('1.') or (
                premise_prev and (next_line.startswith('1. ') or next_line.startswith('2. '))):
            next_counter += 1
            try:
                next_line = lines[2 + i + next_counter + 1]
            except IndexError:
                # reached end of list
                break

        # if we had to merge lines, this is > 0
        if next_counter > 0:
            # increment skip counter to know how many lines were merged and can be skipped
            skip = next_counter
            # merge everything together
            premise = premise + ' ' + ' '.join(lines[2 + i + 1:2 + i + next_counter + 1])
        add_node_to_graph(g, premise_prev, premise)
        premise_prev = premise
    args = _parse_argument_tree(g)
    return title, args, g


def _parse_argument_tree(g):
    leaves = [v for v, d in g.out_degree() if d == 0]
    paths = [nx.shortest_path(g, '1.', leaf) for leaf in leaves]
    stances = nx.get_node_attributes(g, 'stance')
    premises = nx.get_node_attributes(g, 'premise')

    atype2args = {atype: [] for atype in ArgumentType}

    for path in paths:
        # get string of node path to match against regex
        # ignore first node because thats the root node
        stance_path = [f'P{i}' if stances[node] == Stance.PRO else f'C{i}' for i, node in enumerate(path[1:])]
        # adding the "R" for root node
        # stance_path = "R" + "".join(stance_path)
        stance_path = ''.join(stance_path)
        match_count = []
        for atype in ArgumentType:
            # argument path must start from root path
            # rex = "R" + atype.regex
            rex = atype.regex
            m = regex.findall(rex, stance_path, overlapped=True)
            if m:  # matches the pattern!
                for match in m:
                    try:
                        prompt, response = match
                    except ValueError:
                        # could not find prompt + response, only one of them
                        continue
                    if not response:
                        continue  # no response found
                    r_nodes_index_p = [int(i) for i in regex.findall('\d+', prompt)]
                    r_nodes_index_r = [int(i) for i in regex.findall('\d+', response)]

                    r_nodes_p = [path[1:][i] for i in r_nodes_index_p]
                    r_nodes_r = [path[1:][i] for i in r_nodes_index_r]

                    # do not include the root node, which is why we start at 1!
                    text_p = ' '.join(premises[node] for node in r_nodes_p)
                    text_r = ' '.join(premises[node] for node in r_nodes_r)

                    atype2args[atype].append((text_p, text_r))
                    match_count.append((atype, text_p, text_r))
    return atype2args


def add_node_to_graph(g: nx.DiGraph, parent: str, next_node: str):
    level, premise = next_node.split(' ', 1)
    try:
        stance = Stance(premise[:3])
    except ValueError:
        stance = Stance.ROOT
    if stance in [Stance.PRO, Stance.CONTRA]:
        premise = premise[4:]
    premise = premise.strip()

    if parent:
        parent_level, parent_premise = parent.split(' ', 1)
    else:
        parent_level, parent_premise = '', ''
    try:
        parent_stance = Stance(parent_premise[:3])
    except ValueError:
        parent_stance = Stance.ROOT
    if parent_stance == Stance.ROOT:
        parent_premise = parent_premise.strip()
    else:
        parent_premise = parent_premise[4:-1]
    if parent and parent not in g:
        g.add_node(parent_level, stance=parent_stance, premise=parent_premise)
    if level not in g:
        g.add_node(level, stance=stance, premise=premise)
    parent_level = level.rsplit('.', 2)[0] + '.'
    if parent:
        g.add_edge(parent_level, level)


def kialo2csv(files):
    out_file = 'kialo_args_all.csv'
    for file in tqdm(files):
        with open(file, 'r') as f:
            lines = f.readlines()
        try:
            title, args, g = parse_kialo_tree(lines)
        except NetworkXNoPath:
            # happens for at least one, TODO: check why
            continue
        for arg_type, arg_list in args.items():
            for arg in arg_list:
                # write to dataframe
                # arg = f"{title} [{str(arg_type)[len('ArgumentType.'):]}] {arg[0]} {arg[1]}"
                tmp = {
                    'title': title,
                    'type': arg_type.name,
                    'argument': f'{arg[0]} {arg[1]}',
                    'file': file
                }
                df = pd.DataFrame([tmp])
                df.to_csv(out_file, mode='a', index=False, header=False)


if __name__ == '__main__':
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _kialo_files_path = os.path.join(_current_dir, 'kialo-discussions')
    _files = os.listdir(_kialo_files_path)
    # skip the ones in greek / russian
    _files = sorted([os.path.join(_kialo_files_path, _file) for _file in _files])[14:]
    kialo2csv(_files)

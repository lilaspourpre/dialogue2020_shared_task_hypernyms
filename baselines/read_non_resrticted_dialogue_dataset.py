# -*- coding: utf-8 -*-
import json
from collections import defaultdict
from itertools import combinations

import networkx as nx
from pymorphy2 import MorphAnalyzer

from ruwordnet.ruwordnet_reader import RuWordnet

USE_SYNSETS = False
USE_TOPONYMS = False
ruwordnet = RuWordnet("../dataset/ruwordnet.db", None)
morph = MorphAnalyzer()


def read_file(filename):
    with open(filename, encoding='utf-8') as f:
        return [([i.split(";")[0]] + i.split(";")[-2:]) for i in f.read().split("\n")[:-1]]


adj_nouns = ["ДАУНХИЛ", "ИНФОРМВОЙНА", "МАСТИТ", "ШКОЛЕНЬЕ", "ЭКЗИТПОЛ", "АВАРКОМ", "АКТИВ-НЕТТО", "БАМУТСКИЙ",
             "БАСАЕВСКИЙ", "БАШХИМ", "БЛИННАЯ", "БУЛОЧНАЯ", "ГОЙСКОЕ", "ГОНЧАЯ", "ГРАНДЖ", "ДЕТСКАЯ",
             "ДМИТРИЕВ-ЛЬГОВСКИЙ", "ПОДУШЕВОЙ",
             "ДМИТРОВСК-ОРЛОВСКИЙ", "ЖЕЛЕЗНОГОРСК-ИЛИМСКИЙ", "ИРБИТСКОЕ", "КАМЕНКА-ДНЕПРОВСКАЯ", "КАМЕНЬ-КАШИРСКИЙ",
             "КОНДИТЕРСКАЯ", "ЛИКВИДКОМ", "МОГИЛЕВ-ПОДОЛЬСКИЙ", "МРАВИНСКИЙ", "МУНДА", "ОТПУСКНЫЕ", "ПАРИКМАХЕРСКАЯ",
             "ПЕЛЬМЕННАЯ", "ПИРОЖКОВАЯ", "ПЛИССЕ", "ПРАЛИНЕ", "ПРИЕМНАЯ", "РОКОКО", "РЮМОЧНАЯ", "СПАССК-РЯЗАНСКИЙ",
             "ТУ-154М", "УРАЛХИМ", "ЧАЙНАЯ", "ЧЕБУРЕЧНАЯ", "ШАХОВСКАЯ", "ШАШЛЫЧНАЯ", "ШОЛОХОВСКИЙ", "БУШ-МЛАДШИЙ",
             "СВЕРХУРОЧНЫЕ", "ТРАПЕЗНАЯ", "АППАРАТНАЯ", "ДОЛИНСКАЯ", "КАСАТЕЛЬНАЯ", "ЛОБНЯНИН", "МАГАЗИН-КОНДИТЕРСКАЯ",
             "НАДВОРНАЯ", "НАКЛАДНАЯ", "НОВОДВОРСКАЯ", "ОАОБАШХИМ", "ОПЕРАЦИОННАЯ", "ПЕРЕВЯЗОЧНАЯ", "ПОНЧИКОВАЯ",
             "ПРОИЗВОДНАЯ", "ПРОХОДНАЯ", "РУДНЯНИН", "УВОЛЬНИТЕЛЬНАЯ", "УЧИТЕЛЬСКАЯ", "ЧЕРВЛЕННАЯ", "ОТХАРКИВАЮЩЕЕ",
             "САМОТЛОРНЕФТЕПРОМХИМ", "ПРИСТЯЖНАЯ", "УМЕНЬШАЕМОЕ", "БИЗНЕС-ДЖЕТЬ", "ГАЙДАР", "ВЛАДИМИР-ВОЛЫНСКИЙ",
             "ЖЕНЩИНА-ПОЛИЦЕЙСКИЙ", "КАЩЕЙ", "ОБОРОНСТРОЙ", "БОДРОВ-СТАРШИЙ", "БУШ-СТАРШИЙ", "МИХАЛКОВ-КОНЧАЛОВСКИЙ",
             "ШВЫДКОЙ"]
adv_nouns = ["АПРЕ-СКИ", "БАТОН-РУЖ", "ГЕНПОДРЯД", "ЕВРАЗ", "ЗАПАНИБРАТА", "НИЦ", "НПРО", "ПОРТО-НОВО", "РЕПО",
             "РУСГИДРО", "ТОЛУКА-ДЕ-ЛЕРДО", "УИЧИТО", "ЭЙВОН", "ЮНИПРО", "КАЛЛИСТО", "ОКПО", "ПЕЛЕ", "ПОГРАНОКРУГ",
             "ХЕЛИ-СКИ"]


a_z0_9 = set([chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)] + [chr(i) for i in range(48, 58)]
             + [chr(36)])


def segment(word, hypernyms, use_toponyms):
    if word in adj_nouns or word in adv_nouns:
        tag = "NOUN"
    elif word in ["ЗАПОСТИТЬ", "ЗАМЕЛЯТЬ", "ЗАЗЕЛЕНЯТЬ", "ПЕРЕПОСТИТЬ", "ПОСТИТЬ", "РЕПОСТИТЬ", "ЧИФИРЯТЬ"]:
        tag = "INFN"
    elif word in ["БРАВО", "ОРОМО", "ГАВРИЛОВО", "ЗУБОВО", "РЫБНО", "ТУГУРО", "УРУС", "ЮЖНО"]:
        tag = "-"
    else:
        tag = morph.parse(word.lower())[0].tag.POS
    return filter_tag(tag, word, hypernyms, use_toponyms)


def filter_tag(_tag, _word, _hypernyms, use_toponyms):
    if use_toponyms and '2051' in [x[0] for x in _hypernyms]:
        _tag = "NOUN"

    if "---" in _word:
        _tag = "NOUN"

    elif '114955' in [x[0] for x in _hypernyms] or _tag == "ADJS":
        _tag = "NOUN"

    elif _tag and "INFN" in _tag:
        _tag = "VERB"

    elif _tag and ("ADVB" in _tag or "ADJ" in _tag or str(_tag) == "-" or "PRTF" in _tag or "INTJ" in _tag):
        pass

    elif _tag is None or (_tag and _tag != "NOUN"):
        _tag = "NOUN"

    else:
        if "ADJF" in [i.tag.POS for i in morph.parse(_word.lower())] and _word not in adj_nouns \
                and _word.endswith("Й"):
            _tag = "-"
        else:
            _tag = "NOUN"
    return _tag


direct_hyp = read_file('../dataset/full_data/clean_hyp.lst')
second_order_hyp = read_file('../dataset/full_data/clean_mintree2.lst')
existing_synsets = read_file('../dataset/full_data/clean_syn.lst')

full_dataset = defaultdict(set)

for word, synset, synset_name in direct_hyp:
    full_dataset[word].add((synset, synset_name))

for word, synset, synset_name in second_order_hyp:
    full_dataset[word].add((synset, synset_name))

if USE_SYNSETS:
    for word, synset, synset_name in existing_synsets:
        full_dataset[word].add((synset, synset_name))
else:
    for word, synset, synset_name in existing_synsets:
        if word in full_dataset or word in {'ЛУЧИК', 'ПОБИВАНИЕ', 'СОВОЧЕК', 'ИКОНКА', 'БУРКА', 'ПЕРЕТРУЖДАТЬСЯ', 'РАСПЕРЕЖИВАТЬСЯ', 'ОТРЕЖИССИРОВАТЬ', 'ОХРЕНЕТЬ', 'ТУСИТЬ', 'ИЗРИСОВАТЬ', 'ОТРЕКОНСТРУИРОВАТЬ', 'ВТЮХИВАТЬ', 'ПРИШВАРТОВЫВАТЬСЯ', 'ПИКАТЬ', 'ПРОМУЛЬГИРОВАТЬ', 'НАДУРИТЬ', 'РУСОФОБСТВОВАТЬ', 'ПРОДЛЯТЬ', 'ИСПОЛОСОВАТЬ', 'КООПЕРИРОВАТЬСЯ', 'ПЕРЕЗВАНИВАТЬСЯ', 'КРИВИТЬ', 'ДЕКРИМИНАЛИЗОВАТЬ', 'СНЕЖИТЬ', 'ВЫМОРАЖИВАТЬ', 'ПЕРСОНАЛИЗИРОВАТЬ', 'ЮМОРИТЬ', 'КРОВИТЬ', 'ЛЕГИТИМИЗОВАТЬ', 'ПЕРЕУСТРАИВАТЬ', 'УМЫКНУТЬ', 'ЗАГЛОТИТЬ', 'РАЗМИНАТЬ', 'МУРЧАТЬ', 'ОТРАЗИТЬСЯ', 'ПРИКОПАТЬСЯ', 'ТЯПНУТЬ', 'ВБУХИВАТЬ', 'ПРОРИСОВЫВАТЬСЯ', 'СЫРОНИЗИРОВАТЬ', 'УМЯТЬ', 'ТЯВКАТЬ', 'ОДУРМАНИВАТЬ', 'НАВАРИТЬСЯ', 'СОЛИДАРИЗОВАТЬСЯ', 'АКТИВИРОВАТЬ', 'НАРЫВАТЬСЯ', 'ЗАНЫРНУТЬ'}:
            full_dataset[word].add((synset, synset_name))


with open("ruwordnet_non-restricted_nouns_final.tsv", 'w', encoding='utf-8', newline='\n') as w1, \
        open("ruwordnet_non-restricted_verbs_final.tsv", 'w', encoding='utf-8', newline='\n') as w2, \
        open("ruwordnet_non-restricted-nouns_no_labels_final.tsv", 'w', encoding='utf-8', newline='\n') as n1, \
        open("ruwordnet_non-restricted-verbs_no_labels_final.tsv", 'w', encoding='utf-8', newline='\n') as n2:
    counter = 0
    for word, hypernyms in full_dataset.items():

        if len(set(word).intersection(a_z0_9)) != 0:
            continue

        tag = segment(word, hypernyms, USE_TOPONYMS)

        if tag == "NOUN":
            exist = [ruwordnet.get_name_by_id(i + "-N") != "" for i, _ in hypernyms]
        elif tag == "VERB":
            exist = [ruwordnet.get_name_by_id(i + "-V") != "" for i, _ in hypernyms]
        else:
            continue

        if "---" in word:
            word = word.strip("-")

        hypernyms = set([i[1] for i in filter(lambda x: exist[x[0]], enumerate(hypernyms))])

        if hypernyms != set():
            G = nx.Graph()
            for synset, name in hypernyms:
                G.add_node(synset+"-"+tag[0])

            for f_h, s_h in combinations(hypernyms, 2):
                if bool(len(ruwordnet.is_hyponym(f_h[0]+"-"+tag[0], s_h[0]+"-"+tag[0]))) \
                        or bool(len(ruwordnet.is_hyponym(s_h[0]+"-"+tag[0], f_h[0]+"-"+tag[0]))):
                    G.add_edge(f_h[0]+"-"+tag[0], s_h[0]+"-"+tag[0])

            if tag == "NOUN":
                n1.write(f"{word}\n")
            elif tag == "VERB":
                n2.write(f"{word}\n")
            for component in nx.connected_components(G):
                if tag == "NOUN":
                    w1.write(f"{word}\t"
                             f"{json.dumps(list(component))}\t"
                             f"{[ruwordnet.get_name_by_id(i) for i in component]}\n")
                elif tag == "VERB":
                    w2.write(f"{word}\t"
                             f"{json.dumps(list(component))}\t"
                             f"{[ruwordnet.get_name_by_id(i) for i in component]}\n")

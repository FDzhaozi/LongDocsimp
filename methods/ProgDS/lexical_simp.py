import re

from methods.ProgDS.discourse_simp import main_disc
from methods.ProgDS.para_simp import extract_para_from_topics, main_para
from methods.SumDS.gen_paras import gen_shot
from utils.chat import askChatGPT
from utils.config import para_data_path
from collections import defaultdict

import nltk
import csv
from utils.config import lex_data_path


def gen_lex_shot(lex_data_path, sent, shot_set=5):
    words = nltk.pos_tag(nltk.word_tokenize(sent))
    words = [word for word, pos in words if pos in ['NN', 'VB']][:shot_set]

    examples = []
    with open(lex_data_path) as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            for word in words:
                if word in row['complex_lex'] or word in row['simple_lex']:
                    examples.append(row['complex_lex'] + "→" + row['simple_lex'])

    return "、".join(examples)


def lex_simp(sent_list, shot_set):
    """# 对每个段落进行简化,从句子之间的关系以及句子内部的结构两方面考虑
    para_list: 原文的段落列表
    shot_set: zero-shot/one-shot/two-shot 0/1/2
    if_shot_fixed: 是否固定shot True/False
    if_cot: 是否使用COT True/False """

    simple_sents = """"""
    for sent in sent_list:
        shots = gen_lex_shot(lex_data_path, sent, shot_set)
        prompt_sys = f"""You are a query engine equipped with a wide range of simpler alternatives for complex expressions. The task is to identify complex and uncommon vocabulary, phrases, idioms, etc. in a given sentence. And then provide simplified alternatives for these complex elements.\\
      1. Incorporate the replacements into the sentence and ensure that the sentences remain smooth and coherent.\\
      2. Explain an unfamiliar idea using more familiar words and examples that people know.\\
      3. The sentence structure doesn't need to be considered, and the overall meaning should be maintained as much as possible after the replacements are made.\\
    (examples)\\"""

        prompt_user = f"""{shots}\n
        Sentence with complex expression: [{sent}]
        Simple version:"""

        while True:
            messages = [{"role": "system", "content": prompt_sys}]
            d = {"role": "user", "content": prompt_user}
            messages.append(d)
            result = askChatGPT(messages)

            pattern = r"Simple Paragraph: (.*)"

            match = re.search(pattern, result)
            if match:
                simple_sents += match.group(1)
                # print(simple_para)
                break
            else:
                simple_sents += result
                break

        return simple_sents


def main_lex(mode, simple_doc_with_simppara, shot_set):
    subheadings_paras = extract_para_from_topics(mode, simple_doc_with_simppara)
    simple_doc_with_simplex = """"""
    for subheading, paras in subheadings_paras.items():
        simple_doc_with_simplex += subheading + "\n\n"
        for para in paras:
            sents = nltk.sent_tokenize(para)
            simple_sents = lex_simp(sents, shot_set)
            simple_doc_with_simplex += simple_sents + "\n\n"

    return simple_doc_with_simplex


if __name__ == '__main__':
    raw_doc = ""
    # 参数设置
    shot_set = 2
    if_shot_fixed = False
    if_cot = True
    mode, simple_doc_with_subtitle = main_disc(raw_doc)
    simple_doc_with_simppara = main_para(mode, simple_doc_with_subtitle, shot_set, if_shot_fixed, if_cot)
    simple_doc_with_simplex = main_lex(mode, simple_doc_with_simppara, shot_set=5)

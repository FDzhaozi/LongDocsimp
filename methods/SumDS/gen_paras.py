import csv
import random
import re

import nltk
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

from utils.chat import askChatGPT
import re


import pandas as pd
from sentence_transformers import SentenceTransformer
from utils.config import sent_data_path, para_data_path

# 初始化文本嵌入模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def find_sim(raw_para, data_path, nums):
    # 将raw_para编码为向量
    raw_vec = model.encode([raw_para])

    # 读取数据路径中向量库
    df = pd.read_csv(data_path)
    complex_vecs = df['complex_vec'].values

    # 计算余弦相似度,找到topk相似向量
    sims = linear_kernel(raw_vec, complex_vecs).ravel()
    top_indices = sims.argsort()[-nums:][::-1]

    # 构建返回值列表
    shots = []
    for idx in top_indices:
        complex_para = df['complex_para'].iloc[idx]
        simple_para = df['simple_para'].iloc[idx]
        shots.append((complex_para, simple_para))

    return shots


def seg_doc(raw, symbol="\n\n"):
    paras = re.split(symbol, raw)
    paras = [para.strip() for para in paras]
    return paras


def gen_cot(complex_para, simple_para):
    prompt_sys = """In order to simplify a complex and difficult paragraph into a simple and easily readable one, 
    several thoughtful steps are required. Now, please refer to the original paragraph, thought process, simplified 
    paragraph I provided, and generate a corresponding simplified thought process for the given original paragraph - 
    simplified paragraph."""

    prompt_user = f"""Complex paragraph:[Lord Toby Jug (born Brian Borthwick, 18 December 1965 – 2 May 2019) was a 
    British politician. <s> He was the leader of the Cambridgeshire and Huntingdonshire branch of the Official 
    Monster Raving Loony Party, serving as the party's media officer and a prospective parliamentary candidate, 
    until being expelled from the Loony Party in 2014. <s> He founded The Eccentric Party of Great Britain in 2015. 
    <s> Jug was expelled from the Monster Raving Loony Party in 2014 over comments made about UKIP leader Nigel 
    Farage, and for his criticism of pub chain J D Wetherspoon, a company which the Loony party had been attempting 
    to attract as a sponsor. <s> Party leader Howling Laud Hope said that it was "about the fourth time we have asked 
    him to leave the party" and that the Loony Party was "not in the game of upsetting people". <s> When leaving the 
    party, Jug expressed concern over racist comments expressed by Hope in a "Guardian" article from 14 years 
    previously. <s> Jug died on 2 May 2019, at the age of 53.]

    Simple paragraph:[Lord Toby Jug (born Brian Borthwick; 1965 – 2 May 2019) was a British politician. <s> He was 
    the leader of the Cambridgeshire and Huntingdonshire branch of the Official Monster Raving Loony Party. <s> He 
    also served as the party's media officer and a likely parliamentary candidate. <s> After this, he founded the 
    Eccentric Party of Great Britain in 2015. <s> He was removed from the Loony Party in 2014. <s> Jug died on 2 May 
    2019, at the age of 53.]
    
    Reasoning: In the complex paragraph, the information is presented in a more detailed and convoluted manner, 
    while the simplified paragraph presents the same information in a more concise and straightforward way. the 
    simplification process involves restructuring sentences, eliminating non-essential information, 
    removing repetition, and highlighting key events. By doing so, the simplified paragraph presents the main points 
    in a more concise and easily understandable manner.
    
    Complex paragraph:{complex_para}
    
    Simple paragraph:{simple_para}
    
    Reasoning:
    """

    messages = [{"role": "system", "content": prompt_sys}]
    d = {"role": "user", "content": prompt_user}
    messages.append(d)
    result = askChatGPT(messages)

    return result


def gen_sent_cot(complex_sent, simple_sent):
    prompt_sys = """In order to simplify a complex sentence into a simple and easily readable one, 
    several thoughtful steps are required. Now, please refer to the original sentence, thought process, simplified 
    sentence I provided, and generate a corresponding simplified thought process for the given original sentence - 
    simplified sentence."""

    prompt_user = f"""Complex sentence: The employee who works in the accounting department had submitted his 
    resignation letter. Simple sentence: He quit his job. 
    Reasoning: Unnecessary words like "accountant" and "resigned" are replaced with more simple and common words "quit" 
    and "job". Non-essential details are removed to highlight the key point in a more concise way.
    
    Complex sentence: [{complex_sent}]

    Simple sentence: [{simple_sent}]

    Reasoning:
    """

    messages = [{"role": "system", "content": prompt_sys}]
    d = {"role": "user", "content": prompt_user}
    messages.append(d)
    result = askChatGPT(messages)

    return result


def gen_shot(mode, data_path, raw_text, shot_set, if_shot_fixed, if_with_cot):
    # mode = “para” or mode = "sent"

    if if_shot_fixed:
        df = pd.read_csv(data_path)
        shots = df.loc[0:shot_set - 1, ['complex_' + mode, 'simple_' + mode]]

    else:
        shots = find_sim(raw_text, data_path, shot_set)

    if if_with_cot:
        for i in range(len(shots)):
            complex_data = shots.loc[i, 'complex_' + mode]
            simple_data = shots.loc[i, 'simple_' + mode]
            cot = gen_cot(complex_data, simple_data) if mode == "para" else gen_sent_cot(complex_data, simple_data)
            shots.loc[i, 'cot'] = cot

    shots_str = ""
    for i in range(len(shots)):
        complex_data = shots.loc[i, 'complex_' + mode]
        simple_data = shots.loc[i, 'simple_' + mode]
        shots_str += f"Complex {mode}: {complex_data}\n"
        if if_with_cot:
            cot = shots.loc[i, 'cot']
            shots_str += f"\nReasoning: {cot}\n"
        shots_str += f"Simple Paragraph: {simple_data}"

    return shots_str





def simp_sent(sum_text, sent_list, shot_set, if_shot_fixed, if_cot):
    simple_doc = """"""
    for sent in sent_list:

        shots = gen_shot("sent", sent_data_path, sent, shot_set,
                         if_shot_fixed, if_cot)

        prompt_sys = f"""You are a text editor tasked with simplifying a sentence. 
        Your goal is to simplify sentences under the guidance of the summary.
        Here are some operations that may be used:
        1. Simplify complex structures like long sentences.
        2. Replace difficult words with simpler synonyms.  
        3. Remove unnecessary details.
        4. Maintain the key information.
        5. Delete sentences that are less relevant to the main idea.
        Summary: [{sum_text}]"""

        prompt_user = f"""{shots}
         Complex Sentence: [{sent}]
         Reasoning: 
         Simple sentence:"""

        while True:
            messages = [{"role": "system", "content": prompt_sys}]
            d = {"role": "user", "content": prompt_user}
            messages.append(d)
            result = askChatGPT(messages)

            pattern = r"Simple Sentence: (.*)"

            match = re.search(pattern, result)
            if match:
                simple_sent = match.group(1)
                break
            else:
                print("No match, try again")

        simple_doc += simple_sent + ".\n"

    return simple_doc


def simp_para(sum_text, para_list, shot_set, if_shot_fixed, if_cot):
    """# 在摘要的指导下对原文的每个段落进行简化
    sum_text: 已经生成的摘要
    para_list: 原文的段落列表
    shot_set: zero-shot/one-shot/two-shot 0/1/2
    if_shot_fixed: 是否固定shot True/False
    if_cot: 是否使用COT True/False """

    simple_doc = """"""
    for para in para_list:
        shots = gen_shot("para", para_data_path, para, shot_set, if_shot_fixed, if_cot)
        prompt_sys = f"""You are a text editor tasked with simplifying a document. You
        goal is to simplify paragraphs under the guidance of the summary.
        Here are some operations that may be used:
        1. Delete irrelevant sentences based on the summary document
        and context.
        2. Merge complex and redundant sentences to improve readability.
        3. Split complex sentences into simpler ones.
        4. Rephrase sentences with complex words or phrases.
        5. Retain important and already simplified sentences.
        6. Replace difficult expressions with simpler ones.
        Summary:[{sum_text}]
        """

        prompt_user = f"""{shots}\n
        Complex Paragraph: [{para}]
        Reasoning:
        Simple paragraph:"""

        while True:
            messages = [{"role": "system", "content": prompt_sys}]
            d = {"role": "user", "content": prompt_user}
            messages.append(d)
            result = askChatGPT(messages)

            pattern = r"Simple Paragraph: (.*)"

            match = re.search(pattern, result)
            if match:
                simple_para = match.group(1)
                # print(simple_para)
                break
            else:
                print("No match, try again")
        simple_doc += simple_para + "\n\n"

        return simple_doc


def polish_text(text):
    """对生成的文本进行润色"""
    # sys_prompt = """You are a professional article polisher,An easy to read document should have good contextual
    # coherence, good organizational structure, and smooth connecting words. please polish the document I provided to
    # make it more readable. it should be noted that,for each sentence, you hardly need to make any changes. What you
    # need to do is to make the document more coherent and easy to read at the document level.""" user_prompt =
    # f"""Raw document:[{text}] After polish:"""
    pass

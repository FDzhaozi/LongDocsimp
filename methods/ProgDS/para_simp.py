import re

from methods.ProgDS.discourse_simp import main_disc
from methods.SumDS.gen_paras import gen_shot
from utils.chat import askChatGPT
from utils.config import para_data_path
from collections import defaultdict


def extract_para_from_topics(mode, simple_doc_with_subtitle):
    paras = simple_doc_with_subtitle.split("\n\n")

    subheading_paras = defaultdict(list)

    current_subheading = ""
    current_para = ""

    for para in paras:
        if not para:
            continue

        if para.startswith("##"):
            current_subheading = para
            current_para = ""
        else:
            current_para = para
            subheading_paras[current_subheading].append(current_para)

    if mode == "sent":
        for key, value in subheading_paras.items():
            sentences = "".join(value)
            subheading_paras[key] = [sentences]
    # {heading:paras}
    return subheading_paras


def para_simp(para_list, shot_set, if_shot_fixed, if_cot):
    """# 对每个段落进行简化,从句子之间的关系以及句子内部的结构两方面考虑
    para_list: 原文的段落列表
    shot_set: zero-shot/one-shot/two-shot 0/1/2
    if_shot_fixed: 是否固定shot True/False
    if_cot: 是否使用COT True/False """

    simple_doc = """"""
    for para in para_list:
        para_shots = gen_shot("para", para_data_path, para, shot_set, if_shot_fixed, if_cot)
        sent_shots = gen_shot("sent", para_data_path, para, shot_set, if_shot_fixed, if_cot)
        prompt_sys = f"""You are a professional manuscript editor and reviser. The task is to simplify a given 
        paragraph to improve accessibility and readability. When it comes to the meaning and structure of the entire 
        paragraph, you need to simplify it according to the following requirements: 
        1. Identify key points or concepts, simplify the structure, and preserve them. 
        2. Provide additional context or explanations for unfamiliar concepts to enhance understanding. 
        3. Ensure logical connections between sentences with appropriate transitions and consider dividing paragraphs if 
        they contain multiple ideas or information. 
        {para_shots}
        When it comes to the structure between sentences and within individual sentences, you need to simplify it 
        according to the following requirements: 
        1. Consecutive simple sentences can be merged into a single sentence. 
        2. Complex and lengthy sentences can be split into multiple simpler sentences. 
        3. Irrelevant sentences can be deleted. 
        4. Sentence order can be rearranged to enhance the overall flow of the text. 
        5. The simplest sentence structure in English is subject-verb-object-period, subject-verb-object-period and so on. 
        Try to use the simplest structure. 
        {sent_shots}"""

        prompt_user = f"""Complex Paragraph: [{para}]
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


def main_para(mode, simple_doc_with_subtitle, shot_set, if_shot_fixed, if_cot):
    subheading_paras = extract_para_from_topics(mode, simple_doc_with_subtitle)
    simple_doc_with_simppara = """"""
    for subheading, paras in subheading_paras.items():
        simple_paras = para_simp(paras, shot_set, if_shot_fixed, if_cot)
        simple_doc_with_simppara += subheading + "\n\n" + simple_paras

    return simple_doc_with_simppara


if __name__ == '__main__':
    raw_doc = ""
    # 参数设置
    shot_set = 2
    if_shot_fixed = False
    if_cot = True
    mode, simple_doc_with_subtitle = main_disc(raw_doc)
    simple_doc_with_simppara = main_para(mode,simple_doc_with_subtitle,shot_set, if_shot_fixed, if_cot)


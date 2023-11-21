import re

import nltk

from utils.chat import askChatGPT
from utils.config import model_token_limit
from utils.stat import num_tokens_from_messages, stat_para_nums


def if_mesg_out(message):
    if num_tokens_from_messages(message) > (model_token_limit - 300):
        return True
    else:
        return False


def count_paragraphs(raw_doc):
    paras = re.split(r"\n\n", raw_doc)
    paras = [para.strip() for para in paras if para]  # 去除空段落
    return len(paras)


def split_to_sentences(paragraph):
    sentences = re.split(r'[。!?]', paragraph)
    sentences = [sentence.strip() for sentence in sentences if sentence]

    numbered_paragraph = ""
    num_of_sentences = len(sentences)

    for i, sentence in enumerate(sentences):
        number = str(i + 1) + "."
        numbered_sentence = number + " " + sentence
        numbered_paragraph += numbered_sentence + "\n"

    return numbered_paragraph, num_of_sentences


def segment_with_num(raw_doc, symbol="\n\n"):
    paras = re.split(symbol, raw_doc)
    paras = [para.strip() for para in paras]

    numbered_doc = ""
    num_of_paras = len(paras)

    for i, para in enumerate(paras):
        number = str(i + 1) + "."
        numbered_para = number + " " + para
        numbered_doc += numbered_para + "\n\n"

    return numbered_doc, num_of_paras


def check_simp_doc(plan_script):
    # 检查字符串中##的数量
    if plan_script.count("##") < 3:
        print("##数量不足3个")
        return False

    # 按##分割字符串
    sections = plan_script.split("##")
    # 去除其中的空字符串
    sections = [section.strip() for section in sections if section.strip() != ""]
    print("sections: ", sections)

    all_nums = []
    # 检查每个副标题是否符合要求
    for section in sections:
        # 检查副标题是否存在冒号
        if ":" not in section:
            print("副标题不存在冒号")
            return False

        # 检查副标题冒号后面是否有数字
        nums = section.split(":")[-1].strip()
        nums = nums.split(",")
        # 去除空格
        nums = [num.strip() for num in nums if num.strip() != ""]
        print("nums: ", nums)
        all_nums.extend(nums)
        for num in nums:
            if not num.isdigit():
                print("副标题冒号后面不存在数字")
                return False
    # 检查所有副标题冒号后面的所有数字是否连续（从小到大），且不能有重复
    all_nums = [int(num) for num in all_nums]
    # 检查是否有重复
    if len(all_nums) != len(set(all_nums)):
        print("副标题冒号后面的数字有重复")
        return False
    # 检查是否连续（从小到大）
    for i in range(len(all_nums) - 1):
        if all_nums[i + 1] - all_nums[i] != 1:
            print("副标题冒号后面的数字不连续")
            return False

    return True


def add_subtitles(doc_with_nums, plan_script):
    # 初始化文档
    final_doc = ""

    # 分割规划格式,默认每个##为一个部分
    plan_parts = plan_script.split('##')

    for part in plan_parts:

        # 忽略空白部分
        if not part:
            continue

        # 提取副标题和序号列表
        subtitle = part.split(":")[0].strip()
        nums = part.split(":")[1].strip().split(",")

        # 加入副标题
        final_doc += f"## {subtitle}\n\n"

        # 循环加入对应段落
        for num in nums:
            para = re.sub(f"\d+\.", "", doc_with_nums.split(f"{num}.")[0])
            final_doc += para + "\n\n"

    return final_doc


def cluster_sents_and_subheadings(raw_doc):
    # 对于全文只有一段的文档，例如wiki，在discourse-level simp阶段需要先分句，然后对于每个副标题对应的topic都包含多个句子。
    numbered_sents, sent_nums = split_to_sentences(raw_doc)
    prompt_sys = """You are a professional manuscript editor and reviewer. The task
        is to organize and divide an paragraph into multiple distinct topics.
        Each sentence in the article is numbered.
        1. The goal is to maintain a consistent central theme for each topic.
        2. Subheadings need to be generated for each topic.
        3. Irrelevant sentences can be deleted.
        The output format must be a subheading followed by sentence
        numbers, where these sentence numbers represent the same
        topic. 
        The format is as follows: 
        [## subtitle: Fill in a few numbers here to represent all the paragraphs included under this heading]
        for example: 
          [##The situation of women's rights worldwide: 1,2,3,4,5\n
          ##Unfair treatment of women here: 6,7,8\n 
          ##Natasha's Struggle Against Unfair Treatment of Women: 9,10,11,12\n]
          Attention:You don't need to provide the text content of the article, just provide the 
          corresponding serial number and several serial numbers under the same title must be consecutive."
        """

    prompt_user = f"""Source paragraph with sentence numbers:{numbered_sents}\n
        The organized content:"""

    message = [
        {
            "role": "system",
            "content": prompt_sys,
        },
        {
            "role": "user",
            "content": prompt_user,
        }
    ]

    while True:
        plan_script = askChatGPT(message)
        if check_simp_doc(plan_script):
            simple_doc_with_subtitle = add_subtitles(numbered_sents, plan_script)
            break
        else:
            print("No match, try again")

    return simple_doc_with_subtitle


def cluster_paras_and_subheadings(raw_doc):
    # 由于模型生成的内容总是对原文有损的，所以后面采用的方法是生成副标题和对应的段落序号，这样基本也不会有超出模型窗口的问题
    doc_with_nums, para_nums = segment_with_num(raw_doc)
    prompt_sys = """You are a professional manuscript editor and reviewer. The task
    is to organize and divide an article into multiple distinct topics.
    Each paragraph in the article is numbered.
    1. The goal is to maintain a consistent central theme for each topic.
    2. Subheadings need to be generated for each topic.
    3. Irrelevant paragraphs can be deleted.
    The output format must be a subheading followed by paragraph
    numbers, where these paragraph numbers represent the same
    topic. 
    The format is as follows: 
    [## subtitle: Fill in a few numbers here to represent all the paragraphs included under this heading]
    for example: 
      [##The situation of women's rights worldwide: 1,2,3,4,5\n
      ##Unfair treatment of women here: 6,7,8\n 
      ##Natasha's Struggle Against Unfair Treatment of Women: 9,10,11,12\n]
      Attention:You don't need to provide the text content of the article, just provide the 
      corresponding serial number and several serial numbers under the same title must be consecutive."
    """

    prompt_user = f"""Source document with paragraph number:{doc_with_nums}\n
    The organized content:"""

    message = [
        {
            "role": "system",
            "content": prompt_sys,
        },
        {
            "role": "user",
            "content": prompt_user,
        }
    ]

    while True:
        plan_script = askChatGPT(message)
        if check_simp_doc(plan_script):
            simple_doc_with_subtitle = add_subtitles(doc_with_nums, plan_script)
            break
        else:
            print("No match, try again")

    return simple_doc_with_subtitle


def extract_version0(file_doc_list):
    """doc_list:包含多个小列表，每个小列表包含多个字典，每个字典包含title, filename, content, version等信息，小列表中的所有字典都对应同一文章的不同版本
    :return content_doc_list:一个列表，每个元素是一个字符串以及对应的段落数量，是version0的文章"""
    content_doc_list = []
    for i in file_doc_list:
        for j in i:
            if j["version"] == 0:
                # print("title: ", j["title"])
                # print("filename: ", j["filename"])
                doc = j["content"]
                doc, para_nums = stat_para_nums(doc)
                # 列表拼接为字符串
                doc = "\n\n".join(doc)
                temp_list = [doc, para_nums]
                content_doc_list.append(temp_list)
    return content_doc_list


def main_disc(raw_doc):
    # 这一阶段就是给定一个文档，生成一个简化文档（结构清晰的）
    if count_paragraphs(raw_doc) == 1:
        mode = "sent"
        simple_doc_with_subtitle = cluster_sents_and_subheadings(raw_doc)

    else:
        mode = "para"
        simple_doc_with_subtitle = cluster_paras_and_subheadings(raw_doc)
    return mode, simple_doc_with_subtitle


if __name__ == '__main__':
    raw_doc = ""
    mode, simple_doc_with_subtitle = main_disc(raw_doc)

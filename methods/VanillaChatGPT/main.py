import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

from models.gpt_35_chat import get_gpt_35_llm, get_gpt_35_16k_llm
from utils import extract_chunk_paras, extract_chapters, std_output, append_to_json_file, rebuild_json_file, run
from tqdm import tqdm
from models.gpt_4_chat import get_gpt_4_llm
from models.gemini_chat import get_gemini_llm, get_gemini_out_without_safe
from utils import read_str_from_txt, calc_tokens
from data import load_wiki_doc, load_newsela_doc
import json
from google.generativeai.types import HarmCategory, HarmBlockThreshold





if __name__ == "__main__":
    # wiki数据集

    # wiki_set = load_wiki_doc(path=r"D:\Dataset\wiki_auto_plan\wiki_auto\wikiauto_docs_test.csv", nums=200)
    # wiki_complex_set = wiki_set[0]
    # wiki_simple_set = wiki_set[1]

    # newsela数据集
    doc_list, content_list = load_newsela_doc(
        path=r"D:\Dataset\newsela_share_2020\newsela_share_2020\documents\articles\\", nums=100000)

    # gemini
    # gemini_llm = get_gemini_llm()
    #
    #
    # gpt-35
    # gpt_35_llm = get_gpt_35_llm()
    # gpt_35_16k_llm = get_gpt_35_16k_llm()

    # 创建一个空的JSON文件
    # file_path = 'wiki_data_gemini_base_prompt.json'
    file_path = 'newselaA_data_gemini_force_prompt.json'
    with open(file_path, 'w', encoding="utf-8") as file:
        pass

    # base_prompt

    base_prompt = """You are a professional simplified text writer,I need you to simplify the language and structure
    of the raw text to make it more accessible to pupils.  Replace complex words or phrases or technical terms with
    simpler, more familiar words or terms, use more and shorter clauses, and reorganize clauses to make them easier
    to read. Raw:[{raw_text}] Simplified:"""

    # force prompt
    force_prompt = """As a text simplification writer, your task is to simplify the given text content: restate the
    original text in simpler and easier to understand language without changing its meaning as much as possible. You
    can change paragraph or sentence structure, remove some redundant information, and replace complex and uncommon
    expressions with simple and common ones. It should be noted that the task of text simplification is completely
    different from the task of text summarization, so you need to provide a simplified parallel version based on the
    original text, rather than just providing a brief summary. \nRaw:[{raw_text}] \nSimplified:"""

    # one_shot_prompt
    one_shot = read_str_from_txt("prompt/doc_prompt.txt")
    one_shot_prompt = one_shot + """\n Raw:[{raw_text}]
    Simplified:"""

    # for complex_text, simple_text in tqdm(zip(wiki_complex_set[:], wiki_simple_set[:])):
    #     # 生成简化文本
    #     complex_text = del_angle_brackets(complex_text)
    #     simple_text = del_angle_brackets(simple_text)
    #     # base_prompt_gemini = f"""You are a professional simplified text writer,I need you to simplify the language and
    #     # structure of the raw document to make it more accessible to pupils.  Replace complex words or phrases or
    #     # technical terms with simpler, more familiar words or terms, use more and shorter clauses, and reorganize clauses
    #     # to make them easier to read. Please note that this does not require you to write a summary of the document. You
    #     # need to simplify each sentence to generate the corresponding simplified document. Redundant or unimportant
    #     # sentences can be deleted.
    #     # Raw:[{complex_text}]
    #     # Simplified:"""
    #     print("complex_text:", complex_text)
    #     # simplified_text = get_gemini_out_without_safe(base_prompt_gemini)
    #     simplified_text = run(gpt_35_llm, one_shot_prompt, complex_text)
    #     print("simplified_text:", simplified_text)
    #     data = {'complex_text': complex_text,
    #             'simple_text': simple_text,
    #             'gemini_base_prompt': simplified_text}
    #     append_to_json_file(data, file_path)





    less_1000 = 0
    more_1000 = 0
    id = 0
    for names, contents in zip(doc_list[0:8000], content_list[0:8000]):
        raw_text = ""
        ver1_text = ""
        ver2_text = ""
        ver3_text = ""
        ver4_text = ""
        doc_name = ""
        print("id:", id)
        id += 1
        if less_1000 == 500:
            break
        for name, content in zip(names, contents):
            if name.endswith("0.txt"):
                doc_name = name.split(".")[0]
                raw_text = content

            elif name.endswith("1.txt"):
                ver1_text = content
            elif name.endswith("2.txt"):
                ver2_text = content
            elif name.endswith("3.txt"):
                ver3_text = content
            elif name.endswith("4.txt"):
                ver4_text = content
            else:
                print("error")

        if calc_tokens(raw_text) < 1200:
            less_1000 += 1

        else:
            more_1000 += 1
            continue

        print("complex_text:", raw_text)
        print("len of complex_text:", calc_tokens(raw_text))
        #simplified_text = run(gemini_llm, one_shot_prompt, raw_text)
        force_prompt_gemini = force_prompt.format(raw_text=raw_text)
        simplified_text = get_gemini_out_without_safe(force_prompt_gemini)
        print("simplified_text:", simplified_text)
        data = {'doc_name': doc_name,
                'complex_text': raw_text,
                'simple_text_v1': ver1_text,
                'simple_text_v2': ver2_text,
                'simple_text_v3': ver3_text,
                'simple_text_v4': ver4_text,
                'model_output': simplified_text}
        append_to_json_file(data, file_path)

    rebuild_json_file("newselaA_data_gemini_force_prompt.json", "newsela-result/newselaA_gemini_force_prompt.json")



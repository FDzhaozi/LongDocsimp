import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

def read_str_from_txt(file_path):
    # Read the file, utf-8 is used to read the file, the return type is string
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()



def read_json_file(file_path):
    """
    Reads a JSON file and returns a list of all JSON data.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        list: A list of all JSON data.
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def extract_chapters(path):

    chapter_names = []
    original_chapters = []
    simplified_chapters = []


    # 遍历路径下的所有文件夹
    for root, _, files in os.walk(path):
        # 遍历每个文件夹下的txt文件
        original_file = ""
        simplified_file = ""
        # 如果files中的文件名都不包含数字，那么就跳过
        if not any([True for file in files if re.match(r"\d+", file)]):
            continue
        for file in files:
            if file.endswith(".txt"):
                # 获取原始版本和简化版本的文件名
                # 编写正则表达式，如果.txt前面只有数字，那么就是原始版本的文件名
                reg = re.compile(r"^\d+\.txt$")
                if reg.match(file):
                    original_file = os.path.join(root, file)
                else:
                    simplified_file = os.path.join(root, file)

        print("original_file: ", original_file)
        print("simplified_file: ", simplified_file)
        # 计算原始版本和简化版本的段落数
        chapter_names.append(original_file)
        original_content = read_str_from_txt(original_file)
        original_chapters.append(original_content)
        simplified_content = read_str_from_txt(simplified_file)
        simplified_chapters.append(simplified_content)

    return chapter_names, original_chapters, simplified_chapters


def extract_chunk_paras(chunk_size, original_chapter):
    # 将原始章节文本按段落划分
    paragraphs = original_chapter.split("\n\n")  # 假设段落之间由两个换行符分隔

    chunks = []
    current_chunk = []
    current_size = 0

    for paragraph in paragraphs:
        if not paragraph.strip():  # 跳过空段落
            continue
        words = paragraph.split()  # 使用空格分割段落为单词

        if current_size + len(words) <= chunk_size:
            current_chunk.append(paragraph)  # 将段落添加到当前chunk
            current_size += len(words)  # 更新当前chunk的词汇数量
        else:
            chunks.append(current_chunk)  # 当前chunk已满，添加到chunks列表中
            current_chunk = [paragraph]  # 创建新的chunk
            current_size = len(words)  # 更新当前chunk的词汇数量

    # 添加最后一个chunk（如果有剩余的段落）
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def std_output(output_text):
    # 如果字符串中同时包含"[" 和 "]"，那么就提取出其中的字符串
    if "[" in output_text and "]" in output_text:
        # 拿到[和]的索引
        start = output_text.index("[")
        end = output_text.index("]")
        # 提取出[]中的字符串
        output_text = output_text[start + 1:end]
    return output_text

def calc_tokens(text, model="gpt-3.5-turbo"):
    """使用openai的工具，计算文本的token数量"""
    encoding = tiktoken.encoding_for_model(model)
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(text))
    return num_tokens


def append_to_json_file(data, file_path):
    with open(file_path, 'a', encoding="utf-8") as file:
        json.dump(data, file)
        file.write('\n')


def del_angle_brackets(text):
    # 删除尖括号以及其中的内容
    return re.sub(r"<[^>]*>", "", text)


def run(llm, prompt, raw_text):
    # 生成简化文本
    prompt = PromptTemplate.from_template(prompt)
    direct_simp_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    output = direct_simp_chain.invoke(raw_text)["text"]
    output = std_output(output)
    return output


def rebuild_json_file(input_file, output_file):
    with open(input_file, 'r', encoding="utf-8") as input_f:
        lines = input_f.readlines()

    json_data = []
    for line in lines:
        line = line.strip()
        if line:
            json_data.append(json.loads(line))

    with open(output_file, 'w', encoding="utf-8") as output_f:
        output_f.write(json.dumps(json_data, indent=4))
if __name__ == '__main__':
    chapter = """"""

    chunk_size = 1200
    chunks = extract_chunk_paras(chunk_size, chapter)

    for i, chunk in enumerate(chunks):
        print("Chunk", i + 1)
        print("\n\n".join(chunk))
        print("chunk words:", len("\n\n".join(chunk).split()))
        print("----")

import csv
import random
import re

import nltk
import numpy as np
import pandas as pd

from utils.chat import askChatGPT


def sum_raw(raw):
    sys_sum_prompt = """You are a professional writer specialized in summarizing English articles，please generate a 
    summary of this document using the shortest possible language, retaining only the most important information.

To create a high-quality summary for an English article, there are several key points that you should follow. Here 
are the guidelines:

1. Identify the main idea: Understand the central theme or message of the article. This involves grasping the key 
points, arguments, or findings presented.

2. Be concise: Summarize the article in a succinct manner, focusing on the most important information. Avoid 
unnecessary details or excessive elaboration.

3. Maintain accuracy: Ensure that the summary accurately reflects the content and main points of the original 
article. Avoid introducing new ideas or misrepresenting the author's intent.

4. Prioritize relevance: Emphasize the information that is most relevant to the article's main idea. Remove any 
tangential or less significant details.

5. Maintain proper structure: Organize the summary logically, presenting the information in a coherent and cohesive 
manner. Consider using paragraph breaks to separate different ideas or sections."""

    user_sum_prompt = f"""Source document: [{raw}]\n
    summary:"""
    messages = [{"role": "system", "content": sys_sum_prompt}]
    d = {"role": "user", "content": user_sum_prompt}
    messages.append(d)
    result = askChatGPT(messages)

    return result

import re

from utils.chat import askChatGPT, askLongChatGPT


def gpt_eval(vanilla_gpt_doc, test_doc):
    prompt_sys = """You are a professional document review expert with a strong foundation in writing and extensive 
    experience in reviewing."""
    prompt_user = f"""Please compare the following two documents and analyze which one is better 
    simplified based on the factors of coherence, simplicity and faithfulness. 
    Document 1: [{vanilla_gpt_doc}]
    Document 2: [{test_doc}]

   In your analysis, please consider how well each document maintains:

  - Coherence: The logical flow and organization, ensuring smooth transitions between sentences and paragraphs.

  - Simplicity: The level of complexity and difficulty, aiming to make the content more accessible through plain language, shorter sentences, and simpler vocabulary.  

  - Faithfulness: How well each document preserves the core meaning, key information, and intended message of the original document without distorting or misrepresenting them.

   Please note that you must provide some thoughts and analysis on comparing the two documents, and in the last line, present the improved simplified document. 
   Follow the output format: [Reasoning content: ...\n The better simplified document: (Document 1 or Document 2)]
   """

    messages = [{"role": "system", "content": prompt_sys}]
    d = {"role": "user", "content": prompt_user}
    messages.append(d)
    result = askLongChatGPT(messages)
    pattern = r"The better doc: (?P<doc>\w+ \d+)$"

    match = re.search(pattern, result, re.IGNORECASE)

    if match:
        doc = match.group("doc")
        if "document 1" in doc:
            return 1
        elif "document 2" in doc:
            return 2
        else:
            return 0


if __name__ == '__main__':
    gpt_doc = ""
    test_doc = ""
    result = gpt_eval(gpt_doc, test_doc)

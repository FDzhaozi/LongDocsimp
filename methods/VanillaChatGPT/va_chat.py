from utils.chat import askChatGPT


def raw_gpt(raw_text, mode):
    base_prompt = f"""You are a professional simplified article writer,I need you to simplify the language and 
    structure of the raw document to make it more accessible to pupils.  Replace complex words or phrases or 
    technical terms with simpler, more familiar words or terms, use more and shorter clauses, and reorganize clauses 
    to make them easier to read. Please note that this does not require you to write a summary of the document. You 
    need to simplify each sentence to generate the corresponding simplified document. Redundant or unimportant 
    sentences can be deleted. 
    Raw:[{raw_text}]
    Simplified:"""
    force_prompt = f"""As a text simplification writer, your task is to simplify the given text content: restate the 
    original text in simpler and easier to understand language without changing its meaning as much as possible. 
    You can change paragraph or sentence structure, remove some redundant information, and replace complex and uncommon 
    expressions with simple and common ones. It should be noted that the task of text simplification is completely different 
    from the task of text summarization, so you need to provide a simplified parallel version based on the original text, 
    rather than just providing a brief summary. 
    Raw:[{raw_text}]
    Simplified:"""
    one_shot = read_str_from_txt("one_shot_doc.txt")
    one_shot_prompt = one_shot + f"""Referring to the example of text simplification mentioned above, as a text simplification writer, 
    your task is to simplify the given text content: restate the original text in simpler and easier to understand language 
    without changing its meaning as much as possible. You can change paragraph or sentence structure, remove some redundant 
    information, and replace complex and uncommon expressions with simple and common ones.
    Raw:[{raw_text}]
    Simplified:
    """
    simp_doc = """"""
    if mode == "base":
        simp_messages = [{"role": "system", "content": """Please simplify the document according to my instructions"""}]
        d = {"role": "user", "content": base_prompt}
        simp_messages.append(d)
        simp_doc = askChatGPT(messages=simp_messages)
    elif mode == "force":
        simp_messages = [{"role": "system", "content": """Please simplify the document according to my instructions"""}]
        d = {"role": "user", "content": force_prompt}
        simp_messages.append(d)
        simp_doc = askChatGPT(messages=simp_messages)
    elif mode == "expert":
        simp_messages = [{"role": "system", "content": """Please simplify the document according to my instructions"""}]
        d = {"role": "user", "content": expert_prompt}
        simp_messages.append(d)
        simp_doc = askChatGPT(messages=simp_messages)
    return simp_doc

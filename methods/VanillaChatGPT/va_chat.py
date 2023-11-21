from utils.chat import askChatGPT


def raw_gpt(raw_text, mode):
    base_prompt = f"""You are a professional simplified article writer,I need you to simplify the language and structure of the raw document to make it more accessible to pupils. 
    Raw:[{raw_text}]
    Simplified:"""
    force_prompt = f"""You are a professional simplified article writer,I need you to simplify the language and 
    structure of the raw document to make it more accessible to pupils.  Replace complex words or phrases or 
    technical terms with simpler, more familiar words or terms, use more and shorter clauses, and reorganize clauses 
    to make them easier to read. Please note that this does not require you to write a summary of the document. You 
    need to simplify each sentence to generate the corresponding simplified document. Redundant or unimportant 
    sentences can be deleted. Raw:[{raw_text}]
    Simplified:"""
    expert_prompt = f"""I want you to act as a document simplification tool. I want you to replace my upper level 
    words and sentences with more  simplified high-frequency English words and sentences. Keep the meaning same, 
    but make the text more accessible and understandable for individuals with limited language proficiency. My 
    document is [{raw_text}]"""
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

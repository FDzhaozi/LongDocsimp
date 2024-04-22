import json
from utils import calc_tokens, read_json_file, rebuild_json_file, append_to_json_file







if __name__ == '__main__':
    path = r"newsela-result/newselaB_gpt35_oneshot_prompt.json"
    data = read_json_file(path)
    print(len(data))
    print(data[0].keys())

    file_pathA = 'newselaA_data_gpt35_oneshot_prompt.json'
    with open(file_pathA, 'w', encoding="utf-8") as fileA:
        pass

    file_pathB = 'newselaB_data_gpt35_oneshot_prompt.json'
    with open(file_pathB, 'w', encoding="utf-8") as fileB:
        pass

    less_1000 = 0
    more_1000 = 0
    for passage in data:
        complex_text = passage["complex_text"]
        if calc_tokens(complex_text) > 1200:
            more_1000 += 1
        else:
            less_1000 += 1

    print("less 1000: ", less_1000)
    print("more 1000: ", more_1000)


    for passage in data:
        complex_text = passage["complex_text"]
        if calc_tokens(complex_text) > 1200:
            append_to_json_file(passage, file_pathB)
        else:
            append_to_json_file(passage, file_pathA)

    rebuild_json_file('newselaA_data_gpt35_oneshot_prompt.json', 'newsela-result/newselaA_gpt35_oneshot_prompt.json')
    rebuild_json_file('newselaB_data_gpt35_oneshot_prompt.json', 'newsela-result/newselaB_gpt35_oneshot_prompt.json')





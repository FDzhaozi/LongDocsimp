from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc
import json
from tqdm import tqdm

try:
    import platform

    if platform.system() != "Windows":
        import readline
except ImportError:
    print("Install `readline` for a better experience.")


def main():
    chat_model = ChatModel()
    history = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            history = []
            torch_gc()
            print("History has been removed.")
            continue

        print("Assistant: ", end="", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(query, history):
            print(new_text, end="", flush=True)
            response += new_text
        print()

        history = history + [(query, response)]


chat_model = ChatModel()


def single_chat_with_model(query):
    response = ""
    history = []
    for new_text in chat_model.stream_chat(query, history):
        response += new_text

    torch_gc()
    return response


if __name__ == "__main__":
    # main()
    # with open(r"/home/fdz/Desktop/mistral/LLaMA-Factory-main/LLaMA-Factory-main/data/test_merged_discourse.json", "r",
    #           encoding="utf-8") as file:
    #     json_data = json.load(file)
    #
    #
    # for data in tqdm(json_data,desc="processing"):
    #     data["discourse_output"] = single_chat_with_model(data["instruction"] + data["input"])
    #     with open("test_finetune_discourse_mistral_out.json", "a", encoding="utf-8") as file:
    #         json.dump(data, file, ensure_ascii=False, indent=4)

    # with open("test_fine_tuned_discourse_mistral_out.json", "w", encoding="utf-8") as file:
    #     json.dump(json_data, file, ensure_ascii=False, indent=4)

    question = "导致摄像头识别车道线困难的原因"
    answer = single_chat_with_model(question)
    print(answer)

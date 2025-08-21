import os
import itertools
# from rich.console import Console, Group
# from rich.syntax import Syntax
# from rich.markdown import Markdown
# from rich.padding import Padding
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation.streamers import BaseStreamer
from huggingface_hub import login


# green_llm = "\n\033[0;32m<llm>\033[0;0m "


# class CallbackStreamer(BaseStreamer):
#     console = Console()

#     def __init__(self, tokenizer):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.first = True
#         self.current = green_llm  # current line or current code
#         self.in_code = False  # not implemented

#     def put(self, value):
#         decoded = self.tokenizer.batch_decode(value, skip_special_tokens=True)[0]
#         if self.first:  # first token
#             print(green_llm, end="", flush=True)
#             self.first = False
#             return
#         self.current += decoded
#         if not self.in_code:
#             if decoded and "\n" not in decoded:
#                 print(decoded, end="", flush=True)
#             else:  # new line start or end
#                 if "*" in self.current:
#                     print("\r", end='', flush=True)  # reset prev line
#                     if green_llm in self.current:
#                         out = Markdown(self.current)
#                     else:
#                         out = Padding(Markdown(self.current), (0, 0, 0, len("<llm> ")))
#                     self.console.print(out, end='')
#                 print(decoded, end="", flush=True)
#                 print(" " * len("<llm> "), end='', flush=True)  # for next line
#                 self.current = ""
#             if not decoded:
#                 print(flush=True)

#     def end(self):
#         print("\n", end='', flush=True)


initial_chat = [
        # {'role': 'system', 'content': "You're a helpful chatbot that answers questions and solves the tasks you're given. You're direct and only expand on a topic if this is requested by the user. You only provide correct information and say it when you don't know the answer."},
        # {'role': 'system', 'content': "You're a helpful chatbot that does anything Sofia asks you to. You are honest, curious, and don't hallucinate."},
        {'role': 'system', 'content': "You're a helpful chatbot that helps Sofia complete the code she asks, which will generally be repeating a certain structure with small changes each time."},
        {'role': 'assistant', 'content': "Hi Sofia! How can I help you today?"},
    ]


class CallbackStreamer(BaseStreamer):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.first = True

    def put(self, value):
        decoded = self.tokenizer.batch_decode(value, skip_special_tokens=True)[0]
        if self.first:  # first token
            print("\n\033[0;32m<llm>\033[0;0m ", end="")
            self.first = False
            return
        print(decoded, end='', flush=True)
        if decoded and decoded[-1] == "\n":
            print(" " * len("<llm> "), end='', flush=True)

    def end(self):
        print()
        pass


def main(model_name: str, streaming: bool = True, quantize: bool = True) -> None:
    """
    Support an interactive chat with the model in the terminal.

    Args:
        model_name: name of the model for use in huggingface
        streaming: if true, then it shows the output as it's being generated
                   if false, then it uses beam search to select the best response
        quantize: if true, uses 8bit quantization to load the model
    """
    bnb_config = BitsAndBytesConfig(load_in_8bit=quantize)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map='auto', quantization_config=bnb_config, 
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map='auto')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    for i in itertools.count():
        if i == 0 or user_input == "r":
            chat = initial_chat.copy()
            mode = "streaming" if streaming else "beam search"
            print(
                f"\nYou're chatting with {model_name}. "
                "Press 'r' to rest chat. "
                f"Press 'c' to change modes. Current mode is '{mode}'. "
                "Press 'q' to exit."
            )
            print("\n\033[0;32m<llm>\033[0;0m " + chat[-1]['content'])
        user_input = input("\n\033[0;32m<user>\033[0;0m ")
        if user_input == "r":
            continue
        elif user_input == "c":
            streaming = not streaming
            mode = "streaming" if streaming else "beam search"
            print(f"\nMode changed to '{mode}'.")
            continue
        elif user_input == "q":
            break
        chat.append({'role': 'user', 'content': user_input})
        chat_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(chat_text, padding=True, return_tensors='pt').to('cuda')
        if streaming:
            streamer = CallbackStreamer(tokenizer)
            output = model.generate(
                **inputs, do_sample=True,
                streamer=streamer,
                max_new_tokens=512,
            )
            response = tokenizer.decode(
                output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True,
            )[0]
        else:
            output = model.generate(
                **inputs, do_sample=True,
                num_beams=10,
                # num_return_sequences=10,
                max_new_tokens=512,
            )
            response = tokenizer.decode(
                output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True,
            )
            print("\n\033[0;32m<llm>\033[0;0m " + response)
        chat.append({'role': 'assistant', 'content': response})


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    hugkey = os.getenv('HUGGINGFACE_KEY_LLAMA')
    login(hugkey)

    model_name = "meta-llama/LLama-3.2-3B-Instruct"
    main(model_name, quantize=True)

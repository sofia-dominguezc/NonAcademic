import os
import sys
import threading
import signal
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
)

def main():
    # Load environment
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    hf_key = os.getenv('HUGGINGFACE_KEY_LLAMA')
    if hf_key is None:
        print("Hugginface key not found in .env")
        sys.exit(1)

    # Arguments / Settings
    model_name = input("Model (default meta-llama/LLama-3.2-3B-Instruct): ") or 'meta-llama/LLama-3.2-3B-Instruct'
    quant = input("Quantization (none, 8bit) [default 8bit]: ") or '8bit'
    mode = ''
    while mode not in ['1', '2']:
        print("Choose mode:\n1) Streaming text\n2) Beam search")
        mode = input("Mode: ")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_key)
    model_args = {'device_map': 'auto'}
    if quant == '8bit':
        model_args['load_in_8bit'] = True
    print(f"Loading model {model_name} ({quant})...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_args, use_auth_token=hf_key)

    # Chat history
    history = []
    history.append({ 'role': 'system', 'content':
        "You're a helpful chatbot that answers questions and solves the tasks you're given."
        "You're direct and only expand on a topic if this is requested by the user."
        "You only provide correct information and say it when you don't know the answer." })
    history.append({ 'role': 'assistant', 'content': 'Hi Sofia! How can I help you today?' })
    print("Assistant: Hi Sofia! How can I help you today?")

    stop_signal = False
    def _signal_handler(sig, frame):
        nonlocal stop_signal
        stop_signal = True
    signal.signal(signal.SIGINT, _signal_handler)

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == '/quit':
            print("Exiting.")
            break
        history.append({ 'role': 'user', 'content': user_input })

        # Prepare inputs
        inputs = [dict(role=m['role'], content=m['content']) for m in history]
        prompt = tokenizer(
            ''.join(f"{m['content']}\n" for m in inputs), return_tensors='pt'
        ).to(model.device)

        if mode == '1':
            # Streaming mode
            det = ''
            while det not in ['y', 'n']:
                det = input("Deterministic (greedy)? [y/n]: ")
            do_sample = (det == 'n')
            top_k = None
            if do_sample:
                top_k = int(input("Top-k sampling (e.g. 50): ") or 50)

            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generate_kwargs = {
                'max_new_tokens': 256,
                'streamer': streamer,
                'do_sample': do_sample,
            }
            if top_k:
                generate_kwargs['top_k'] = top_k

            stop_signal = False
            # Launch generation in thread
            thread = threading.Thread(
                target=model.generate,
                kwargs={**prompt, **generate_kwargs}
            )
            thread.start()
            print("Assistant: ", end='', flush=True)
            try:
                # Wait until done or interrupted
                while thread.is_alive():
                    if stop_signal:
                        print("\n[Generation stopped by user]")
                        break
                thread.join()
            except Exception:
                pass
            # Note: partial output is already printed by streamer
            history.append({ 'role': 'assistant', 'content': '<streamed output>' })

        else:
            # Beam search mode
            beams = int(input("Beam size (default 5): ") or 5)
            returns = min(5, beams)
            out = model.generate(
                **prompt,
                max_new_tokens=256,
                num_beams=beams,
                num_return_sequences=returns,
                early_stopping=True
            )
            texts = [tokenizer.decode(o, skip_special_tokens=True) for o in out]
            # main answer
            main = texts[0]
            print(f"Assistant: {main}")
            history.append({ 'role': 'assistant', 'content': main })
            if returns > 1:
                view = input("View other beam answers? [y/n]: ")
                if view.lower() == 'y':
                    for idx, alt in enumerate(texts[1:], start=2):
                        print(f"{idx}) {alt}")
                        choose = input("Select this to respond or enter to skip: ")
                        if choose.strip() == str(idx):
                            history.append({ 'role': 'assistant', 'content': alt })
                            break

if __name__ == '__main__':
    main()

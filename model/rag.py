import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
import sys, os
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

config = {
    "model": "THUDM/GLM-4-9B-0414",
    "chunk_size": 500,
    "chunk_overlap": 100,
    "similarity_mode": "cosine",
    "threshold": 0.1,
    "top_k": 3
}


ref_directory  = "/home/loliterrorist/Desktop/shoppingHelper/reference_file"

def load_files(dir):
    files_content  = []
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        try:
            with open(filepath, "r") as f:
                files_content.append(f.read())
        except Exception as e:
            logging.error(f"there was an error while opening {filename}")     
            logging.error(e)
    return files_content        

# def retrieve
if __name__ == "__main__":
    text = load_files(ref_directory)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code = True)
    model = AutoModel.from_pretrained(config["model"], trust_remote_code = True)
    user_input = input()
    message = {"role": "user", "content": user_input}
    history = []
    response, history = model.chat(tokenizer, user_input, history = history)
    print(response)
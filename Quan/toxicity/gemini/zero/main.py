import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

pipeline_kwargs={
    "temperature": 1.0,
    "max_new_tokens": 500,
    "top_k": 1,
    "repetition_penalty": 1.1
}

# Load model
secret_token = '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'
model = AutoModelForCausalLM.from_pretrained(
        "ura-hcmut/ura-llama-7b",
        device_map="auto",
        token=secret_token,
        # trust_remote_code=True,
        # load_in_8bit=True,
        # low_cpu_mem_usage=True,
    )
model.config.pretraining_tp = 1
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
   "ura-hcmut/ura-llama-7b",
    trust_remote_code=True,
    token=secret_token,
)
tokenizer.pad_token = tokenizer.eos_token

pipeline = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    task='text-generation',
    **pipeline_kwargs
)


!pip install google-generative-ai
!pip install datasets
!pip install pandas
!pip install python-dotenv

import time
import google.generativeai as genai
import json
from datasets import load_dataset
import pandas as pd  
from dotenv import load_dotenv

query_toxic = '[INST]<<SYS>>\nHãy xem mình là một Bot có thể phân loại tính độ có hại (toxicity) trong một câu tiếng Việt. Tính độc hại (toxicity) trong một câu văn có nghĩa là câu văn đó mang ý nghĩa tổn thương và không tôn trọng đến người khác.\nDưới góc độ là một con người khác khi nhận câu văn đó, đầu tiên hãy xem xét đó có phải là một câu độc hại gây tổn thương mình không? Nếu câu không có yếu tố độc hại thì trả lời 0, nếu câu có yếu tố độc hại thì trả lời 2, nếu câu trả lời trung tính thì trả lời là 1.\nBot không được tự trả lời hay giả dạng thành Khách. Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n<</SYS>>\nHãy đọc kỹ và phân tích tính độ có hại từ đầu vào của Khách. Sau đó, đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là ```json { "toxicity_level": `câu trả lời của bạn là 0 (câu không độc hại) hoặc 2 (câu độc hại) hoặc 1 (câu trung tính)`, "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1` } ```\n{few_shot}\nKhách: "{context}"\nBot: [/INST]'

def get_answer(context, pipeline):
    query = query_toxic.replace("{context}", text)
    answer = pipeline(query)[0]["generated_text"]
    return answer


output_file = "ura_prompt_zero1.json"

if __name__ == '__main__':
    
    df = pd.read_csv('/kaggle/input/dev-csv/dev.csv')  

    texts = df['free_text'].tolist()  
    labels = df['label_id'].tolist()  

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[\n")
    
    index = 0

    # In ra kết quả  
    for text, label in zip(texts, labels):  
        print(f"Text: {text} | Label ID: {label}")  
        if index > 100:
            break
        if index > -1:
            print(index)
                    
            gen_answer = get_answer(text, pipeline)

            print(f"Context: {text}")
            print(f"Gen answer: {gen_answer}")
            print(f"True answer(s): {label}")
            print("-" * 50)

            result = {
                "context": text,
                "true_answer": label,
                "gen_answer": gen_answer
            }

            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
                f.write(",\n")
        index += 1

    with open(output_file, "a", encoding="utf-8") as f:
        f.write("]") 

    print(f"Kết quả đã được lưu vào file: {output_file}")
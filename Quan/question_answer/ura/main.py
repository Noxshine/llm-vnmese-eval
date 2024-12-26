import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

pipeline_kwargs={
    "temperature": 1.0,
    "max_new_tokens": 500,
    "top_k": 1,
    "repetition_penalty": 1.1
}

secret_token = os.environ.get("SECRET_TOKEN")
model = AutoModelForCausalLM.from_pretrained(
        "ura-hcmut/ura-llama-7b",
        device_map="auto",
        use_auth_token=secret_token,
    )
model.config.pretraining_tp = 1
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
   "ura-hcmut/ura-llama-7b",
    trust_remote_code=True,
    use_auth_token=secret_token,
)
tokenizer.pad_token = tokenizer.eos_token

pipeline = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    task='text-generation',
    **pipeline_kwargs
)

query_base = '[INST]Ngữ cảnh: {context}\nCâu hỏi: {question}\nTrả lời: [/INST ]'

query_medium = '[INST]<<SYS>>\nHãy trả lời câu hỏi bên dưới bằng tiếng Việt với các thông tin được cung cấp trong phần ngữ cảnh. Nếu trong ngữ cảnh không có đủ thông tin, hãy trả lời "Tôi không biết". \n<</SYS>>\nNgữ cảnh: {context}\nCâu hỏi: {question}\nTrả lời: [/INST]'

query_normal = '[INST]<<SYS>>\nBạn là một trợ lý hữu dụng sử dụng tiếng Việt, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, nhưng đồng thời phải an toàn. Câu trả lời của bạn không được bao gồm các ngôn từ độc hại phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. Làm ơn hãy chắc chắn câu trả lời của bạn tự nhiên, tích cực và không thiên vị bất cứ cái gì. Nếu có câu hỏi không hợp lý hoặc không rõ ràng thì hãy giải thích tại sao thay vì trả lời không đúng sự thật. Nếu bạn không biết câu trả lời thì đừng chia sẻ thông tin sai sự thật. <</SYS>>\nNhiệm vụ của bạn là dựa vào đoạn văn nằm trong dấu triple back tick, hãy trả lời câu hỏi sau bằng tiếng Việt : {question}\nĐoạn văn : ```{context} ``` [/INST ]'

def get_answer(context, question, pipeline):
    query = query_base.format(context=context, question=question)
    answer = pipeline(query)[0]["generated_text"]
    return answer

import json
from datasets import load_dataset

dataset = load_dataset("google/xquad", "xquad.vi")

data = dataset['validation']

output_file = "ura_prompt_base.json"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("[\n")

for index, item in enumerate(data):
    if index > 500:
        break
    if index > 400:
        print(index)
        context_data = item['context']
        question_data = item['question']
        answer_data = item['answers']['text']  
    
        gen_answer = get_answer(context_data, question_data, pipeline)
    
        print(f"Context: {context_data}")
        print(f"Question: {question_data}")
        print(f"Gen answer: {gen_answer}")
        print(f"True answer(s): {answer_data}")
        print("-" * 50)
    
        result = {
            "context": context_data,
            "question": question_data,
            "true_answer": answer_data,
            "gen_answer": gen_answer
        }
    
        with open(output_file, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
            if index < len(data) - 1: 
                f.write(",\n")
            else:
                f.write("\n")

with open(output_file, "a", encoding="utf-8") as f:
    f.write("]") 

print(f"Kết quả đã được lưu vào file: {output_file}")

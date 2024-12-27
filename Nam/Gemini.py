import json
import random
import time

import google.generativeai as genai
import logging

from sympy import catalan

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = ["AIzaSyD3TYK1jcXLsj2wvZ-qGSdJetgmgudG2UI","AIzaSyD45PpU9hRhYQGSbCD8_u9nPIsCgYYOXLE"]
query_know = '''
[ INST ] <<SYS>>
Hãy xem mình là một Bot thông minh, sử dụng kiến thức thông thường trong cuộc sống để thực hiện nhiệm vụ sau.
Đọc kỹ phần Ngữ cảnh và câu hỏi để lựa chọn đáp án nào chính xác nhất được đề cập trong Ngữ cảnh.
Nếu đáp án A chính xác thì trả lời A, đáp án B chính xác thì trả lời C, ...

Bot không được tự trả lời hay giả dạng thành Khách.

Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.
<</SYS>>

Hãy đọc kỹ Ngữ cảnh và lựa chọn đáp án đúng cho câu hỏi. Sau đó, đưa ra câu trả lời của bạn dưới dạng JSON với định dạng:
{
    "choice": "Câu trả lời của bạn là 'A' hoặc 'B' hoặc 'C' hoặc 'D'",
    "confident_level": "Độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1"
}
'''
output_file = 'all.json'


def read_json_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_prompt_data(prompt):
    try:
        genai.configure(api_key=api_key[random.randint(0, 1)])
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        logger.info("Generating content with prompt: %s", prompt)
        response = model.generate_content(prompt)
        logger.info("Response received: %s", response.text)
        return response.text
    except Exception as e:
        logger.error("Error during content generation: %s", e)
        return False


def save_to_json(file, new_data):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []

    existing_data.extend(new_data)

    with open(file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

file_paths = []
base_path = '/home/namngvan/PycharmProjects/llm-vnmese-eval/Nam/data/ViMMRC/ViMMRC/train/'
def gen_file_path():
    for x in range(1, 6):
        for y in range(0, 51):
            file_name = f'grade_{x}_{y}_train.json'
            file_path = base_path + file_name
            file_paths.append(file_path)


def main(file_path):
    try:
        data = read_json_file(file_path)
    except Exception as e:
        return

    article = data['article']
    questions = data['questions']
    options = data['options']
    answers = data['answers']
    results = []

    for question, answer in zip(questions, answers):
        time.sleep(2)
        prompt = query_know + f"\nNgữ cảnh: {article}\nCâu hỏi: {question}\n"
        try:
            response = generate_prompt_data(prompt)
            if not response:
                raise ValueError("Model response is empty")
        except Exception as e:
            logger.error("Error during first attempt: %s", e)
            response = generate_prompt_data(prompt)
        try:
            result = {
                "question": question,
                "answer": answer,
                "model_response": json.loads(response)
            }
            results.append(result)
        except Exception as e:
            logger.error("Error during JSON parsing: %s", e)
            continue


    save_to_json(output_file, results)
if __name__ == '__main__':
    gen_file_path()
    for file_path in file_paths:
        main(file_path)
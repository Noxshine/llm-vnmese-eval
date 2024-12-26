import os
import time

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import random

from VNMese_SPELL_CORRECTION_VSEC.main import calculate_mean, calculate_em
from metric import calculate_cer, calculate_wer, calculate_ced, calculate_wed

load_dotenv()


MLQA_PATH = '../data/mlqa-vi.json'
MLQA_MASKED_PATH = '../data/mlqa-masked-vi.json'
LOG_FILE_MLQA = 'log/MLQA_mask_prediction.txt'

prompt_mask_prediction = """[ INST ] <<SYS>>
Hãy xem mình là một Bot có thể thay thế token [MASKED] thành một từ thích hợp trong một câu tiếng Việt. Chú ý, 
Bot không thêm bớt các từ và ký tự trong câu, chỉ thay thế các token [MASKED] thành một từ thích hợp trong một
câu tiếng Việt. Bot không được tự trả lời hay giả dạng thành Khách. Đây là đoạn hội thoại giữa Bot và Khách.
<</SYS>>
Khách :  "Tôi thích ăn nhiều [MASKED] ăn khác nhau."  
Bot :  "Tôi thích ăn nhiều món ăn khác nhau."  
Khách :  "Chúng ta cần [MASKED] quản thực phẩm tốt hơn , trong thời tiết lạnh giá này."  
Bot :  "Chúng ta cần bảo quản thực phẩm tốt hơn , trong thời tiết lạnh giá này."  
Khách : "Hôm nay [MASKED] tiết rất đẹp."  
Bot : "Hôm nay thời tiết rất đẹp."  
Khách : "{query}"
Bot: [/INST] """

def get_prompt(query):
    return prompt_mask_prediction.format(query=query)

def generate_prompt_data(prompt):
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(prompt)

        return response.text

        # return clean_json_string(response.text) if response and response.text else "Không có câu trả lời."
    except Exception as e:
        print(f"Lỗi khi kiểm tra tìm kiếm với Gemini: {e}")
        return False

def preprocess_generate_data(input_string):
    input_string = input_string.strip()
    input_string.replace("\n", "")
    input_string.replace(" ,", ",")
    if input_string.startswith('"') and input_string.endswith('"'):
        return input_string[1:-1]  # Remove the surrounding quotes
    return input_string  # Return as-is if no quotes

def eval_MLQA(PATH_DIR):
    df = pd.read_json(PATH_DIR, lines=True)

    # create log file
    if not os.path.exists(LOG_FILE_MLQA):
        os.makedirs(os.path.dirname(LOG_FILE_MLQA), exist_ok=True)

    '''
    @metric
    '''
    EM = 0
    CER = 0
    WER = 0
    CED = 0
    WED = 0

    for i in range(0, 5):
        data = df.iloc[i]

        gr_truth = data['text']
        masked_text = data['masked_text']

        predict_sentence = preprocess_generate_data(generate_prompt_data(get_prompt(masked_text)))

        # print(text)
        # print(predict)

        # calculate EM
        EM = calculate_mean(EM, calculate_em(predict_sentence, gr_truth, masked_text), i)
        print("EM: ", EM)

        # calculate CER
        CER = calculate_mean(CER, calculate_cer(predict_sentence, gr_truth), i)
        print("CER: ", CER)

        # calculate WER
        WER = calculate_mean(WER, calculate_wer(predict_sentence, gr_truth), i)
        print("WER: ", WER)

        # calculate CED
        CED = calculate_mean(CED, calculate_ced(predict_sentence, gr_truth), i)
        print("CED: ", CED)

        # calculate WED
        WED = calculate_mean(WED, calculate_wed(predict_sentence, gr_truth), i)
        print("WED: ", WED)



        with open(LOG_FILE_MLQA, 'a', encoding='utf-8') as f:
            f.write("---------------------------\n")
            f.write(f"[No{i+1}]\n")
            f.write(f"[WRONG]: {masked_text}\n")
            f.write(f"[TRUTH]: {gr_truth}\n")
            f.write(f"[PRED] : {predict_sentence}\n")
            f.write(f"EM: {EM}\n")
            f.write(f"CER: {CER}\n")
            f.write(f"WER: {WER}\n")
            f.write(f"CED: {CED}\n")
            f.write(f"WED: {WED}\n")
            f.write("---------------------------\n")

        time.sleep(20)

def calculate_em(predict_sentence, gr_truth, masked_text):
    predict_sentence = predict_sentence.split()
    gr_truth = gr_truth.split()
    masked_text = masked_text.split()

    # total
    total_mask = 0
    predict_true_mask = 0

    for i in range(0, len(masked_text)):
        if masked_text[i] == "[MASKED]":
            total_mask = total_mask + 1
            if gr_truth[i] == predict_sentence[i]:
                predict_true_mask = predict_true_mask + 1

    return predict_true_mask/total_mask

def mask_random_word(sentence, mask_percentage=0.1):
    words = sentence.split()  # Split the sentence into words
    num_words_to_mask = max(1, int(len(words) * mask_percentage))  # Calculate how many words to mask
    indices_to_mask = random.sample(range(len(words)), num_words_to_mask)  # Randomly select indices to mask

    for index in indices_to_mask:
        words[index] = "[MASKED]"

    return ' '.join(words)  # Join the remaining words back into a sentence

if __name__ == '__main__':

    '''
    Read raw mlqa file and save to json
    '''
    # df = pd.read_json(MLQA_PATH)
    #
    # datas = df["data"]
    # paragraphs = [data["paragraphs"] for data in datas]
    # context = [p["context"] for para in paragraphs for p in para]
    #
    # df = pd.DataFrame({'text': context})

    '''
    Apply random [MASKED]
    '''
    # df = pd.read_json(ViQuad_PATH, lines=True)
    # df['masked_text'] = df['text'].apply(mask_random_word)
    # df.to_json(MLQA_MASKED_PATH, orient='records', lines=True, force_ascii=False)


    '''
    Test 
    '''
    eval_MLQA(MLQA_MASKED_PATH)
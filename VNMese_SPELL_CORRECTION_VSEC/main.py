import json
import os
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

from metric import calculate_cer, calculate_wer, calculate_ced, calculate_wed, calculate_mean

load_dotenv()

VSEC_PATH = '../data/VSEC.jsonl'


prompt_spell_correction = """[INST] <<SYS>>Hãy xem mình là một Bot có thể tìm và sửa các lỗi sai chính tả có trong một câu tiếng Việt. 
    Chú ý, Bot không chỉnh sửa, tự động xoá khoảng trắng hay thêm bớt các từ trong câu, chỉ sửa các từ bị sai chính tả. 
    Bot không tự trả lời hay giả dạng thành Khách. 
    Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách. <</SYS>> 
    Khách :  "Tôi thíc ăn nhìu món ăn khác nhau."  
    Bot :  "Tôi thích ăn nhiều món ăn khác nhau."  
    Khách :  "Chúng ta cần bão quãn thực phẩm tốt hơn , trong thời tiết lạnh giá này."  
    Bot :  "Chúng ta cần bảo quản thực phẩm tốt hơn , trong thời tiết lạnh giá này."  
    Khách : "Hôm nay thời tiếc rất đep."  
    Bot : "Hôm nay thời tiết rất đẹp."  
    Khách : "{query}"
    Bot: [/INST] """

def generate_prompt_data(prompt):
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY_1"))
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(prompt)

        return preprocess_generate_data(response.text)

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

def prompt_spell_correction_loading(query):
    return prompt_spell_correction.format(query=query)


def VSEC_evaluate(dataset):

    # create log file
    if not os.path.exists(LOG_FILE_VSEC):
        os.makedirs(os.path.dirname(LOG_FILE_VSEC), exist_ok=True)

    for i in range(600, 800):
        wr_num = 0

        # data loading
        data = dataset[i]

        # get detail data
        wrong_sentence = data['text']   # full sentence with spell incorrect

        print("[WRONG] :", wrong_sentence)
        annotations = data['annotations']

        # get ground truth
        gr_truth = wrong_sentence.split(' ')
        for anno in annotations:
            if not anno["is_correct"]:
                id = anno["id"]
                gr_truth[id - 1] = anno["alternative_syllables"][0]

                wr_num = wr_num + 1

        gr_truth = " ".join(gr_truth) # groud truth
        gr_truth = gr_truth.strip() # remove blank first and last
        print("[TRUTH] :", gr_truth)

        # predict
        prompt = prompt_spell_correction_loading(wrong_sentence)
        predict_sentence = generate_prompt_data(prompt) # predict
        print("[PRED] :", predict_sentence)

        with open(LOG_FILE_VSEC, 'a', encoding='utf-8') as f:
            f.write("---------------------------\n")
            f.write(f"[No{i+1}]\n")
            f.write(f"[WRONG]: {wrong_sentence}\n")
            f.write(f"[TRUTH]: {gr_truth}\n")
            f.write(f"[PRED] : {predict_sentence}\n")
            f.write("---------------------------\n")

        time.sleep(20)


def calculate_em(pred, target, wr_num):

    """
    Calculate extract match
    :param pred: predict sentence
    :param target: ground truth sentence
    :param wr_num: total false of input data
    :return:
    """
    wr_pre = 0

    pred = pred.split(" ")
    target = target.split(" ")

    for i in range(0, len(pred)):
        if pred[i] != target[i]:
            wr_pre = wr_pre + 1

    print("number wrong: ", wr_num, " number predict true: ", wr_num - wr_pre)
    return (wr_num - wr_pre) / wr_num

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
        return data

def convert_txt_to_json(file_path, output_path):
    result = []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()

    record = {}
    for line in lines:
        line = line.strip()
        if line.startswith("[WRONG]"):
            record['wrong'] = line.split(":", 1)[1].strip()
        elif line.startswith("[TRUTH]"):
            record['truth'] = line.split(":", 1)[1].strip()
        elif line.startswith("[PRED]"):
            record['pred'] = line.split(":", 1)[1].strip()
        elif line.startswith("[No"):
            if record:
                result.append(record)
                record = {}
    if record:  # Add the last record if exists
        result.append(record)

    with open(output_path, 'w') as json_file:
        json.dump(result, json_file, indent=4)

LOG_FILE_VSEC = 'log/VSEC_spell_correction.txt'
LOG_FILE_VSEC_JSON = 'log/VSEC_spell_correction.json'

if __name__ == '__main__':

    # load dataset
    # dataset = load_data(VSEC_PATH)

    # evaluate EM
    # VSEC_evaluate(dataset)

    # save txt to json file
    # convert_txt_to_json(LOG_FILE_VSEC, LOG_FILE_VSEC_JSON)

    '''
     @metric - fix to last value whenever continue running
     '''
    EM = 0
    CER = 0
    WER = 0
    CED = 0
    WED = 0

    df = pd.read_json(LOG_FILE_VSEC_JSON)

    for index, row in df.iterrows():

        if index > 10 :
            break

        wr = row["wrong"]
        trh = row["truth"]
        pred = row["pred"]

        CER = calculate_mean(CER, calculate_cer(pred, trh), index)
        WER = calculate_mean(WER, calculate_wer(pred, trh), index)
        CED = calculate_mean(CED, calculate_ced(pred, trh), index)
        WED = calculate_mean(WED, calculate_wed(pred, trh), index)

    print(CER)
    print(WER)
    print(CED)
    print(WED)

    '''
    0.0044814339754256334
    0.019886363636363636
    0.6363636363636364
    0.42565748957778759
    '''

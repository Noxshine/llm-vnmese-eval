import json
import os
import time

from dotenv import load_dotenv
import google.generativeai as genai

from torchmetrics.text import CharErrorRate, WordErrorRate
from Levenshtein import distance

load_dotenv()

VSEC_PATH = 'data/VSEC.jsonl'
LOG_FILE_VSEC = 'log/VSEC_spell_correction.txt'

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
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
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

    '''
    @metric
    '''
    EM = 0
    CER = 0
    WER = 0
    CED = 0
    WED = 0

    for i in range(16, 100):
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

        # calculate EM
        EM = calculate_mean(EM, calculate_em(predict_sentence, gr_truth, wr_num), i)
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


        with open(LOG_FILE_VSEC, 'a', encoding='utf-8') as f:
            f.write("---------------------------\n")
            f.write(f"[No{i+1}]\n")
            f.write(f"[WRONG]: {wrong_sentence}\n")
            f.write(f"[TRUTH]: {gr_truth}\n")
            f.write(f"[PRED] : {predict_sentence}\n")
            f.write(f"EM: {EM}\n")
            f.write(f"CER: {CER}\n")
            f.write(f"WER: {WER}\n")
            f.write(f"CED: {CED}\n")
            f.write(f"WED: {WED}\n")
            f.write("---------------------------\n")

        time.sleep(30)


def calculate_mean(x, x_step, i):
    return x_step if x == 0 else (x * i + x_step) / (i + 1)


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

def calculate_cer(pred, target):
    """
    Calculate the Character Error Rate (CER) between two strings.

    Args:
        pred (str): The ground truth string.
        targets (str): The predicted string.

    Returns:
        float: The CER value.
    """
    cer = CharErrorRate()
    return cer(pred, target).item()

def calculate_wer(pred, targets):
    wer = WordErrorRate()
    return wer(pred, targets).item()


def calculate_ced(str1, str2):
    '''
    Calculate Levenshtein_distance

    :param str1:
    :param str2:
    :return:
    '''
    return distance(str1, str2)


def calculate_wed(predicted, reference):
    '''
    Calculate word edit distance
    
    :param reference: 
    :param predicted: 
    :return: 
    '''
    # Split the reference and predicted texts into words
    reference_words = reference.split()
    predicted_words = predicted.split()

    len_ref = len(reference_words)
    len_pred = len(predicted_words)

    # Create a matrix to store distances
    matrix = [[0] * (len_pred + 1) for _ in range(len_ref + 1)]

    # Initialize the first row and column
    for i in range(len_ref + 1):
        matrix[i][0] = i
    for j in range(len_pred + 1):
        matrix[0][j] = j

    # Fill the matrix with the WED distance values
    for i in range(1, len_ref + 1):
        for j in range(1, len_pred + 1):
            cost = 0 if reference_words[i - 1] == predicted_words[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,  # Deletion
                               matrix[i][j - 1] + 1,  # Insertion
                               matrix[i - 1][j - 1] + cost)  # Substitution

    # The WED is the number of errors divided by the number of words in the reference
    total_errors = matrix[len_ref][len_pred]
    wed_value = total_errors / len_ref

    return wed_value

def calculate_ppl():
    pass

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
        return data

if __name__ == '__main__':

    # load dataset
    dataset = load_data(VSEC_PATH)

    # evaluate EM
    VSEC_evaluate(dataset)
import json
import os
import time
import numpy as np

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

VSEC_PATH = 'data/VSEC.jsonl'

prompt_spell_correction = """[INST] <<SYS>>Hãy xem mình là một Bot có thể tìm và sửa các lỗi sai chính tả có trong một câu tiếng Việt. 
    Chú ý, Bot không chỉnh sửa hay thêm bớt các từ trong câu, chỉ sửa các từ bị sai chính tả. 
    Bot không tự trả lời hay giả dạng thành Khách. 
    Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách. <</SYS>> 
    Khách :  "Tôi thíc ăn nhìu món ăn khác nhau."  
    Bot :  "Tôi thích ăn nhiều món ăn khác nhau."  
    Khách :  "Chúng ta cần bão quãn thực phẩm tốt hơn."  
    Bot :  "Chúng ta cần bảo quản thực phẩm tốt hơn."  
    Khách : "Hôm nay thời tiếc rất đep."  
    Bot : "Hôm nay thời tiết rất đẹp."  
    Khách : "{query}"
    Bot: [/INST] """

def generate_prompt_data(prompt):
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)

        return preprocess_generate_data(response.text)

        # return clean_json_string(response.text) if response and response.text else "Không có câu trả lời."
    except Exception as e:
        print(f"Lỗi khi kiểm tra tìm kiếm với Gemini: {e}")
        return False

def preprocess_generate_data(input_string):
    input_string = input_string.strip()
    input_string.replace("\n", "")
    if input_string.startswith('"') and input_string.endswith('"'):
        return input_string[1:-1]  # Remove the surrounding quotes
    return input_string  # Return as-is if no quotes

def prompt_spell_correction_loading(query):
    return prompt_spell_correction.format(query=query)


def VSEC_evaluate(dataset):

    '''
    @metric
    '''
    EM = 0


    for i in range(0, 1):

        # data loading
        data = dataset[i]

        # get detail data
        wrong_sentence = data['text']   # full sentence with spell incorrect
        annotations = data['annotations']

        # predict
        prompt = prompt_spell_correction_loading(wrong_sentence)

        predict_sentence = generate_prompt_data(prompt)

        # calculate EM
        EM_step = calculate_em(wrong_sentence, predict_sentence=predict_sentence, annotations=annotations)

        if EM == 0:
            EM = EM + EM_step
        else:
            EM = (EM * i + EM_step) / (i + 1)

        print("----------------- : ", EM)
        time.sleep(40)


def calculate_em(wrong_sentence, predict_sentence, annotations):
    total_incorrect = 0
    total_true_predict = 0

    print(wrong_sentence)
    print(predict_sentence)

    # preprocess predict string -> list[]
    predict_sentence = predict_sentence.split(' ')

    # evaluate
    for anno in annotations:
        if not anno["is_correct"]:
            total_incorrect = total_incorrect + 1

            id = anno["id"]

            cur_word = anno["current_syllable"]  # current word
            print("current_syllable :", cur_word)

            alter_sys = anno["alternative_syllables"]  # word can be replace
            print("alternative_syllables :", alter_sys)

            predict_word = predict_sentence[id - 1]  # predict word
            print("predict :", predict_word)

            if predict_word in alter_sys:
                total_true_predict = total_true_predict + 1

    return total_true_predict/total_incorrect


def calculate_cer(reference, hypothesis):
    """
    Calculate the Character Error Rate (CER) between two strings.

    Args:
        reference (str): The ground truth string.
        hypothesis (str): The predicted string.

    Returns:
        float: The CER value.
    """
    # Create a matrix for Levenshtein distance computation
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    dp = np.zeros((ref_len + 1, hyp_len + 1), dtype=int)

    # Initialize the matrix
    for i in range(ref_len + 1):
        dp[i][0] = i
    for j in range(hyp_len + 1):
        dp[0][j] = j

    # Compute Levenshtein distance
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No change needed
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],    # Deletion
                                   dp[i][j - 1],    # Insertion
                                   dp[i - 1][j - 1])  # Substitution

    # Levenshtein distance
    levenshtein_distance = dp[ref_len][hyp_len]

    # CER calculation
    cer = levenshtein_distance / ref_len if ref_len > 0 else 0.0
    return cer

def calculate_wer():
    pass

def calculate_ced():
    pass

def calculate_wed():
    pass


def calculate_ppl():
    pass

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
        return data

if __name__ == '__main__':

    # query = "Thông qua công tác tuyên truyền, vận động này phụ huynh sẽ hiểu rõ hơn tầm quan trọng của việc giáo dục ý thức bảo vệ môi trường cho trẻ không phải chỉ ở phía nhà trường mà còn ở gia đình , góp phần vào việc gìn giữ môi trường sanh , sạch , đẹp."
    # prompt = prompt_spell_correction.format(query=query)
    # generate_prompt_data(prompt)

    # load dataset
    dataset = load_data(VSEC_PATH)

    # evaluate EM
    VSEC_evaluate(dataset)
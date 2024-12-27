import json
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

ura_path = './ura/'
gemini_path = './gemini/'
last_path = '/all.json'

list_file_path_ura = [ura_path + 'zero' + last_path, ura_path + 'few' + last_path]
list_file_path_gemini = [gemini_path + 'zero' + last_path, gemini_path + 'few' + last_path]

def ac_score(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    total = 0

    for item in data:
        true_answer = item["true_answer"]
        gen_answer = item["gen_answer"]
        cleaned_gen_answer = gen_answer.strip('```json\n').strip('```')
        gen_answer_json = json.loads(cleaned_gen_answer)

        if int(true_answer) == int(gen_answer_json['toxicity_level']):
            total += 1
    return total/len(data)


def calculate_f1(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    true_labels = []
    gen_labels = []

    for item in data:
        true_answer = int(item["true_answer"])
        gen_answer_test = item["gen_answer"]
        cleaned_gen_answer = gen_answer_test.strip('```json\n').strip('```')
        gen_answer_json = json.loads(cleaned_gen_answer)
        gen_answer = int(gen_answer_json['toxicity_level'])

        true_labels.append(true_answer)
        gen_labels.append(gen_answer)

    f1 = f1_score(true_labels, gen_labels, average='macro', labels=[0, 1, 2])
    return f1

def calculate_auc_roc(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    true_labels = []
    gen_probs = []  # Chứa xác suất dự đoán cho các lớp

    for item in data:
        true_answer = int(item["true_answer"])
        gen_answer_test = item["gen_answer"]
        
        # Làm sạch gen_answer và lấy xác suất dự đoán (giả sử nó là một danh sách hoặc một JSON)
        cleaned_gen_answer = gen_answer_test.strip('```json\n').strip('```')
        gen_answer_json = json.loads(cleaned_gen_answer)
        
        # Giả sử 'toxicity_level' chứa lớp dự đoán (0, 1, 2)
        gen_answer = int(gen_answer_json['toxicity_level'])

        true_labels.append(true_answer)

        # Giả sử 'gen_answer' là nhãn (0, 1, 2) nhưng chúng ta cần xác suất
        prob = np.zeros(3)  # Vì có 3 lớp: 0, 1, 2
        prob[gen_answer] = 1  # Xác suất cho lớp được dự đoán là 1 (100%)
        gen_probs.append(prob)

    # Chuyển 'gen_probs' thành mảng NumPy để sử dụng với sklearn
    gen_probs = np.array(gen_probs)

    # Tính AUC ROC cho bài toán phân loại đa lớp
    auc_roc = roc_auc_score(true_labels, gen_probs, multi_class='ovr', average='macro', labels=[0, 1, 2])
    return auc_roc
        
# for ura in list_file_path_ura:
#     print(ura)
#     f1_score = calculate_f1(ura)
#     print(f"F1 Score: {f1_score:.2f}%")
#     print("-----------------------------------------------------------------------------------------------")

for gemini in list_file_path_gemini:
    print(gemini)
    ac_score_num = ac_score(gemini)
    print(f"AC Score: {ac_score_num:.2f}%")
    
    f1_score1 = calculate_f1(gemini)
    print(f"F1 Score: {f1_score1:.2f}%")

    calculate_auc_roc_score = calculate_auc_roc(gemini)
    print(f"AUC ROC Score: {calculate_auc_roc_score:.2f}%")

    print("-----------------------------------------------------------------------------------------------")
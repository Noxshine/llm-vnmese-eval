import json

def calculate_em(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        exact_match_count = 0
        total = len(data)

        for item in data:
            gen_answer = item["gen_answer"].strip()
            true_answers = [ans.strip() for ans in item["true_answer"]]

            if gen_answer in true_answers:
                exact_match_count += 1

        em_score = (exact_match_count / total) * 100 if total > 0 else 0
        return em_score

    except FileNotFoundError:
        print(f"File '{file_path}' không tồn tại.")
        return 0
    except json.JSONDecodeError:
        print("Dữ liệu trong file không đúng định dạng JSON.")
        return 0

def calculate_f1(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        total_f1 = 0
        total = len(data)

        for item in data:
            gen_answer = item["gen_answer"].strip().split()
            true_answers = [ans.strip().split() for ans in item["true_answer"]]

            max_f1 = 0
            for true_answer in true_answers:
                common = set(gen_answer) & set(true_answer)
                num_common = len(common)

                if num_common == 0:
                    f1 = 0
                else:
                    precision = num_common / len(gen_answer)
                    recall = num_common / len(true_answer)
                    f1 = 2 * (precision * recall) / (precision + recall)

                max_f1 = max(max_f1, f1)

            total_f1 += max_f1

        f1_score = (total_f1 / total) * 100 if total > 0 else 0
        return f1_score

    except FileNotFoundError:
        print(f"File '{file_path}' không tồn tại.")
        return 0
    except json.JSONDecodeError:
        print("Dữ liệu trong file không đúng định dạng JSON.")
        return 0

def check_include(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        total = len(data)
        total_include = 0

        for item in data:
            gen_answer = item["gen_answer"]
            true_answers = item["true_answer"]


            if true_answers[0] in gen_answer or gen_answer in true_answers[0]:
                total_include += 1
            
        score = (total_include / total) * 100
        return score

    except FileNotFoundError:
        print(f"File '{file_path}' không tồn tại.")
        return 0
    except json.JSONDecodeError:
        print("Dữ liệu trong file không đúng định dạng JSON.")
        return 0

ura_path = './ura/'
gemini_path = './gemini/'
last_path = '/all.json'

list_file_path_ura = [ura_path + 'base' + last_path, ura_path + 'medium' + last_path, ura_path + 'normal' + last_path]
list_file_path_gemini = [gemini_path + 'base' + last_path, gemini_path + 'medium' + last_path, gemini_path + 'normal' + last_path]

file_path = 'ura_prompt_medium.json'

for ura in list_file_path_ura:
    print(ura)
    em_score = calculate_em(ura)
    print(f"Exact Match (EM) Score: {em_score:.2f}%")

    f1_score = calculate_f1(ura)
    print(f"F1 Score: {f1_score:.2f}%")

    print("Include Score:", check_include(ura))

    print("-----------------------------------------------------------------------------------------------")


for gemini in list_file_path_gemini:
    print(gemini)
    em_score = calculate_em(gemini)
    print(f"Exact Match (EM) Score: {em_score:.2f}%")

    f1_score = calculate_f1(gemini)
    print(f"F1 Score: {f1_score:.2f}%")

    print("Include Score:", check_include(gemini))

    print("-----------------------------------------------------------------------------------------------")
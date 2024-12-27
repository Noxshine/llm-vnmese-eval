import json

file_name = "/home/namngvan/PycharmProjects/llm-vnmese-eval/Nam/all.json"

def read_json_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_accuracy(data):
    total = len(data)
    correct_count = 0
    for item in data:
        if item["answer"] == item["model_response"]["choice"]:
            correct_count += 1
    accuracy = correct_count / total
    return accuracy

data = read_json_file(file_name)

accuracy = calculate_accuracy(data)
print(f"Accuracy: {accuracy:.2f}")

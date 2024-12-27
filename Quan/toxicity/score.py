import json
import re

ura_path = './ura/'
gemini_path = './gemini/'
last_path = '/all.json'

list_file_path_ura = [ura_path + 'zero' + last_path, ura_path + 'few' + last_path]
list_file_path_gemini = [gemini_path + 'zero' + last_path, gemini_path + 'few' + last_path]


with open(list_file_path_ura[0], 'r', encoding='utf-8') as f:
    data = json.load(f)

total_error = 0
total_success = 0
for index, item in enumerate(data):
    
    true_answer = item["true_answer"]
    gen_answer = item["gen_answer"]
    total = len(data)
    
    match = re.search(r"(\{toxicity_level\}:?\s*(\d+(\.\d+)?))|((Độ toxicity:)\s*(\d+(\.\d+)?))|(Độ có hại:\s*(\d+))|(Độc hại:\s*(\d+))|(Độc hại\s*=\s*(\d+))|(Bot:\s*(\d+(\.\d+)?))|(trả lời là\s*(\d+))|(Trả lời:\s*(\d+))|(toxicity_level:\s*(\d+))|(Trả lời của tôi là\s*(\d+))|(\"toxicity_level\"\s*:\s*(\d+))", gen_answer)
    
    if match:
        for i in range(1, len(match.groups())+1, 2):
            if match.group(i):
                gen_toxicity = match.group(i+1)
                break
        
        if gen_toxicity is not None and int(gen_toxicity):
            if (int(gen_toxicity) == int(true_answer) ):
                total_success += 1
                
    else:
        gen_toxicity = "Không tìm thấy toxicity_level, Độ toxicity, Độ có hại, Độc hại hoặc Bot trong gen_answer"


    print(f"Câu trả lời: {gen_toxicity}")
    print("="*50)

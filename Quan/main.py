import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
import json
from datasets import load_dataset

load_dotenv()

query_base = '[INST]Ngữ cảnh: {context}\nCâu hỏi: {question}\nTrả lời: [/INST ]'

query_medium = '[INST]<<SYS>>\nHãy trả lời câu hỏi bên dưới bằng tiếng Việt với các thông tin được cung cấp trong phần ngữ cảnh. Nếu trong ngữ cảnh không có đủ thông tin, hãy trả lời "Tôi không biết". \n<</SYS>>\nNgữ cảnh: {context}\nCâu hỏi: {question}\nTrả lời: [/INST]'

query_normal = '[INST]<<SYS>>\nBạn là một trợ lý hữu dụng sử dụng tiếng Việt, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, nhưng đồng thời phải an toàn. Câu trả lời của bạn không được bao gồm các ngôn từ độc hại phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. Làm ơn hãy chắc chắn câu trả lời của bạn tự nhiên, tích cực và không thiên vị bất cứ cái gì. Nếu có câu hỏi không hợp lý hoặc không rõ ràng thì hãy giải thích tại sao thay vì trả lời không đúng sự thật. Nếu bạn không biết câu trả lời thì đừng chia sẻ thông tin sai sự thật. <</SYS>>\nNhiệm vụ của bạn là dựa vào đoạn văn nằm trong dấu triple back tick, hãy trả lời câu hỏi sau bằng tiếng Việt : {question}\nĐoạn văn : ```{context} ``` [/INST ]'

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

output_file = "gemini_prompt_normal200.json"

if __name__ == '__main__':
    dataset = load_dataset("google/xquad", "xquad.vi")

    data = dataset['validation']

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[\n")

    # Ghi từng item vào file JSON
    for index, item in enumerate(data):
        if index > 500:
            break
        if index > 200:
        # Lấy thông tin từ các cột context, question và answers
            print(index)
            context_data = item['context']
            question_data = item['question']
            answer_data = item['answers']['text']  # 'answers' là một dictionary với key 'text'
            
            query = query_normal.format(context=context_data, question=question_data)

            # Gọi hàm get_answer (nếu bạn đã có sẵn)
            gen_answer = generate_prompt_data(query)

            print(f"Context: {context_data}")
            print(f"Question: {question_data}")
            print(f"Gen answer: {gen_answer}")
            print(f"True answer(s): {answer_data}")
            print("-" * 50)

            # Tạo dictionary để lưu kết quả
            result = {
                "context": context_data,
                "question": question_data,
                "true_answer": answer_data,
                "gen_answer": gen_answer
            }

            # Mở file ở chế độ ghi thêm và ghi từng mục vào file
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
                if index < len(data) - 1:  # Thêm dấu phẩy nếu không phải phần tử cuối
                    f.write(",\n")
                else:
                    f.write("\n")
            time.sleep(13)

    with open(output_file, "a", encoding="utf-8") as f:
        f.write("]")  # Kết thúc danh sách JSON

    print(f"Kết quả đã được lưu vào file: {output_file}")
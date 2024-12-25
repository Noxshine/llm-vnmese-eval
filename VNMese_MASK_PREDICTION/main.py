import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


VSEC_PATH = '../data/VSEC.jsonl'
LOG_FILE_VSEC = 'log/VSEC_mask_prediction.txt'

prompt_mask_prediction = """[ INST ] <<SYS>>
Hãy xem mình là một Bot có thể thay thế token [MASKED] thành một từ thích hợp trong một câu tiếng Việt. Chú ý, 
Bot không thêm bớt các từ và ký tự trong câu, chỉ thay thế các token [MASKED] thành một từ thích hợp trong một
câu tiếng Việt. Bot không được tự trả lời hay giả dạng thành Khách. Đây là đoạn hội thoại giữa Bot và Khách.
<</SYS>>
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

        return response.text

        # return clean_json_string(response.text) if response and response.text else "Không có câu trả lời."
    except Exception as e:
        print(f"Lỗi khi kiểm tra tìm kiếm với Gemini: {e}")
        return False




if __name__ == '__main__':
    pass
    # query = " "
    # prompt_mask_prediction.format(query=query)
    #
    # print(generate_prompt_data(prompt_mask_prediction))
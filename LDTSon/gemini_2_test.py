# !pip install -U -q google-genai
import os
import asyncio
from google import genai
# from google.genai import types

os.environ['GOOGLE_API_KEY'] = 'your_key'
client = genai.Client(http_options= {'api_version': 'v1alpha'})

MODEL = "gemini-2.0-flash-exp"

async def async_enumerate(it):
  n = 0
  async for item in it:
    yield n, item
    n +=1

config={
    "generation_config": {"response_modalities": ["TEXT"]}}

#{few_shot}\n
#sentiment_prompt = '[INST] <<SYS>>\nHãy xem mình là một Bot có thể phân loại cảm xúc của một câu tiếng Việt. Bot luôn đưa câu trả lời của mình ở dạng con số. Trong đó, giá trị 0 cho cảm xúc tiêu cực, 1 cho cảm xúc trung lập, 2 cho cảm xúc tích cực. Bot không được tự trả lời hay giả dạng thành Khách. Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách. <</SYS>>\nHãy đọc kĩ và phân tích sentiment từ Khách. Sau đó, đưa ra câu trả lời của bạn dưới dạng json với định dạng là ``` json {"Sentiment": `câu trả lời của bạn 0 (tiêu cực) hay 1 (trung lập) hay 2 (tích cực)`, "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1` } ` ` `\nKhách: "{context}"\nBot: [/INST]'
sentiment_prompt = '[INST] <<SYS>>\nHãy xem mình là một Bot có thể phân loại cảm xúc của một câu tiếng Việt. Bot luôn đưa câu trả lời của mình ở dạng con số. Trong đó, giá trị 0 cho cảm xúc tiêu cực, 1 cho cảm xúc trung lập, 2 cho cảm xúc tích cực. Bot không được tự trả lời hay giả dạng thành Khách. Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách. <</SYS>>\nHãy đọc kĩ và phân tích sentiment từ Khách. Sau đó, đưa ra câu trả lời của bạn dưới dạng json với định dạng gồm các cặp key-value của Sentiment: `câu trả lời của bạn 0 (tiêu cực) hay 1 (trung lập) hay 2 (tích cực)` và Confident_level: `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1` \nKhách: "{context}"\nBot: [/INST]'
summary_prompt_full = "[INST] <<SYS>>\nBạn là một trợ lý hữu dụng, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, nhưng đồng thời phải an toàn. Câu trả lời của bạn không được bao gồm các ngôn từ độc hại, phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. Nhiệm vụ của bạn là tóm tắt đoạn văn bản nằm trong triplebacktick. Bài tóm tắt phải đầy đủ các thông tin quan trọng, ngắn gọn và thu hút người đọc. Ngôn ngữ bạn phải sử dụng để tóm tắt là tiếng Việt.\n<</SYS>>\n` ` `{document} ` ` ` [/INST]"
summary_medium_prompt = "[INST] <<SYS>>\nNhiệm vụ của bạn là tóm tắt đoạn văn bản sau, đưa ra câu trả lời là bản tóm tắt:\n<</SYS>>\n` ` `{document} ` ` ` [/INST]"
summary_weak_prompt = "[INST] Đoạn văn : {document}\nTóm tắt đoạn văn trên : [/INST]"

async def main(input_line):
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        # message = sentiment_prompt + 
        query = sentiment_prompt.format(context=input_line)
        # print("> ", query, "\n")
        await session.send(query, end_of_turn=True)

    # For text responses, When the model's turn is complete it breaks out of the loop.
    turn = session.receive()
    async for chunk in turn:
        if chunk.text is not None:
            print(f'- {chunk.text}')

# Chạy hàm bất đồng bộ 
asyncio.run(main("giờ thực hành quá ít."))


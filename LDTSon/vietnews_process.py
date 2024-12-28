import os

def extract_news_parts(file_path, abstract_file, details_file):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Initialize placeholders
    title = ""
    abstract = ""
    content = []

    # Extract parts based on structure
    section = "title"
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            if section == "title":
                section = "abstract"
            elif section == "abstract":
                section = "content"
            continue
        
        if section == "title":
            title += line
        elif section == "abstract":
            abstract += line
        elif section == "content":
            content.append(line)
    
    # Combine details into one part
    details = f"{title}\n" + "\n".join(content)

    # Write abstract to its file
    with open(abstract_file, 'w', encoding='utf-8') as file:
        file.write(abstract)

    # Write details to its file
    with open(details_file, 'w', encoding='utf-8') as file:
        file.write(details)


#Path to the uploaded file
folder_path = 'C:/Users/THAI SON/Desktop/GenAI_Foundation/data_vietnews/test_tokenized'
folder_abstract_path = 'C:/Users/THAI SON/Desktop/GenAI_Foundation/data_vietnews/abstract'
#os.makedirs(folder_abstract_path)
folder_details_path = 'C:/Users/THAI SON/Desktop/GenAI_Foundation/data_vietnews/details'
#os.makedirs(folder_details_path)

#VietNews: https://doi.org/10.1109/NICS48868.2019.9023886
#read files in folder_path
for filename in os.listdir(folder_path):
    if filename.endswith('.txt.seg'):
        file_path = os.path.join(folder_path, filename)
        abstract_file_path = os.path.join(folder_abstract_path, filename.replace('.txt.seg', '_abstract.txt'))
        details_file_path = os.path.join(folder_details_path, filename.replace('.txt.seg', '_details.txt'))
        extract_news_parts(file_path, abstract_file_path, details_file_path)
        # print(f"{filename} written")
# print("Details:\n", details)
import json
import re
import pandas as pd
from hanspell import spell_checker
from tqdm import tqdm

def preprocess_sentence(sentence):
    words = sentence.replace("\n", " ")
    words = re.sub(r'[^가-힣0-9a-zA-Z\s]', ' ', words)
    words = re.sub(' +', ' ', words)
    return words

def apply_bio_tags(tokens, aspects):
    subordinate_aspects = ['기능/효과', '편의성/활용성']
    delete_aspects = ['피부타입']
    aspect_tags = ['O'] * len(tokens)

    back_aspects = []
    for aspect in aspects:
        if aspect['Aspect'] not in delete_aspects:
            if aspect['Aspect'] not in subordinate_aspects:
                aspect_text = spell_checker.check(aspect['SentimentText'])
                if not aspect_text:
                    return None
                aspect_tokens = preprocess_sentence(aspect_text.checked).split()
                
                start_index = -1
                for i in range(len(tokens) - len(aspect_tokens) + 1):
                    if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                        start_index = i
                        break
                
                if start_index != -1:
                    aspect_tags[start_index] = 'B-' + aspect['Aspect']
                    for j in range(1, len(aspect_tokens)):
                        aspect_tags[start_index + j] = 'I-' + aspect['Aspect']
            else:
                back_aspects.append(aspect)
    
    for aspect in back_aspects:
        aspect_text = spell_checker.check(aspect['SentimentText'])
        if not aspect_text:
            return None
        aspect_tokens = preprocess_sentence(aspect_text.checked).split()
        
        start_index = -1
        for i in range(len(tokens) - len(aspect_tokens) + 1):
            if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                start_index = i
                break
        
        if start_index != -1:
            no_tag = True
            for j in range(len(aspect_tokens)):
                if aspect_tags[start_index + j] != 'O':
                    no_tag = False
                    break
            
            if no_tag:  # 덮어쓰기 불가
                aspect_tags[start_index] = 'B-' + aspect['Aspect']
                for j in range(1, len(aspect_tokens)):
                    aspect_tags[start_index + j] = 'I-' + aspect['Aspect']
    
    return aspect_tags

def adjust_raw_text(raw_text, aspects):
    for aspect in aspects:
        sentiment_text = aspect['SentimentText']
        start_index = raw_text.find(sentiment_text)
        if start_index != -1:
            before = raw_text[:start_index]
            after = raw_text[start_index + len(sentiment_text):]
            raw_text = before + ' ' + sentiment_text + ' ' + after
    return ' '.join(raw_text.split())


### BIO tagging with py-hanspell
file_path = "../Dataset/cosmetic.json"
with open(file_path, encoding="UTF-8") as f:
    data = json.loads(f.read())

result_list = []
column_list = ["Index", "Word", "Aspect"]

for d in tqdm(data):
    text = adjust_raw_text(d['RawText'], d['Aspects'])
    text = spell_checker.check(text)
    if text:
        tokens = preprocess_sentence(text.checked).split()
        a_tags = apply_bio_tags(tokens, d['Aspects'])
        if a_tags is not None:
            index = d['Index']

            for token, a_tag in zip(tokens, a_tags):
                result_list.append([index, token, a_tag])

df = pd.DataFrame(result_list, columns=column_list)
df.to_csv("../Dataset/Aspect/cosmetic_aspect.csv", encoding='utf-8', index=False)
from openai import OpenAI
import pandas as pd
import json
import re
from tqdm import tqdm


# load data
test = pd.read_csv('Dataset/Sentiment/cosmetic_Sentiment_test.csv')
test_indices = test['Index'].unique()

file_path = "Dataset/cosmetic_hanspell.json"
with open(file_path, encoding="UTF-8") as f:
    data = json.loads(f.read())
    data = pd.DataFrame(data)

test_df = data[data['Index'].apply(int).isin(test_indices)]
test_df = pd.DataFrame(test_df)[['Index', 'RawText', 'Aspects']]
test_df['GPT'] = ''


# prompt
prompt = '''
Q: "유통기한도 넉넉하고  구성도 많아서 선물 하기 좋네요.   만족합니다."
A: [
    {
        "Aspect": "유통기한",
        "SentimentText": "유통기한도 넉넉하고",
        "SentimentPolarity": "1"
    },
    {
        "Aspect": "제품구성",
        "SentimentText": "구성도 많아서 선물 하기 좋네요.",
        "SentimentPolarity": "1"
    }
]

Q: "대용량으로 넉넉하게 사용할 수 있고 무난하고 순한 편이네요 제품 구성은 좋으나 가격면에서 타 사이트 대비 다소 비싼 면이 있습니다. "
A: [
    {
        "Aspect": "용량",
        "SentimentText": "대용량으로 넉넉하게 사용할 수 있고",
        "SentimentPolarity": "1"
    },
    {
        "Aspect": "자극성",
        "SentimentText": "무난하고 순한 편이네요",
        "SentimentPolarity": "1"
    },
    {
        "Aspect": "제품구성",
        "SentimentText": "제품 구성은 좋으나 ",
        "SentimentPolarity": "1"
    },
    {
        "Aspect": "가격",
        "SentimentText": "가격면에서 타 사이트 대비 다소 비싼 면이 있습니다.",
        "SentimentPolarity": "-1"
    }
]

Q: "커버는 별로 화장후 바람불면 끈적임이 있어 머리카락이 달라붙어요.  화면에서 쇼호스트가 티슈붙여 보여줄땐 안붙던데ㅠ 저는 .. 왜그런걸까요?",
A: [
    {
        "Aspect": "커버력",
        "SentimentText": "커버는 별로",
        "SentimentPolarity": "-1"
    },
    {
        "Aspect": "흡수력",
        "SentimentText": "화장후 바람불면 끈적임이 있어 머리카락이 달라붙어요.",
        "SentimentPolarity": "-1"
    }
]

'''


# openai
OPENAI_API_KEY = "KEY" # write your key here
client = OpenAI(api_key=OPENAI_API_KEY)

skip_idx = []
for i, row in tqdm(test_df[2540:].iterrows(), total=len(test_df[2540:])):
  rawText = row['RawText']

  content = f'''{prompt}
Q: {rawText}
A: '''

  system_content = """All Aspect Classes: '유통기한', '제품구성', '자극성', '가격', '흡수력', '제형', '발림성', '품질', '기능/효과', '사용감', '윤기/피부(톤)', '용기', '성분', '탄력', '편의성/활용성', '디자인', '색상', '밀착력/접착력', '커버력', '탈모개선', '세정력', '지속력/유지력', '두피보호', '머릿결관리', '향/냄새', '용량/사이즈', '거품력', '스타일링효과', '발색력', '이염', '분사력', '보습력/수분감/쿨링감'

All SentimentPolarity Classes: '0', '-1', '1'"""

  try:
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": content}
      ]
    )
    
    test_df.loc[i, 'GPT'] = response.choices[0].message.content
  except:
    skip_idx.append(i)


# extract 
test_df['extracted_GPT'] = ''

success_num, except_num = 0
for i, row in test_df.iterrows():
    gpt = row['GPT']
    matches = re.findall(r'\[([^\]]+)\]', gpt)
    if not matches:
        matches = re.findall(r'\[([^\]]+)\]', '[' + gpt + ']')
        
    if matches:
        match = matches[0]
        if match:
            match = match.replace('}\n{', '},\n{')
            match = match.replace('}\n    {', '},\n    {')
            if match.endswith('},\n'): match = match[:-2]
            json_string = f'[{match}]'
            try:
                data_list = json.loads(json_string)
                test_df.at[i, 'extracted_GPT'] = data_list
                success_num += 1
            except:
                print(0, match)
                except_num += 1
        else:
            print(1, match)
            except_num += 1
print(success_num, except_num, len(test_df))

index_, text_, aspect_, sentiment_ = [], [], [], []

good, bad = 0, 0
for i, row in test_df.iterrows():
    for asp in row['extracted_GPT']:
        try:
            check = row['Index'], asp['SentimentText'], asp['Aspect'], asp['SentimentPolarity']
            index_.append(str(row['Index']))
            text_.append(str(asp['SentimentText']))
            aspect_.append(str(asp['Aspect']))
            sentiment_.append(str(asp['SentimentPolarity']))
            good += 1
        except:
            print(asp)
            bad += 1
print(good, bad)

gpt_ABSA = pd.DataFrame({'Index': index_, 'Text': text_, 'Aspect': aspect_, 'Sentiment': sentiment_})
gpt_ABSA.to_csv('Results/gpt_ABSA.csv', index=False)
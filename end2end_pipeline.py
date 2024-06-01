import os
import json
import pickle
import pandas as pd
import torch
import itertools
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from tqdm import tqdm

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from konlpy.tag import Okt
okt = Okt()

from bareunpy import Tagger
API_KEY="koba-TIO6IFA-6JWEYBI-TVWRV3Y-UPRSO7I"
tagger = Tagger(API_KEY, 'localhost')

##### Aspect BIO #####
class CustomAspectDataset(Dataset):
    def __init__(self, data, tag2id, tokenizer, max_length=128):
        data = data.reset_index().rename(columns={'index': 'original_index'})
        self.data = data.groupby('Index').apply(lambda x: (x['Word'].tolist(), x['original_index'].tolist()), include_groups=False)
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, indices = self.data.iloc[idx]

        encoding = self.tokenizer(words, return_offsets_mapping=True, is_split_into_words=True, padding='max_length', truncation=True, max_length=self.max_length)
        word_ids = encoding.word_ids()
        offsets = encoding.pop('offset_mapping')

        index_ids = [-100] * len(word_ids)
        current_word = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None:
                if word_idx != current_word:
                    current_word = word_idx
                    if offsets[i][0] == 0:
                        index_ids[i] = indices[current_word]

        encoding['indices'] = index_ids

        return {key: torch.tensor(val) for key, val in encoding.items()}

def predict_aspect(loader, model):
    model.eval()
    all_predictions = []
    index_mapping = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Aspect Prediction'):
            batch = {key: val.to(device) for key, val in batch.items()}
            indices = batch.pop('indices')
            outputs = model(**batch)
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            active = (indices != -100)
            predictions = predictions[active]
            indices = indices[active]
            
            index_mapping.extend(indices.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return all_predictions, index_mapping

# tag2id, id2tag
with open('Dataset/Aspect/Aspect_tag2id.pickle', 'rb') as f:
    Aspect_tag2id = pickle.load(f)
    id2tag = {Aspect_tag2id[key]: key for key in Aspect_tag2id}

# tokenizer, model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
aspect_model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(Aspect_tag2id)).to(device)
aspect_model.load_state_dict(torch.load('Model/best_model_Aspect.pth')) # load best model

do_allproducts = True
if do_allproducts:
    # data (ver. all)
    aspect_df = pd.read_csv('Dataset/Aspect/cosmetic_aspect_test.csv')
    product_name = None
else:
    # data (ver. one product)
    aspect_df = pd.read_csv('Dataset/Aspect/cosmetic_aspect_replace.csv')
    with open('Dataset/cosmetic.json', encoding="UTF-8") as f:
        data = json.loads(f.read())
        data = pd.DataFrame(data)
        data['Index'] = data['Index'].apply(int)
    product_name = '주름관리 멀티밤 더블세트'
    aspect_df = aspect_df.merge(data[['Index', 'ProductName']], on='Index')
    aspect_df = aspect_df[aspect_df['ProductName'].str.contains(product_name, na=False)]

# dataset, dataloader
aspect_dataset = CustomAspectDataset(aspect_df, Aspect_tag2id, tokenizer, 256)
aspect_loader = DataLoader(aspect_dataset, batch_size=16, shuffle=False)

# predict
aspect_predictions, aspect_indices = predict_aspect(aspect_loader, aspect_model)

aspect_prediction_df = pd.DataFrame({
    'Row_Index': aspect_indices,
    'Prediction': [id2tag[ap] for ap in aspect_predictions]
})

df = aspect_df.merge(aspect_prediction_df, left_index=True, right_on='Row_Index').drop(['Row_Index'], axis=1)

# extract sentences
results = []
current_sentence, current_tag, current_index = "", None, None

for _, row in tqdm(df.iterrows(), total=len(df), desc='Extract Sentences 1'):
    if row['Prediction'].startswith('B-'):
        if current_sentence:
            results.append((current_index, current_sentence, current_tag))
        current_sentence = row['Word']
        current_tag = row['Prediction'][2:]
        current_index = row['Index']
    elif row['Prediction'].startswith('I-'):
        if current_tag and current_tag == row['Prediction'][2:]:
            current_sentence += " " + row['Word']
        else:  # 앞의 태그와 다른 경우 새 문장 시작
            if current_sentence:
                results.append((current_index, current_sentence, current_tag))
            current_sentence = row['Word']
            current_tag = row['Prediction'][2:]
            current_index = row['Index']
    else:
        if current_sentence:
            results.append((current_index, current_sentence, current_tag))
            current_sentence = ""
            current_tag = None
            current_index = None
if current_sentence:
    results.append((current_index, current_sentence, current_tag))

F = [[], ['MMA'], ['MMD'], ['MMN']]
N = [['NF'], ['NP'], ['NNP'], ['NNG'], ['NNG', 'NNG'], ['NNG', 'XSN'], ['VA', 'ETN'], ['VV', 'ETN']]
J = [[], ['JX'], ['JC']]
nouns = [f + n + j for f, n, j in itertools.product(F, N, J)]

indices, sentences, tags = [], [], []
for i, (index, sentence, tag) in tqdm(enumerate(results), total=len(results), desc='Extract Sentences 2'):
    # POS_tag = [t for _, t in okt.pos(sentence)]
    POS_tag = [t for _, t in tagger.tags([sentence]).pos()]
    if POS_tag in nouns and i + 1 < len(results):
        next_index, next_sentence, _ = results[i + 1]
        if index == next_index:
            sentence += ' ' + next_sentence

    indices.append(index)
    sentences.append(sentence)
    tags.append(tag)

extracted_df = pd.DataFrame({'Index': indices, 'Text': sentences, 'Aspect': tags})


##### Sentiment Analysis #####
class CustomSentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.data.iloc[idx]['Text'],
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def predict_sentiment(loader, model):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Sentiment Prediction'):
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())

    return all_predictions

# model
sentiment_model = AutoModelForSequenceClassification.from_pretrained('Model/best_model_Sentiment', num_labels=3).to(device) # load best model

# data
sentiment_dataset = CustomSentimentDataset(extracted_df, tokenizer)
sentiment_loader = DataLoader(sentiment_dataset, batch_size=64, shuffle=False)

# predict
sentiment_predictions = predict_sentiment(sentiment_loader, sentiment_model)
extracted_df['Sentiment'] = sentiment_predictions
extracted_df.loc[extracted_df['Sentiment'] == 2, 'Sentiment'] = -1

# save
extracted_df.to_csv('Results/ABSA.csv', index=False, encoding='UTF-8')


##### WordCloud #####
do_wordcloud = False
if do_wordcloud:
    output_folder = "./Results/WordCloud"
    os.makedirs(output_folder, exist_ok=True) 
    korean_stopwords = set([
        '하다', '것', '그', '있다', '이다', '되다', '있다', '없다', '같다', '맞다'
    ])
    excluded_pos = {'Josa', 'Suffix', 'Punctuation', 'KoreanParticle', 'Conjunction'}

    def generate_word_cloud(data, filename, font_path="static/font/NanumGothic.ttf"):
        if data:
            wordcloud = WordCloud(
                width=1600, height=800,
                background_color='white',
                font_path=font_path,
                collocations=False
            ).generate_from_frequencies(data)
            
            plt.figure(figsize=(16, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(f'{output_folder}/{filename}')

    def filter_and_combine_data(data, aspect, display_columns):
        stopwords = korean_stopwords.union(aspect.split('/'))

        filtered_tokens = []
        for text in data[display_columns]:
            text = text.replace('잘 ', '').replace('돼다', '되다')
            pos_tags = okt.pos(text, norm=True, stem=True)
            pos_tags = [(word, pos) for word, pos in pos_tags if pos not in excluded_pos and word not in stopwords]

            i = 0
            while i < len(pos_tags):
                word, tag = pos_tags[i]
                if i < len(pos_tags) - 1:
                    next_word, next_tag = pos_tags[i + 1]
                    if word in ["안"] and next_tag in ["Verb", "Adjective"]:
                        filtered_tokens.append(word + next_word)
                        i += 1
                    elif tag in ["Verb", "Adjective"]:
                        if any(next_word.startswith(prefix) for prefix in ["않", "아니"]):
                            filtered_tokens.append(word.replace('다', '지') + next_word)
                        elif any(next_word.startswith(prefix) for prefix in ["모르"]):
                            if word[-2:] == '하다':
                                word = word[:-2] + '한지'
                            elif word[-1:] == '다':
                                jongseong = (ord(word[-2]) - 44032) % 588 % 28
                                if jongseong:
                                    word = word[:-1] + '은지'
                                else:
                                    word = word[:-2] + chr(ord(word[-2]) + 4) + '지'
                            filtered_tokens.append(word + next_word)
                        elif len(word) > 1:
                            filtered_tokens.append(word)
                        else:
                            i -= 1
                        i += 1
                    elif len(word) > 1:
                        filtered_tokens.append(word)
                elif len(word) > 1:
                    filtered_tokens.append(word)
                i += 1
        word_freq = Counter(filtered_tokens)
        return word_freq

    def _wordcloud(data, aspect, sentiment):
        specific_data = data[(data['Aspect'] == aspect) & (data['Sentiment'] == sentiment)]
        specific_freqs = filter_and_combine_data(specific_data, aspect, 'Text')
        if product_name is not None:
            generate_word_cloud(specific_freqs, f"wordcloud_{product_name}_{aspect.replace('/', '')}_{sentiment}.png")
        else:
            generate_word_cloud(specific_freqs, f"wordcloud_all_{aspect.replace('/', '')}_{sentiment}.png")

    aspect_name = '향/냄새'
    _wordcloud(extracted_df, aspect_name, 1)
    _wordcloud(extracted_df, aspect_name, 0)
    _wordcloud(extracted_df, aspect_name, -1)
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup

import pickle
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report


### load data
dir = 'Dataset/Aspect'
train_df = pd.read_csv(f'{dir}/cosmetic_aspect_train.csv')
valid_df = pd.read_csv(f'{dir}/cosmetic_aspect_valid.csv')
test_df = pd.read_csv(f'{dir}/cosmetic_aspect_test.csv')


### Dataset and Model
loss_fn = CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, data, tag2id, tokenizer, max_length=128):
        data = data.reset_index().rename(columns={'index': 'original_index'})
        self.data = data.groupby('Index').apply(lambda x: (x['Word'].tolist(), x['Aspect'].tolist(), x['original_index'].tolist()), include_groups=False)
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, labels, indices = self.data.iloc[idx]
        labels = [self.tag2id[label] for label in labels]

        encoding = self.tokenizer(words, return_offsets_mapping=True, is_split_into_words=True, padding='max_length', truncation=True, max_length=self.max_length)
        word_ids = encoding.word_ids()
        offsets = encoding.pop('offset_mapping')

        label_ids = [-100] * len(word_ids)
        index_ids = [-100] * len(word_ids)
        current_word = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None:
                if word_idx != current_word:
                    current_word = word_idx
                    if offsets[i][0] == 0:
                        label_ids[i] = labels[current_word]
                        index_ids[i] = indices[current_word]

        encoding['labels'] = label_ids
        encoding['indices'] = index_ids

        return {key: torch.tensor(val) for key, val in encoding.items()}

def test_model(loader, model, mode='test'):
    model.eval()
    total_loss = 0
    all_predictions, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=mode):
            indices = batch.pop('indices')
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            labels = batch['labels']
            
            active = (indices != -100)
            logits = logits[active]
            predictions = predictions[active]
            labels = labels[active]

            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    loss = total_loss / len(loader) if len(loader) > 0 else 0
    accuracy = sum(all_labels[i] == all_predictions[i] for i in range(len(all_labels))) / len(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    
    print(f"{mode} Loss: {loss:.4f}")
    if mode != 'test':
        print(f"{mode} Accuracy: {accuracy:.4f}")
        print(f"{mode} Precision: {precision:.4f}")
        print(f"{mode} Recall: {recall:.4f}")
        print(f"{mode} F1 Score: {f1:.4f}\n")
    else:
        print(classification_report(all_labels, all_predictions))

    return f1

def train_model(model, train_loader, valid_loader, num_epochs=10, patience=3):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_f1 = 0.0
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        # train
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc='train'):
            indices = batch.pop('indices')
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)

            logits = outputs.logits
            labels = batch['labels']
            
            active = (indices != -100)
            logits = logits[active]
            labels = labels[active]
            
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"epoch {epoch+1}: train loss {total_loss / len(train_loader)}")

        # valid
        valid_f1 = test_model(valid_loader, model, 'valid')
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            torch.save(model.state_dict(), 'Model/Aspect/best_roberta_Aspect.pth')
            no_improve_epochs = 0
            print(f"Checkpointing new best model with F1: {best_f1}")
        else:
            no_improve_epochs += 1
            print(f"No improvement in validation F1 for {no_improve_epochs} epochs")
        
        if no_improve_epochs >= patience:
            print("Early stopping triggered")
            break
        print()

unique_tags = train_df['Aspect'].unique()
tag2id = {unique_tags[i]: i for i in range(len(unique_tags))}

with open('Dataset/Aspect/Aspect_tag2id.pickle', 'wb') as f:
    pickle.dump(tag2id, f)

with open('Dataset/Aspect/Aspect_tag2id.pickle', 'rb') as f:
    Aspect_tag2id = pickle.load(f)

model_name = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(Aspect_tag2id)).to(device)

train_dataset = CustomDataset(train_df, Aspect_tag2id, tokenizer)
valid_dataset = CustomDataset(valid_df, Aspect_tag2id, tokenizer, 256)
test_dataset = CustomDataset(test_df, Aspect_tag2id, tokenizer, 256)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


### train
train_model(model, train_loader, valid_loader)


### test
model.load_state_dict(torch.load('Model/Aspect/best_roberta_Aspect.pth'))
test_model(test_loader, model)


### inference
def predict(loader, model):
    model.eval()
    all_predictions = []
    index_mapping = []

    with torch.no_grad():
        for batch in tqdm(loader):
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

id2tag = {Aspect_tag2id[key]: key for key in Aspect_tag2id}
test_predictions, test_indices = predict(test_loader, model)

predictions_df = pd.DataFrame({
    'Row_Index': test_indices,
    'Prediction': [id2tag[tp] for tp in test_predictions]
})

new_test_df = test_df.merge(predictions_df, left_index=True, right_on='Row_Index').drop(['Row_Index'], axis=1)
new_test_df.to_csv('Results/prediction.csv', encoding='UTF-8', index=False)
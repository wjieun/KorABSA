import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import evaluate
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report


# load data
dir = 'Dataset/Sentiment'
train_df = pd.read_csv(f'{dir}/cosmetic_Sentiment_train.csv')
valid_df = pd.read_csv(f'{dir}/cosmetic_Sentiment_valid.csv')
test_df = pd.read_csv(f'{dir}/cosmetic_Sentiment_test.csv')

train_df = train_df[train_df['Aspect'] != '피부타입']
valid_df = valid_df[valid_df['Aspect'] != '피부타입']
test_df = test_df[test_df['Aspect'] != '피부타입']

train_df.loc[train_df['Sentiment'] == -1, 'Sentiment'] = 2
valid_df.loc[valid_df['Sentiment'] == -1, 'Sentiment'] = 2
test_df.loc[test_df['Sentiment'] == -1, 'Sentiment'] = 2


# RandomUnderSampling
X = train_df.drop('Sentiment', axis=1)
y = train_df['Sentiment']

sampling_strategy = {1: 10000, 2: 10000, 0: 6128}
rus = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
X_resampled, y_resampled = rus.fit_resample(X, y)

train_df = pd.DataFrame(X_resampled, columns=X.columns)
train_df['Sentiment'] = y_resampled


# dataset and model
class CustomDataset(Dataset):
    def __init__(self, data, label, tokenizer, max_length=64):
        self.data = data
        self.label = label
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Text']
        label = self.data.iloc[idx][self.label]

        encoding = self.tokenizer.encode_plus(
            text,
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
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    m1 = evaluate.load('accuracy')
    m2 = evaluate.load('f1')

    acc = m1.compute(predictions=preds, references=labels)['accuracy']
    f1 = m2.compute(predictions=preds, references=labels, average='macro')['f1']

    return {'accuracy': acc, 'f1': f1}
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'klue/roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

label = 'Sentiment'
train_dataset = CustomDataset(train_df, label, tokenizer)
valid_dataset = CustomDataset(valid_df, label, tokenizer)
test_dataset = CustomDataset(test_df, label, tokenizer)


# train
output_dir = "./Model/checkpoints/sentiment"
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()


# predict and evaluate
def predict(loader, model):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())

    return all_predictions

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
predictions = predict(test_loader, model)
print(classification_report(test_df['Sentiment'].tolist(), predictions))
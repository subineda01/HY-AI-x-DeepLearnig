import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

# 예시 데이터 로드 (pandas DataFrame으로 가정)
data = pd.read_csv("/content/train.csv")

# 텍스트와 라벨 추출
sentences = data['text'].values
labels = data['label'].values

# 라벨 인코딩
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# 데이터셋 분리
X_train, X_val, y_train, y_val = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# 토크나이저 및 단어장 생성
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(X_train), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 데이터셋 클래스 정의
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(text)
        token_ids = [self.vocab[token] for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# 데이터셋 및 데이터로더 생성
train_dataset = TextDataset(X_train, y_train, vocab, tokenizer)
val_dataset = TextDataset(X_val, y_val, vocab, tokenizer)

def collate_batch(batch):
    text_list, label_list = [], []
    for _text, _label in batch:
        text_list.append(torch.tensor(_text, dtype=torch.long))
        label_list.append(torch.tensor(_label, dtype=torch.long))
    return pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"]), torch.stack(label_list)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

import torch.nn as nn
import torch.optim as optim
import optuna

# 모델 정의
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_units, num_classes, dropout_rate):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.lstm = nn.LSTM(embed_dim, lstm_units, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_units, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        x = self.fc(x)
        return x

# 하이퍼파라미터 튜닝 함수
def objective(trial):
    # 하이퍼파라미터 정의
    embed_dim = trial.suggest_int('embed_dim', 32, 256, step=32)
    lstm_units = trial.suggest_int('lstm_units', 32, 256, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    
    # 모델 초기화
    model = LSTMClassifier(vocab_size=len(vocab), embed_dim=embed_dim, lstm_units=lstm_units, num_classes=6, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 모델 학습
    model.train()
    for epoch in range(10):
        for texts, labels in train_dataloader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # 모델 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in val_dataloader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

# Optuna 튜너 실행
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# 최적 하이퍼파라미터 출력
best_params = study.best_params
print(f"Best parameters: {best_params}")

# 최적 하이퍼파라미터로 모델 학습
best_model = LSTMClassifier(vocab_size=len(vocab), 
                            embed_dim=best_params['embed_dim'], 
                            lstm_units=best_params['lstm_units'], 
                            num_classes=6, 
                            dropout_rate=best_params['dropout_rate']).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])

# 모델 학습
best_model.train()
for epoch in range(20):
    for texts, labels in train_dataloader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = best_model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 모델 평가
best_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for texts, labels in val_dataloader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = best_model(texts)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Validation Accuracy: {accuracy}')

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# 감정 레이블 정의
labels = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

# 커스텀 데이터셋 클래스
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 데이터 로드
def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    labels = [label2id[label] for label in df['label'].tolist()]
    return texts, labels

# 데이터셋 및 데이터로더 생성
def create_data_loader(texts, labels, tokenizer, max_len, batch_size):
    ds = EmotionDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=2)

# 모델 설정
def train_model(data_loader, model, device, optimizer, scheduler, num_epochs):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# 예측 함수
def predict(text, model, tokenizer, max_len, device):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask)
    _, prediction = torch.max(outputs.logits, dim=1)
    return id2label[prediction.item()]

# 메인 함수
def main():
    # 설정
    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    # 데이터 파일 경로
    DATA_FILE_PATH = 'path/to/your/data.csv'

    # 데이터 로드
    texts, labels = load_data(DATA_FILE_PATH)

    # 토크나이저 및 모델 로드
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=len(labels))

    # 데이터로더 생성
    data_loader = create_data_loader(texts, labels, tokenizer, MAX_LEN, BATCH_SIZE)

    # Optimizer 및 Scheduler 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(data_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # 모델 학습
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(data_loader, model, device, optimizer, scheduler, EPOCHS)

    # 예측 테스트
    test_text = "I am so happy today!"
    emotion = predict(test_text, model, tokenizer, MAX_LEN, device)
    print(f"Predicted emotion for '{test_text}': {emotion}")

if __name__ == "__main__":
    main()

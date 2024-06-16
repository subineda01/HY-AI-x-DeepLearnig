import os
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['HF_HOME'] = "/home/subin/AI+x/cache"
os.environ['MPLCONFIGDIR'] = "/home/subin/AI+x/matplotlib_cache"

import torch
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

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

def load_data(file_path):
    logging.debug(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    logging.debug(f"Loaded {len(texts)} texts and {len(labels)} labels")
    return texts, labels

def create_data_loader(texts, labels, tokenizer, max_len, batch_size):
    logging.debug(f"Creating data loader")
    ds = EmotionDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=2)

def train_model(train_loader, val_loader, model, device, optimizer, scheduler, num_epochs):
    model = model.to(device)
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for d in train_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        logging.debug(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for d in val_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                labels = d["label"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        logging.debug(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

    return train_losses, val_losses

def plot_losses(train_losses, val_losses, filename='losses.png'):
    plt.figure(figsize=(10, 8))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Loss per Epoch')
    plt.savefig(filename)
    plt.close()

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    cm = confusion_matrix(true_labels, predictions)
    return accuracy, precision, recall, f1, cm

def plot_confusion_matrix(cm, class_names, filename='confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()

def generate_wordcloud(text, filename='wordcloud.png'):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

def main():
    try:
        # 설정
        PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
        MAX_LEN = 128
        BATCH_SIZE = 16
        EPOCHS = 5
        LEARNING_RATE = 2e-5

        TRAIN_DATA_FILE_PATH = "/home/subin/AI+x/data/training.csv"
        VAL_DATA_FILE_PATH = "/home/subin/AI+x/data/validation.csv"
        TEST_DATA_FILE_PATH = "/home/subin/AI+x/data/test.csv"
        MODEL_SAVE_PATH = "/home/subin/AI+x/model/bert_emotion_model.bin"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_names = ["sadness","joy","love","anger","fear","surprise"]

        logging.debug("Loading data")
        train_texts, train_labels = load_data(TRAIN_DATA_FILE_PATH)
        val_texts, val_labels = load_data(VAL_DATA_FILE_PATH)
        test_texts, test_labels = load_data(TEST_DATA_FILE_PATH)

        logging.debug("Loading tokenizer")
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        train_loader = create_data_loader(train_texts, train_labels, tokenizer, MAX_LEN, BATCH_SIZE)
        val_loader = create_data_loader(val_texts, val_labels, tokenizer, MAX_LEN, BATCH_SIZE)
        test_loader = create_data_loader(test_texts, test_labels, tokenizer, MAX_LEN, BATCH_SIZE)

        logging.debug("Loading model")
        model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=6)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        logging.debug("Starting training")
        train_losses, val_losses = train_model(train_loader, val_loader, model, device, optimizer, scheduler, EPOCHS)

        plot_losses(train_losses, val_losses)

        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        logging.debug(f"Model saved to {MODEL_SAVE_PATH}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return

    try:
        logging.debug("Loading saved model for evaluation")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        accuracy, precision, recall, f1, cm = evaluate_model(model, test_loader, device)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        plot_confusion_matrix(cm, class_names)

        combined_text = ' '.join(test_texts)
        generate_wordcloud(combined_text)
        
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")

if __name__ == "__main__":
    main()

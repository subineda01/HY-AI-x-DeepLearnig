# 딥러닝 기반 텍스트의 감정 분석
Members : 

이가빈, 화학과, gabin0713@hanyang.ac.kr

장수빈, 수학과, subineda01@hanyang.ac.kr

박승현, 경영학부, boyojeck@hanyang.ac.kr

이상백, 기계공학부, leesangbaek98@naver.com



🔍 목차
1. Proposal
2. DataSets
3. Methodology & Evaluation
4. Conclusion-discussion
5. Related Works

-------------------------
# I.Proposal
- Why are you doing this?

 &ensp;딥러닝을 이용한 감정 인식 기술은 우리가 미처 알지 못했던 인간의 감정 패턴과 심리 상태를 더 깊이 이해할 수 있게 해줍니다. 이러한 감정 분류 시스템을 통해 개인적인 측면으로는 사람들의 온라인 활동과 소셜 미디어 게시물에서 감정을 분석하여 우울증, 불안, 스트레스 등으 정신 건강 문제를 조기에 발견 가능하게 해준다. 또한, 인공지능 비서나 교육용 로봇 등을 통해 학업적으로나 사무적인 효과를 극대화시키고, 인간-컴퓨터 상호작용(HCI)을 크게 향상시킬 수 있다. 사회적인 측면으로는 기업들이 고객 서비스에서 감정 인식 기술을 통해 고객의 감정 상태를 실시간으로 파악하여, 피드백 수용이 용이하다. 소셜 미디어에서 발생하는 혐오 발언이나 사이버 불링을 감지하여, 이를 사전에 방지함으로써 사회적 문제를 해결하는 수단으로 사용될 수 있다. 결론적으로, 딥러닝을 이용한 감정 인식 연구는 다양한 사회적, 학문적, 경제적 이점을 제공하며, 인류의 삶을 더 나은 방향으로 이끌어갈 수 있는 잠재력을 가지고 있다. 이러한 연구를 지속하고 발전시키는 것은 우리의 삶의 질을 향상시키고, 더 나은 사회를 만드는 데 중요한 역할을 할 것이라 판단하여 선정하게 되었다.

- What do you want to see at the end?

 &ensp;딥러닝 기반 감정 인식 기술을 통해 정신 건강 관리, 고객 서비스, 인간-컴퓨터 상호작용, 사회적 문제 해결 등 다양한 분야에서 실질적인 변화를 이루고자 한다. 이 기술은 사람들의 온라인 활동과 소셜 미디어 게시물에서 감정을 분석하여 우울증, 불안, 스트레스 등의 문제를 조기에 발견하고 예방하며, 고객의 감정을 실시간으로 파악하여 더 나은 서비스와 사용자 경험을 제공할 수 있다. 또한, 혐오 발언이나 사이버 불링을 감지하여 안전한 인터넷 환경을 조성하고, 대규모 데이터를 분석하여 사회적 트렌드와 감정 변화를 파악함으로써 효과적인 정책 수립을 지원한다. 궁극적으로, 이 연구를 통해 인간 감정의 복잡성을 이해하고 다양한 학문 분야에서 새로운 이론과 실천 방안을 개발하며, 예술과 문화 연구 등에서도 혁신적인 변화를 이끌어낼 수 있기를 기대한다. 이를 위해, 과거 데이터를 기반으로 미래의 감정을 예측하는 모델을 개발하여, 사람들이 더 나은 의사결정을 할 수 있도록 돕고자 하는 모델을 기반으로 만들고자 한다.

-------------------------
# II.DataSets
커뮤니티 기반의 독립 연구소인 DAIR.AI에서 제공하는 'Emotion Dataset'을 이용한다.

해당 데이터는 twitter API를 통해 수집된 영어문장을 여섯가지 기본감정들(anger, fear, joy, love, sadness, surprise)로 분류되었다.

선행 연구인 ‘CARER: Contextualized Affect Representations for Emotion Recognition’의 접근을 기반으로 데이터가 가공처리 된다.

Prior research Link : <https://aclanthology.org/D18-1404.pdf>  

Data link : <https://github.com/dair-ai/emotion_dataset>

## DataSets info

'Emotion Dataset' 중 학습을 위해 제공한 split data를 사용한다.

![데이터파일](https://github.com/subineda01/HY-AI-x-DeepLearnig/blob/main/image/dataset.png)

train.csv(16,000), validation.csv(2,000), test.csv(2,000)
-> 총 20,000개의 데이터 (1968KB)

<train.csv의 데이터 그래프>

<img src="https://github.com/subineda01/HY-AI-x-DeepLearnig/blob/main/image/distribution%20of%20label.png?raw=true" width="500" height="500"/>

## Data example
```sh
"text" : "im feeling quite sad and sorry for myself but ill snap out of it soon",
"label": 0
```

## Features
- text : 한 개의 문장으로 구성된 string 형태의 feature
- label : 감정을 분류한 라벨로 int 형태의 feature, 6가지 상태를 표현

| Emotion | label |
| ------- | ------- |
| sadness | 0 |
| joy | 1 |
| love | 2 |
| anger | 3 |
| fear | 4 |
| surprise | 5 |



-----------------------
# III.Methodology & Evaluation
## 1. Lstm classification
    
문장과 같은 시계열 데이터를 처리하기 위해서는 주로 RNN(순환신경망, Recurrent Neural Network)을 사용한다.
  
문장 속 이전 단어의 정보를 기억하는 것을 시작으로, 다음의 새로운 단어와의 정보를 합쳐서 처리하면서 AI는 단어의 순서와 문맥을 이해할 수 있게 된다.

이전의 노드에서 나온 한개의 정보와 새로운 단어의 정보만을 처리하기 때문에, 긴 문장에 대하여 처리할 때 앞의 정보를 잘 기억하지 못할 수 있는 문제가 발생한다.

![image](https://github.com/subineda01/HY-AI-x-DeepLearnig/blob/main/image/vaniila_rnn_and_different_lstm_ver2.png?raw=true)
    
전통적인 RNN의 단점을 보완한 RNN의 일종을 LSTM(장단기 메모리, Long Short-Term Memory)라고 한다. 
  
해당 모델은 메모리 셀에 입력 게이트, 망각 게이트, 출력 게이트를 추가하여 불필요한 기억을 지우고, 기억할 것을 유지시키는 작업을 수행한다.

망각 게이트에 의해 일부 기억을 잃고, 입력게이트에 의해 유지시킬 기억을 저장한 셀 상태 $`C_t`$가 추가되어 다음 메모리 셀로 전파된다. 

해당 프로젝트는 LSTM를 사용하여 문장을 처리하고, Clssification 모델을 붙이는 것으로 문장의 감정을 분류하는 모델을 만들게 되었다.

모델이 문장을 처리하기 위해서는, 문장이 모델이 이해할 수 있는 토큰의 형식으로 전처리 되어야 한다.(Tokenization)

프로젝트에서는 ```torchtext.data.utils``` 라이브러리를 사용하여 토큰화를 진행하였다. 

```
Text: Hello, world!
Tokens: ['hello', ',', 'world', '!']
Token indices: [3, 4, 5, 6]
```
토큰화는 다음과 같이 텍스트 데이터에서 고유한 토큰을 수집하고, 이를 인덱스로 매핑하여 사전을 형성한다.

이후 임베딩(Word Embedding)을 통해 각 토큰을 벡터 형태로 변환하여 PyTorch에서 사용할 수 있는 형태로 변환한다.

임베딩 된 토큰을 이용하여 문장을 재구성하고(Token indices) 문장 학습을 진행시킨다. 

Activation Function은 가장 보편적인 ```Adam```을 사용하였으며, loss function로는 ```nn.CrossEntropyLoss()```을 사용하였다.

```nn.CrossEntropyLoss()```에 포함된 ```softmax``` 함수를 이용하여 6가지의 감정으로 분류하였다.



학습머신 : 12th Gen Intel(R) Core(TM) i7-12700H, ddr5 16GB

### total code
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import torch.optim as optim

# 데이터 로드 및 전처리
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('validation.csv')

# 훈련 데이터 전처리
train_sentences = train_data['text'].values
train_labels = train_data['label'].values

# 검증 데이터 전처리
val_sentences = val_data['text'].values
val_labels = val_data['label'].values

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_sentences), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

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

def collate_batch(batch):
    text_list, label_list = [], []
    for _text, _label in batch:
        text_list.append(torch.tensor(_text, dtype=torch.long))
        label_list.append(torch.tensor(_label, dtype=torch.long))
    return pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"]), torch.stack(label_list)

train_dataset = TextDataset(train_sentences, train_labels, vocab, tokenizer)
val_dataset = TextDataset(val_sentences, val_labels, vocab, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size='', shuffle=True, collate_fn=collate_batch)  # 배치사이즈 조정
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_units, num_classes, dropout_rate):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.lstm = nn.LSTM(embed_dim, lstm_units, num_layers = '', batch_first=True) # LSTM 레이어 수 조정
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_units, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        x = self.fc(x)
        return x

# 하이퍼파라미터 설정
embed_dim = ''
lstm_units = ''
dropout_rate = ''
learning_rate = ''
num_epochs = ''

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(vocab_size=len(vocab), embed_dim=embed_dim, lstm_units=lstm_units, num_classes=6, dropout_rate=dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 손실 기록을 위한 리스트 초기화
train_losses = []
val_losses = []

# 모델 학습
model.train()
for epoch in range(num_epochs):
    train_loss = 0.0
    for texts, labels in train_dataloader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 에포크별 평균 훈련 손실 계산
    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)

    # 검증 데이터 손실 계산
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for texts, labels in val_dataloader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # 에포크별 평균 검증 손실 계산
    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

# 손실 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='blue')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# 모델 평가 함수
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

'''
# 모델 저장
torch.save(model.state_dict(), 'model_weights.pth')
torch.save(model, 'model.pth')
'''

# 테스트 데이터 로드 및 전처리
test_data = pd.read_csv('test.csv')
test_sentences = test_data['text'].values
test_labels = test_data['label'].values
test_labels = label_encoder.transform(test_labels)

test_dataset = TextDataset(test_sentences, test_labels, vocab, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

# 테스트 데이터로 모델 평가
test_accuracy = evaluate_model(model, test_dataloader)
print(f'Test Accuracy: {test_accuracy}')

```

## Result

Default parameter는 다음과 같이 설정하였다. 

```
lstm_layer = 1
Batch_size = 32
embed_dim = 128
lstm_units = 128
dropout_rate = 0.3
learning_rate = 1e-3
num_epochs = 10
```
![default](https://github.com/subineda01/HY-AI-x-DeepLearnig/blob/main/image/%EA%B8%B0%EB%B3%B8(10%EC%97%90%ED%8F%AD,1%EB%A0%88%EC%9D%B4%EC%96%B4)0.594.png?raw=true)
```Accuracy : 0.594```

Loss그래프에서 epoch가 지날 때마다 Train Loss가 감소하고, Validation Loss 또한 유의미하게 감소하는 것을 확인할 수 있었다.

하지만 감소의 정도가 크지 않고, 정확도 또한 59.4% 수준으로 높지 않았기 때문에 다음과 같은 hyper parameter tuning을 진행하였다.

### (1) learning_rate = 1e-2

![lr](https://github.com/subineda01/HY-AI-x-DeepLearnig/blob/main/image/10epoch_1lstm_lr001_0.454.png?raw=true)
```Accuracy : 0.454```

### (2) Batch_size = 64

![Batch](https://github.com/subineda01/HY-AI-x-DeepLearnig/blob/main/image/10epoch_1lstm_64batch_accu62.35.png?raw=true)
```Accuracy : 0.6235```

### (3) lstm_layer = 2

![layer](https://github.com/subineda01/HY-AI-x-DeepLearnig/blob/main/image/2%EB%A0%88%EC%9D%B4%EC%96%B40.872.png?raw=true)
```Accuracy : 0.872```

### (4) num_epochs = 20

![epoch](https://github.com/subineda01/HY-AI-x-DeepLearnig/blob/main/image/20epoch%EC%9C%BC%EB%A1%9C%EB%8A%98%EB%A6%BC0.8725.png?raw=true)
```Accuracy : 0.8725```

위 4가지 parameter tuning 중 가장 효과적이었던 (3)```lstm_layer```와 (4)```num_epochs```를 조정하는 것으로 모델의 성능을 향상시켰다. 

![last](https://github.com/subineda01/HY-AI-x-DeepLearnig/blob/main/image/15%EC%97%90%ED%8F%AD,2%EB%A0%88%EC%9D%B4%EC%96%B40.9005.png?raw=true)
```Accuracy : 0.9005```

이외에도 정확도를 높이기 위해 여러가지 hyper parameter tuning을 시도하였으나, 더 이상 올라가지 않았기 때문에 높은 성능을 보일 수 있는 다른 모델을 탐색하였다.

그 결과 Bert를 사용한 새로운 모델을 구성하였다. 

-----------------------

## 2. BertForSequenceClassification
 &ensp;BERT(Bidirectional Encoder Representations from Transformers)는 Goolge에서 개발한 자연어 처리 모델로, 2018년에 발표되었다. 텍스트의 문맥을 양방향으로 이해하는 데 뛰어나 더 뛰어난 성능을 지니고 있다. 
BERT는 대규모 텍스트 코퍼스에서 학습되어 수억개의 단어를 학습하고 있다. 사전 학습된 이해도를 이용해 전이 학습을 진행하여 목적에 적합한 뛰어난 모델을 구성할 수 있다. 
BertForSequenceClassification 모델은 Hugging Face의 Transformer 라이브러리에서 제공하는 텍스트 분류 작업을 위한 BERT 기반 모델이다. 이 모델은 BERT의 기본 아키텍처 위에 분류를 위한 추가 레이어를 포함하고 있다.

전체구조

![image](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/18a71006-4452-4258-93b4-3a8a0c0ff3ab)

 &ensp;모델은 크게 두가지 구조인 BertModel과 Classifier로 이루어져 있다. BertModel은 Transformer layer가 여러겹으로 쌓여있는 본체이다. 이는 BertEmbedding 부분과 BertEncoder부분으로 나누어져 있다. 

### BertEmbedding

![image](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/589d2e7d-aeda-44d5-8a0c-9f73000fd8b6)

BertEmbedding은 문장을 입력으로 받아 토큰화 시키고 token, segment, position을 임베딩하여 값으로 만들고 더해서 반환해주는 역할을 한다.

입력 임베딩(Input Embeddings):
   * Token Embedding : 각 토큰에 대한 고유한 임베딩 벡터
   * Segment Embedding : 문장이 두개일 때 첫 문장과 두 번째 문당을 구분하기 위한 임베딩 벡터
   * Position Embedding : 각 토큰의 위치를 나타내는 임베딩 벡터. 문장 내에서 각 토큰의 순서를 모델이 알 수 있게 한다.

### BertEncoder

BERT는 트랜스포머(Transformer) 모델의 인코더 부분만 사용한다. 이는 여러 층의 인코더 블록으로 구성된다.

#### - 트랜스포머 인코더 개요

BERT의 인코더는 트랜스포머 인코더 블록의 스택으로 구성된다. 트랜스포머 인코더는 여러 층의 인코더 블록으로 구성되며, 각 블록은 다음 두 가지 주요 구성 요소로 이루어져 있다.

1. Multi-Head Self-Attention Mechanism:
   - Query, Key, Value 행렬을 계산하고, Attention 점수를 통해 토큰 쌍의 관계를 학습한다.
     
![QKV Calculation](https://latex.codecogs.com/svg.latex?Q%20%3D%20XW_Q%2C%20%5Cquad%20K%20%3D%20XW_K%2C%20%5Cquad%20V%20%3D%20XW_V)

2. Position-wise Feed-Forward Neural Network:
   - 두 개의 선형 변환과 비선형 활성화 함수로 하여 완전 연결 신경망을 구성한다.
     
![Feed-Forward Neural Network](https://latex.codecogs.com/svg.latex?%5Ctext%7BFFN%7D(x)%20%3D%20%5Ctext%7Bmax%7D(0%2C%20xW_1%20%2B%20b_1)W_2%20%2B%20b_2)
  
 &ensp;이와 같이 입력 텍스트를 토크나이즈하고 임베딩을 통해 모델에 입력하는 과정은 LSTM 모델에서의 임베딩 과정과 유사하다. BERT 모델 또한 이를 통해 입력 텍스트의 복잡한 관계를 학습하고, 텍스트 분류 작업을 수행한다.

이 인코더 레이어들은 입력 임베딩을 점진적으로 더 복잡한 표현으로 변환하며, 최종적으로 입력 시퀸스의 각 토큰에 대한 풍부한 문맥 정보를 포함한 고차원 벡터 표현을 출력한다.   

3. 잔차 연결과 층 정규화 (Residual Connections and Layer Normalization)

각 트랜스포머 인코더 블록은 두 개의 서브레이어(Sublayer)로 구성되어 있다. 
각 서브레이어 후에 잔차 연결과 층 정규화를 적용하여 학습을 안정화하고 성능을 향상시킨다. 


+)부가 설명

- **어텐션 점수 계산**: Query와 Key의 내적을 통해 각 토큰 쌍의 점수를 계산하고, 이를 스케일링 후 소프트맥스 함수를 적용하여 가중치를 얻는다.
  
   ![Attention Score](https://latex.codecogs.com/svg.latex?%5Ctext%7BAttention%7D(Q%2C%20K%2C%20V)%20%3D%20%5Ctext%7Bsoftmax%7D%5Cleft(%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%5Cright)V)

- **Multi-Head Attention**: 여러 개의 어텐션 헤드를 사용하여 각 헤드의 출력을 결합시킨다.
  
   ![Multi-Head Attention](https://latex.codecogs.com/svg.latex?%5Ctext%7BMultiHead%7D(Q%2C%20K%2C%20V)%20%3D%20%5Ctext%7BConcat%7D(%5Ctext%7Bhead%7D_1%2C%20%5Cldots%2C%20%5Ctext%7Bhead%7D_h)W_O)

위의 원리를 활용한```BertForSequenceClassification```라이브러리를 사용하여 분류 모델을 생성하였다. 

학습머신 : Intel(R) Xeon(R) Platinum 8462Y+ 메모리 1024GB
### Total code
```python
#1. 환경 설정 및 라이브러리 로드
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

#2. 데이터셋 클래스 정의
#EmotionDataset 클래스는 데이터셋을 관리하고, BERT 모델이 요구하는 형식으로 데이터를 변환

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

#3. 데이터 로드 및 데이터 로더 생성
#CSV 파일에서 데이터를 로드하고, 데이터 로더를 생성하는 함수들

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

#4. 모델 학습 함수
#이 함수는 모델을 학습시키고, 각 에포크(epoch)마다 손실(loss)을 기록

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

#5. 손실 그래프 그리기
#학습 및 검증 손실을 그래프로 나타낸다.

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

#6. 모델 평가 함수
#모델을 평가하고, 정확도, 정밀도, 재현율, F1 점수 및 혼동 행렬을 계산

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

#7. 혼동 행렬 그리기
#혼동 행렬을 시각

def plot_confusion_matrix(cm, class_names, filename='confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()

#8. 워드 클라우드 생성
#텍스트 데이터를 기반으로 워드 클라우드를 생성

def generate_wordcloud(text, filename='wordcloud.png'):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

#9. 메인 함수
#전체 파이프라인을 실행하는 메인 함수

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
```
- 데이터 로드 및 전처리: CSV 파일에서 데이터를 로드하고, BERT의 입력 형식에 맞게 토크나이즈한다.
- 모델 학습: 학습 데이터를 사용하여 BERT 모델을 학습시키고, 각 에포크마다 손실을 기록한다.
- 모델 평가: 테스트 데이터를 사용하여 모델을 평가하고, 정확도, 정밀도, 재현율, F1 점수 및 혼동 행렬을 계산한다.
- 시각화: 손실 그래프, 혼동 행렬, 워드 클라우드를 시각화한다.

### Word Cloud
![wordcloud](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/7c09d6b2-6d35-499e-829f-e3a0c45c03dc)

word Cloud 이미지는 텍스트 데이터에서 단어들의 빈도나 중요도를 시각적으로 표현한다. 'feel', 'feeling', really' 등의 단어들이 크게 표시되어 텍스트에서 빈도가 높거나 중요도가 큰 단어임을 표시한다. 

### Loss Graph

![losses](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/877302d4-9837-4efa-a285-7cf732a61549)


### Confusion Matrix
![confusion_matrix](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/c118b4a3-fb0e-40ca-be83-a38c75df86da)

### Result

![image](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/cd90b260-6261-4686-971f-1b6c57635c0b)

 &ensp;다양한 하이퍼파라미터를 가지고 실험하였다. 학습률을 2e-3 2e-4 2e-r-5를 사용하여 실험 해본 결과 2e-5일 때 가장 높은 성능을 기록하였다. 
 
 &ensp;에포크 수는  5 10 30을 가지고 실험 해본 결과 에포크 수가 커지면 커질수록 validation loss가 커짐을 확인 할 수 있었다. 
 
 &ensp;validation set에서는 에포크 1 이후로 더이상 학습을 잘 하지 못하는 것으로 보인다.(학습을 시키지 않은 상태에서 모델에 validation.csv를 통과시킨 결과 0.135의 정확도가 나왔다. LLM이기 떄문에 1 epoch만으로 충분한 학습이 되었을 것으로 예측되었다.) 
 
 &ensp;따라서 에포크의 수를 늘리는 것은 과적합을 만든다고 판단하여 에포크 수를 작게 설정하였다. 마지막으로 배치 수를 16 32 64로 변경해 보았지만 큰 차이는 없었고, 결과적으로 정확도와 재현율이 모두 93%대를 기록하였다.

최종 하이퍼 파리미터
```
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
```
-----------------------

# IV. Conclusion: Discussion

 &ensp;위 프로젝트에서 감정을 텍스트로부터 인식하는 두가지 모델을 제시하였다. 학습용으로 공개된 데이터 세트이기 때문에 이미 많은 모델이 제안되고 있지만, 이에 대한 참조 없이 적절한 알고리즘을 골라 모델을 완성하였다. 
LSTM 약 90%, BERT 약 93%로 꽤 준수한 정확도를 기록하였지만, 성능을 향상시킬 수 있는 다른 방법에 대하여 논의하였다. 

 &ensp;20epoch, 5epoch만에 과적합이 발생하였기 때문에 기존의 데이터보다 많은 양의 데이터를 이용해 학습시킬 필요성이 있어보인다. 또한 가중치 규제(Regularization)를 적용하거나, 조금 더 높은 Dropout 비율을 설정하는 방법도 존재한다. 

 &ensp;처음 데이터셋을 보면 인덱스 0(sadness), 1(joy)의 데이터가 절반이상을 차지하고 있다. 이런 편향된 데이터는 모델의 정확도를 높이는데 방해되는 요인으로 작용하였을 것이다. 
 
 &ensp;해당 데이터셋은 twitter API를 통해 추출하였기 때문에 해당 커뮤니티의 문화를 담고 있다. 크롤링 기술을 이용하여 다양한 SNS 문장을 확보할 수 있다면 전반적으로 개선되고 일반화된 모델을 생성하는데 도움이 되었을 것이다. 

<위 모델을 통한 새로운 기술>

- 정신 건강 관리 시스템의 혁신

 &ensp;조기 경고 시스템: 소셜 미디어와 온라인 활동을 실시간으로 모니터링하여 우울증, 불안 등의 정신 건강 문제를 조기에 경고할 수 있는 시스템을 개발할 수 있다.
개인 맞춤형 치료 계획: 감정 인식 데이터를 기반으로 개인 맞춤형 치료 계획을 세우고, 정기적으로 환자의 감정 상태를 모니터링하여 치료의 효과를 극대화할 수 있다.

 - 고객 서비스 및 사용자 경험 향상

 &ensp;실시간 감정 분석: 고객의 감정을 실시간으로 분석하여 즉각적인 대응을 통해 고객 만족도를 높일 수 있다.
개인화된 서비스 제공: 고객의 감정 상태에 기반한 맞춤형 서비스 제공으로 고객 충성도를 높일 수 있다.

 - 인간-컴퓨터 상호작용 개선

 &ensp;감정 반응 AI 비서: 감정을 이해하고 반응하는 AI 비서나 로봇을 개발하여 사용자와의 상호작용을 더욱 자연스럽고 인간적으로 만들 수 있다.
교육 및 엔터테인먼트 분야: 감정을 이해하는 교육용 로봇이나 엔터테인먼트 시스템을 통해 학습 효과를 극대화하고 사용자 경험을 향상시킬 수 있다.

 - 사회적 문제 해결

 &ensp;사이버 불링 및 혐오 발언 감지: 소셜 미디어에서 사이버 불링이나 혐오 발언을 실시간으로 감지하여 사전 예방 조치를 취할 수 있다.
사회적 트렌드 분석: 대규모 데이터를 분석하여 사회적 트렌드와 감정 변화를 파악하고, 이를 기반으로 효과적인 정책 수립을 지원할 수 있다.

<상용화 및 성공 가능성>

딥러닝 기반 감정 인식 기술은 다음과 같은 이유로 상용화와 성공 가능성이 높다.

 - 다양한 적용 분야: 정신 건강, 고객 서비스, HCI, 사회적 문제 해결 등 다양한 분야에서 활용 가능성이 높아 시장 수요가 크다.

 - 기술의 정밀도 및 신뢰성: CARER 알고리즘의 높은 정확도와 신뢰성으로 인해 실질적인 문제 해결에 기여할 수 있다.

 - 기술의 유연성: 이 기술은 여러 언어와 문화적 맥락에서도 적용 가능하여 글로벌 시장에서도 활용될 수 있다.

 - 지속적인 발전 가능성: 딥러닝과 그래프 기반 방법의 발전으로 기술이 지속적으로 개선될 수 있어 장기적인 성장 가능성이 높다.

 &ensp;딥러닝 기반 감정 인식 기술은 여러 산업 분야에서 혁신적인 변화를 가져올 수 있는 잠재력을 가지고 있다. 이 기술은 정신 건강 관리, 고객 서비스, 인간-컴퓨터 상호작용, 사회적 문제 해결 등 다양한 분야에서 실질적인 변화를 이끌어낼 수 있다. 또한, 상용화 가능성이 높고, 다양한 새로운 기술로 발전할 수 있는 가능성이 크다. 앞으로도 지속적인 연구와 발전을 통해 인류의 삶의 질을 향상시키고, 더 나은 사회를 만드는 데 중요한 역할을 할 것이다.

-----------------------

# V. Related Works & References

툴(Tool): python

라이브러리(Library):
```python
numpy
pandas
matplotlib.pyplot
sklearn
torchtext
torch
transformers
seaborn
wordcloud
collections
```

### 블로그(Blog)
[torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

[pytorch로 RNN, LSTM 구현하기](https://justkode.kr/deep-learning/pytorch-rnn/)

[08-02 장단기 메모리(Long Short-Term Memory, LSTM)](https://wikidocs.net/22888)

[BERT(huggingface)](https://huggingface.co/transformers/v3.0.2/model_doc/bert.html)

### 논문
[Contextualized Affect Representations for Emotion Recognition](https://aclanthology.org/D18-1404.pdf)

[CARER: Contextualized Affect Representations for Emotion Recognition](https://aclanthology.org/D18-1404.pdf)

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)







# 딥러닝 기반 텍스트의 감정 분석
Members : 

이가빈, 화학과, gabin0713@hanyang.ac.kr

장수빈, 수학과,

박승현, 경영학부, boyojeck@hanyang.ac.kr

이상백, 기계공학부, leesangbaek98@naver.com



🔍 목차
1. Proposal
2. DataSets
3. Methodology
4. Evaluation & Analysis
5. Conclusion-discussion
6. Related Works

-------------------------
# I.Proposal
- Why are you doing this?

딥러닝을 이용한 감정 인식 기술은 우리가 미처 알지 못했던 인간의 감정 패턴과 심리 상태를 더 깊이 이해할 수 있게 해줍니다. 이러한 감정 분류 시스템을 통해 개인적인 측면으로는 사람들의 온라인 활동과 소셜 미디어 게시물에서 감정을 분석하여 우울증, 불안, 스트레스 등으 정신 건강 문제를 조기에 발견 가능하게 해준다. 또한, 인공지능 비서나 교육용 로봇 등을 통해 학업적으로나 사무적인 효과를 극대화시키고, 인간-컴퓨터 상호작용(HCI)을 크게 향상시킬 수 있다. 사회적인 측면으로는 기업들이 고객 서비스에서 감정 인식 기술을 통해 고객의 감정 상태를 실시간으로 파악하여, 피드백 수용이 용이하다. 소셜 미디어에서 발생하는 혐오 발언이나 사이버 불링을 감지하여, 이를 사전에 방지함으로써 사회적 문제를 해결하는 수단으로 사용될 수 있다. 결론적으로, 딥러닝을 이용한 감정 인식 연구는 다양한 사회적, 학문적, 경제적 이점을 제공하며, 인류의 삶을 더 나은 방향으로 이끌어갈 수 있는 잠재력을 가지고 있다. 이러한 연구를 지속하고 발전시키는 것은 우리의 삶의 질을 향상시키고, 더 나은 사회를 만드는 데 중요한 역할을 할 것이라 판단하여 선정하게 되었다.

- What do you want to see at the end?

딥러닝 기반 감정 인식 기술을 통해 정신 건강 관리, 고객 서비스, 인간-컴퓨터 상호작용, 사회적 문제 해결 등 다양한 분야에서 실질적인 변화를 이루고자 한다. 이 기술은 사람들의 온라인 활동과 소셜 미디어 게시물에서 감정을 분석하여 우울증, 불안, 스트레스 등의 문제를 조기에 발견하고 예방하며, 고객의 감정을 실시간으로 파악하여 더 나은 서비스와 사용자 경험을 제공할 수 있다. 또한, 혐오 발언이나 사이버 불링을 감지하여 안전한 인터넷 환경을 조성하고, 대규모 데이터를 분석하여 사회적 트렌드와 감정 변화를 파악함으로써 효과적인 정책 수립을 지원한다. 궁극적으로, 이 연구를 통해 인간 감정의 복잡성을 이해하고 다양한 학문 분야에서 새로운 이론과 실천 방안을 개발하며, 예술과 문화 연구 등에서도 혁신적인 변화를 이끌어낼 수 있기를 기대한다. 이를 위해, 과거 데이터를 기반으로 미래의 감정을 예측하는 모델을 개발하여, 사람들이 더 나은 의사결정을 할 수 있도록 돕고자 하는 모델을 기반으로 만들고자 한다.

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
# III.Methodology
## 1. Lstm classification
    
문장과 같은 시계열 데이터를 처리하기 위해서는 주로 RNN(순환신경망, Recurrent Neural Network)을 사용합니다. 
  
문장 속 이전 단어의 정보를 기억하는 것을 시작으로, 다음의 새로운 단어와의 정보를 합쳐서 처리하면서 AI는 단어의 순서와 문맥을 이해할 수 있게 됩니다.

이전의 노드에서 나온 한개의 정보와 새로운 단어의 정보만을 처리하기 때문에, 긴 문장에 대하여 처리할 때 앞의 정보를 잘 기억하지 못할 수 있는 문제가 발생합니다.
    
전통적인 RNN의 단점을 보완한 RNN의 일종을 LSTM(장단기 메모리, Long Short-Term Memory)라고 합니다. 
  
해당 모델은 메모리 셀에 입력 게이트, 망각 게이트, 출력 게이트를 추가하여 불필요한 기억을 지우고, 기억할 것을 유지시키는 작업을 수행합니다.

망각 게이트에 의해 일부 기억을 잃고, 입력게이트에 의해 유지시킬 기억을 저장한 셀 상태 $`C_t`$가 추가되어 다음 메모리 셀로 전파됩니다. 



해당 모델은 lstm에 classification 을 붙인 모델임.

lstm설명 -> 토크나이즈 부가 설명 -> activation 및 loss function 설명

학습머신 : intel i7 12gen, ddr5 16GB

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
## 2. BertForSequenceClassification
위 모델은 Hugging Face의 Transformer 라이브러리에서 제공하는 모델로 텍스트 분류 작업을 위해 설계된 BERT 기반 모델이이다. 이 모델은 BERT의 기본 아키텍쳐 위에 분류를 위한 추가 레이어를 포함하고 있다.

전체구조

![image](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/18a71006-4452-4258-93b4-3a8a0c0ff3ab)

모델은 크게 두가지 구조인 BertModel과 Classifier로 이루어져 있다. BertModel은 Transformer layer가 여러겹으로 쌓여있는 본체입니다. 이는 BertEmbedding 부분과 BertEncoder부분으로 나누어져 있다. 

### BertEmbedding

![image](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/589d2e7d-aeda-44d5-8a0c-9f73000fd8b6)

BertEmbedding은 문장을 입력으로 받아 token, segment, position을 임베딩하여 값으로 만들고 더해서 반환해주는 역할을 한다.
1. 토크나이징(Tokenization):
   * 입력 텍스트는 WordPiece 토크나이저를 통해 토큰으로 분해된다.
   * 토큰은 고유한 정수로 매핑된다.
2. 입력 임베딩(Input Embeddings):
   * Token Embedding : 각 토큰에 대한 고유한 임베딩 벡터
   * Segment Embedding : 문장이 두개일 때 첫 문장과 두 번째 문당을 구분하기 위한 임베딩 벡터
   * Position Embedding : 각 토큰의 위치를 나타내는 임베딩 벡터. 문장 내에서 각 토큰의 순서를 모델이 알 수 있게 한다.

### BertEncoder

BERT는 트랜스포머(Transformer) 모델의 인코더 부분만 사용한다.

#### 트랜스포머 인코더 개요

BERT의 인코더는 트랜스포머 인코더 블록의 스택으로 구성된다. 트랜스포머 인코더는 여러 층의 인코더 블록으로 구성되며, 각 블록은 다음 두 가지 주요 구성 요소로 이루어져 있다.

1. Multi-Head Self-Attention Mechanism:
   - 각 토큰이 다른 모든 토큰과의 관계(주의 메커니즘)를 학습할 수 있게 한다.
   - 다양한 주의(attention) 헤드를 사용하여 서로 다른 부분에 집중할 수 있다.

2. Position-wise Feed-Forward Neural Network:
   - Self-attention의 출력을 각 토큰에 대해 독립적으로 처리하는 완전 연결 네트워크
   - 비선형 활성화 함수를 사용하여 복잡한 표현을 학습

#### BERT 인코더 구성 요소

##### Multi-Head Self-Attention

Multi-Head Self-Attention 메커니즘은 각 토큰이 문장의 다른 모든 토큰과의 관계를 학습할 수 있게 한다.

- **Query, Key, Value 계산**: 입력 임베딩을 세 개의 행렬 \( W_Q \), \( W_K \), \( W_V \)에 곱하여 Query, Key, Value 행렬을 만든다.
  
   ![QKV Calculation](https://latex.codecogs.com/svg.latex?Q%20%3D%20XW_Q%2C%20%5Cquad%20K%20%3D%20XW_K%2C%20%5Cquad%20V%20%3D%20XW_V)

- **어텐션 점수 계산**: Query와 Key의 내적을 통해 각 토큰 쌍의 점수를 계산하고, 이를 스케일링 후 소프트맥스 함수를 적용하여 가중치를 얻는다.
  
   ![Attention Score](https://latex.codecogs.com/svg.latex?%5Ctext%7BAttention%7D(Q%2C%20K%2C%20V)%20%3D%20%5Ctext%7Bsoftmax%7D%5Cleft(%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%5Cright)V)

- **Multi-Head Attention**: 여러 개의 어텐션 헤드를 사용하여 각 헤드의 출력을 결합
  
   ![Multi-Head Attention](https://latex.codecogs.com/svg.latex?%5Ctext%7BMultiHead%7D(Q%2C%20K%2C%20V)%20%3D%20%5Ctext%7BConcat%7D(%5Ctext%7Bhead%7D_1%2C%20%5Cldots%2C%20%5Ctext%7Bhead%7D_h)W_O)

### Position-wise Feed-Forward Neural Network

각 토큰에 대해 독립적으로 작동하는 두 개의 선형 변환과 비선형 활성화 함수로 구성된 완전 연결 신경망

![Feed-Forward Neural Network](https://latex.codecogs.com/svg.latex?%5Ctext%7BFFN%7D(x)%20%3D%20%5Ctext%7Bmax%7D(0%2C%20xW_1%20%2B%20b_1)W_2%20%2B%20b_2)

### 잔차 연결과 층 정규화 (Residual Connections and Layer Normalization)

각 트랜스포머 인코더 블록은 두 개의 서브레이어(Sublayer)로 구성:

1. **Self-Attention Sublayer**: Multi-Head Self-Attention을 적용
2. **Feed-Forward Sublayer**: Position-wise Feed-Forward Neural Network를 적용

각 서브레이어 후에는 잔차 연결과 층 정규화를 적용하여 학습을 안정화하고 성능을 향상

# IV. Evaluation & Result
### Word Cloud
![wordcloud](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/7c09d6b2-6d35-499e-829f-e3a0c45c03dc)

### Loss Graph

![losses](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/f59b3c2c-3543-4fa6-b6e5-db7cfd8b9b79)

### Confusion Matrix
![confusion_matrix](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/c118b4a3-fb0e-40ca-be83-a38c75df86da)

### Result

![image](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/cd90b260-6261-4686-971f-1b6c57635c0b)
다양한 하이퍼파라미터를 가지고 실험을 해보았음. 학습률을 2e-3 2e-4 2e-r-5를 사용하여 실험 해본 결과 2e-5일 때의 성능이 제일 나았음. 에포크 수는  5 10 30을 가지고 실험 해본 결과 에포크 수가 커지면 커질수록 validation loss가 커짐을 확인 할 수 있었음. 따라서 에포크 수는 5로 설정하였음. 마지막으로 배치 수를 16 32 64로 변경해 보았지만 큰 차이는 없었음. 결과적으로 정확도와 재현율이 모두 93%대를 기록하였음

# V. Conclusion: Discussion

감정을 텍스트로부터 인식하는 데 있어 기존 방법들을 뛰어넘는 새로운 그래프 기반 알고리즘을 선보인다. 이 알고리즘은 감정이 표현되는 다양한 언어적 뉘앙스를 포착하고 모델링하기 위해 풍부한 구조적 설명자를 생성한다. 제안된 방법은 단어 임베딩을 통해 더욱 풍부해진 패턴 기반 표현을 사용하여 감정 인식 작업에서 뛰어난 성능을 보였준다.

위 모델을 통한 새로운 기술

정신 건강 관리 시스템의 혁신
조기 경고 시스템: 소셜 미디어와 온라인 활동을 실시간으로 모니터링하여 우울증, 불안 등의 정신 건강 문제를 조기에 경고할 수 있는 시스템을 개발할 수 있다.
개인 맞춤형 치료 계획: 감정 인식 데이터를 기반으로 개인 맞춤형 치료 계획을 세우고, 정기적으로 환자의 감정 상태를 모니터링하여 치료의 효과를 극대화할 수 있다.

고객 서비스 및 사용자 경험 향상
실시간 감정 분석: 고객의 감정을 실시간으로 분석하여 즉각적인 대응을 통해 고객 만족도를 높일 수 있다.
개인화된 서비스 제공: 고객의 감정 상태에 기반한 맞춤형 서비스 제공으로 고객 충성도를 높일 수 있다.

인간-컴퓨터 상호작용 개선
감정 반응 AI 비서: 감정을 이해하고 반응하는 AI 비서나 로봇을 개발하여 사용자와의 상호작용을 더욱 자연스럽고 인간적으로 만들 수 있다.
교육 및 엔터테인먼트 분야: 감정을 이해하는 교육용 로봇이나 엔터테인먼트 시스템을 통해 학습 효과를 극대화하고 사용자 경험을 향상시킬 수 있다.

사회적 문제 해결
사이버 불링 및 혐오 발언 감지: 소셜 미디어에서 사이버 불링이나 혐오 발언을 실시간으로 감지하여 사전 예방 조치를 취할 수 있다.
사회적 트렌드 분석: 대규모 데이터를 분석하여 사회적 트렌드와 감정 변화를 파악하고, 이를 기반으로 효과적인 정책 수립을 지원할 수 있다.

상용화 및 성공 가능성
딥러닝 기반 감정 인식 기술은 다음과 같은 이유로 상용화와 성공 가능성이 높다.

다양한 적용 분야: 정신 건강, 고객 서비스, HCI, 사회적 문제 해결 등 다양한 분야에서 활용 가능성이 높아 시장 수요가 크다.
기술의 정밀도 및 신뢰성: CARER 알고리즘의 높은 정확도와 신뢰성으로 인해 실질적인 문제 해결에 기여할 수 있다.
기술의 유연성: 이 기술은 여러 언어와 문화적 맥락에서도 적용 가능하여 글로벌 시장에서도 활용될 수 있다.
지속적인 발전 가능성: 딥러닝과 그래프 기반 방법의 발전으로 기술이 지속적으로 개선될 수 있어 장기적인 성장 가능성이 높다.

딥러닝 기반 감정 인식 기술은 여러 산업 분야에서 혁신적인 변화를 가져올 수 있는 잠재력을 가지고 있다. 이 기술은 정신 건강 관리, 고객 서비스, 인간-컴퓨터 상호작용, 사회적 문제 해결 등 다양한 분야에서 실질적인 변화를 이끌어낼 수 있다. 또한, 상용화 가능성이 높고, 다양한 새로운 기술로 발전할 수 있는 가능성이 크다. 앞으로도 지속적인 연구와 발전을 통해 인류의 삶의 질을 향상시키고, 더 나은 사회를 만드는 데 중요한 역할을 할 것이다.

# VI. Related Works & References

툴(Tool): 

라이브러리(Library): 

### 블로그(Blog)
### 논문
[Contextualized Affect Representations for Emotion Recognition](https://aclanthology.org/D18-1404.pdf)






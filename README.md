# 딥러닝 기반 텍스트의 감정 분석
Members : 

이가빈, 화학과, gabin0713@hanyang.ac.kr

장수빈, 수학과,

박승현, 경영학부, boyojeck@hanyang.ac.kr

이상백, 기계공학부,



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
# III.DataSets
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

# IV. Evaluation & Analysis

# V. Conclusion: Discussion







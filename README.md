# HY-AI-x-DeepLearnig
Members : 
이가빈, 화학과, gabin0713@hanyang.ac.kr
장수빈, 수학과,
박승현, 경영학부,
이상백, 기계공학부,

🔍 목차
1. Proposal
2. DataSets
3. Methodology
4. Evaluation & Analysis
5. Related Works

Proposal
- Motivation: Why are you doing this?
- What do you want to see at the end?




# III.Methodology
### 앙상블 기법
앙상블 기법은 여러 모델의 예측을 결합하여 최종 예측을 도출하는 방법으로, 개별 모델의 단점을 보완하고 성능을 향상시킬 수 있음
일반적으로 앙상블 기법은 과적합을 줄이고 모델의 일반화 성능을 향상시키는데 유리함

### 앙상블 기법 종류
1. 배깅 : 여러 모델을 병렬로 학습시키고, 각 모델의 예측을 평균 또는 투표 방식으로 결합하는 방식 ex) Random Forest
   
    ![image](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/86028d89-58c0-4fac-8e13-ea390d1a465a)

2. 부스팅 : 모델을 순차적으로 학습시키고, 이전 모델이 잘못 예측한 샘플에 가중치를 부여하여 다음 모델을 학습시키는 방식 ex) XGBoost, LightGBM
   
   ![image](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/08a8e8c2-2b9b-472b-8b56-7fa1f7e2490c)


3. 스태킹 : 여러 다른 유형의 모델을 학습시키고, 이들 모델의 예측을 기반으로 메타 모델을 학습시켜 최종 예측을 도출하는 방식
   
![image](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/c255d052-0b30-4a5a-bd94-6266931866b1)

Machine Reading Comprehension 에서 앙상블 접근 방식이 out of distribution의 정확도를 개선하는데 있어 효과적이라는 방식임
근거 논문 : https://ar5iv.labs.arxiv.org/html/2107.00368
따라서 저희는 다양한 대규모 분류 모델들을 학습시킨 뒤 앙상블 기법을 사용하여 모델의 정확도를 더 향상시키는 방식을 사용하려 합니다.
이중에서도 가중치를 동일하게 하는 equal weighting 방식을 사용할 겁니다.

### Equal Weighting
![image](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/60497872-bdb2-4a56-a237-19c2173c2e71)

1. 평균 : 각 모델의 예측 확률을 평균내어 최종 예측을 도출하는 방식. 모든 모델의 예측을 동일하게 고려함
2. 곱셈 : 각 모델의 예측 확률을 곱하여 최종 예측을 도출하는 방식. 모든 모델이 특정 위치에 대해 높은 확률을 보일 때 그 위치를 강조함
3. 최대값 : 각 모델의 예측 확률 중 가장 높은 값을 선택하여 최종 예측을 도출. 단 하나의 모델이라도 높은 확신을 가진 위치를 강조함
4. 최소값 : 각 모델의 예측 확률 중 가장 낮은 값을 선택하여 최종 예측을 도출. 가장 자신감이 낮은 모델의 예측을 고려하여 가장 보수적인 접근을 취함
이 중 평균을 취하는 것이 일반적인 경우에 안정적이며 모든 모델의 예측을 균형있게 반영하기에 평균을 취하는 방식을 택했습니다.

다음으로는 앙상블 기법에 활용할 모델들을 소개합니다.

### 1. BertForSequenceClassification
위 모델은 Hugging Face의 Transformer 라이브러리에서 제공하는 모델로 텍스트 분류 작업을 위해 설계된 BERT 기반 모델입니다. 이 모델은 BERT의 기본 아키텍쳐 위에 분류를 위한 추가 레이어를 포함하고 있습니다.'

전체구조

![image](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/18a71006-4452-4258-93b4-3a8a0c0ff3ab)

모델은 크게 두가지 구조인 BertModel과 Classifier로 이루어져 있습니다. BertModel은 Transformer layer가 여러겹으로 쌓여있는 본체입니다. 이는 BertEmbedding 부분과 BertEncoder부분으로 나누어져 있습니다. BertEmbedding은 문장을 입력으로 받아 token, segment, position을 임베딩하여 값으로 만들고 더해서 반환해주는 역할을 합니다.

BertEmbedding

![image](https://github.com/subineda01/HY-AI-x-DeepLearnig/assets/144909753/589d2e7d-aeda-44d5-8a0c-9f73000fd8b6)

BertEncoder

임베딩 되어 있는 값들을 토대로 Multi-Head Attention을 통해 입력 토큰 간의 관계를 학습합니다. 이후 어텐션의 출력을 반환하고 Residual Connection을 통해 인코딩을 완료합니다.

이후 클래스 수에 맞는 출력 벡터로 변환되어 소프트맥스 함수를 통해 출력 벡터를 확률 분포로 변환하여 최종 예측을 진행합니다.


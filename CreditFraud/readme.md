## Credit Card Fraud Detection

#### Dataset

-   [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/dalpozz/creditcardfraud)
-   2013년 9월 유럽 카드 소지자들의 거래 내역
-   2일간 발생한 284,807 건의 거래 내역, 492건의 사기 거래를 포함 (0.172% - 불균형 데이터)
-   입력 변수 : 거래 시각(Time) + PCA로 변환된 입력 변수 28개(V1~V28) + 금액(Amount)
-   반응 변수(Class) : 사기(=1), 정상(=0)

#### Model

-   일반적인 ANN 기법으로 접근하는 것이 더 나을 것으로 판단되나, CNN 기법으로 접근해 보려는 시도


#### 사용법?

-   `explorer.py` : `./data` 폴더에서 `creditcard.txt`파일을 읽어서 `train_data.pkl`, `test_data.pkl`, `smote_data.pkl`을 생성. (실제 모델에서는 `smote_data.pkl`만 사용함...)
-   `CNN_fraud_train.py` : 1차원 CNN을 사용하여 학습. 학습 시도시 `./runs/` 폴더 아래 모델 결과가 저장(checkpoints)
-   `CNN_fraud_test.py` : test에 사용할 모델이 저장된 폴더 (예시: `./runs/1491871762/checkpoints`)를 코드 내 30번째 줄에 입력 후 실행
`tf.flags.DEFINE_string("checkpoint_dir", "./runs/1491871762/checkpoints", "Checkpoint directory from training run")`

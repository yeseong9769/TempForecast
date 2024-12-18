# Temperature Prediction Project

LSTM 기반의 시계열 모델을 사용하여 기상 데이터로부터 향후 24시간의 온도를 예측하는 프로젝트입니다.

## 프로젝트 구조

```
├── data/
│   └── cleaned_weather.csv     # 전처리된 기상 데이터
├── images/                     # 학습 결과 및 예측 시각화 저장 디렉토리
├── models/                     # 학습된 모델 저장 디렉토리
├── model.py                    # LSTM 기반 날씨 예측 모델 구현
├── train.py                   # 모델 학습 실행 스크립트
├── predict.py                 # 예측 실행 스크립트
├── trainer.py                 # 모델 학습 관련 클래스
├── utils.py                   # 데이터 처리 및 유틸리티 함수
└── requirements.txt           # 프로젝트 의존성 패키지
```

## 주요 기능

### 모델 구조 (model.py)
- LSTM 기반의 시계열 예측 모델
- 입력: 24시간 시퀀스의 기상 데이터 (상대습도, 기압, 풍속)
- 출력: 향후 24시간의 온도 예측
- 2개의 LSTM 레이어와 2개의 Linear 레이어로 구성

### 데이터 처리 (utils.py)
- CSV 파일에서 기상 데이터 로드 및 전처리
- StandardScaler를 사용한 입력 변수 정규화
- MinMaxScaler를 사용한 타겟 변수(온도) 정규화
- 시계열 데이터를 시퀀스 형태로 변환
- 훈련/검증 데이터 분리 (8:2 비율)

### 학습 프로세스 (trainer.py)
- MSE Loss와 Adam 옵티마이저 사용
- MSE, RMSE, MAE 메트릭 추적
- 학습 과정 시각화 (training_history.png 저장)

## 사용 방법

### 1. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 모델 학습
```bash
python train.py [옵션]
```

주요 학습 매개변수:
- `--model_dir`: 모델 저장 경로 (기본값: "./models")
- `--data_path`: 데이터 파일 경로 (기본값: "./data/cleaned_weather.csv")
- `--seq_length`: 입력 시퀀스 길이 (기본값: 24)
- `--forecast_horizon`: 예측 기간 (기본값: 24)
- `--hidden_size`: LSTM 히든 레이어 크기 (기본값: 256)
- `--n_layers`: LSTM 레이어 수 (기본값: 2)
- `--dropout`: 드롭아웃 비율 (기본값: 0.2)
- `--n_epochs`: 총 학습 에폭 수 (기본값: 50)
- `--batch_size`: 배치 크기 (기본값: 64)
- `--learning_rate`: 학습률 (기본값: 0.001)
- `--verbose`: 로그 출력 레벨 (0: 없음, 1: 기본, 2: 디버그)

### 3. 예측 실행
```bash
python predict.py
```
- 가장 최근에 학습된 모델을 자동으로 로드
- 최근 24시간의 데이터를 사용하여 향후 24시간의 온도 예측
- 예측 결과를 그래프로 시각화 (images/temperature_forecast.png)

## 데이터 출처
- 출처: https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting/
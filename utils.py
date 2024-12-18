import numpy as np 
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
import os

def load_data(config, features, target):
    """
    CSV 파일에서 데이터를 로드하고 전처리.

    Args:
        config (Namespace): 설정 객체, 데이터 파일 경로를 포함.
        features (list): 학습에 사용할 특성 변수.
        target (str): 예측 대상 변수(target).

    Returns:
        pd.DataFrame: 전처리된 데이터프레임.
    """
    df = pd.read_csv(config.data_path)

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    features = features + [target]
    df = df[features]

    plt.plot(df[target])
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.title("Temperature over time")
    plt.savefig("images/temperature.png")

    return df

def create_sequences(data, features, target, seq_length, forecast_horizon):
    """
    시계열 데이터를 시퀀스 형태로 변환.

    Args:
        data (pd.DataFrame): 입력 데이터프레임
        features (list): 학습에 사용할 특성 변수
        target (str): 예측 대상 변수(target)
        seq_length (int): 시퀀스 길이
        forecast_horizon (int): 예측 대상 시간 간격

    Returns:
        Tuple[np.ndarray, np.ndarray]: 입력 시퀀스와 타겟 시퀀스
    """
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        feature_window = data[i:(i + seq_length)][features].values
        target_window = data[target][i + seq_length: i + seq_length + forecast_horizon]
        X.append(feature_window)
        y.append(target_window)
    return np.array(X), np.array(y)

def prepare_data(df, config, features, target):
    """
    데이터를 훈련 및 검증 세트로 분리하고 정규화.

    Args:
        df (pd.DataFrame): 전처리된 데이터프레임.
        config (Namespace): 설정 객체로, 시퀀스 길이 및 예측 대상 시간 간격을 포함.
        features (list): 입력 변수(feature) 리스트.
        target (str): 예측 대상 변수(target).

    Returns:
        Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
            훈련 및 검증 데이터 (X_train, y_train), (X_test, y_test).
    """
    seq_length = config.seq_length
    forecast_horizon = config.forecast_horizon

    # 데이터 분리
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    valid_df = df[train_size:]

    # MinMaxScaler를 사용하여 데이터 정규화
    mm_features = MinMaxScaler()
    mm_target = MinMaxScaler()

    train_df[features] = mm_features.fit_transform(train_df[features])
    train_df[target] = mm_target.fit_transform(train_df[target].values.reshape(-1, 1))
    valid_df[features] = mm_features.transform(valid_df[features])
    valid_df[target] = mm_target.transform(valid_df[target].values.reshape(-1, 1))
    
    print(train_df.head())
    print(train_df[features].mean())
    print(train_df[features].std())

    # 시퀀스 생성
    X_train, y_train = create_sequences(train_df, features, target, seq_length, forecast_horizon)
    X_test, y_test = create_sequences(valid_df, features, target, seq_length, forecast_horizon)
    
    # Tensor로 변환
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    print("훈련 데이터 시퀀스 형태:", X_train.shape, y_train.shape)
    print("검증 데이터 시퀀스 형태:", X_test.shape, y_test.shape)

    return (X_train, y_train), (X_test, y_test)

def generate_model_filename(config):
    """설정값을 기반으로 모델 파일 이름 생성
    
    Args:
        config: 학습 설정값을 담은 객체
        
    Returns:
        str: 생성된 모델 파일 경로
        
    Example:
        model_seq24_hor24_hid256_lay2_drop0.2_ep50_bat64_lr0.001_20231218_235959.pth
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = (
        f"model_"
        f"seq{config.seq_length}_"
        f"hor{config.forecast_horizon}_"
        f"hid{config.hidden_size}_"
        f"lay{config.n_layers}_"
        f"drop{config.dropout}_"
        f"ep{config.n_epochs}_"
        f"bat{config.batch_size}_"
        f"lr{config.learning_rate}_"
        f"{timestamp}.pth"
    )
    return os.path.join(config.model_dir, model_name)
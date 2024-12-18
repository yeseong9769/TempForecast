import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import WeatherPredicter
from utils import load_data, prepare_data, generate_model_filename
from trainer import Trainer

def define_argparser():
    p = argparse.ArgumentParser()

    # 필수 설정 값
    p.add_argument('--model_dir', default="./models", help="Directory to save model")
    p.add_argument('--data_path', default="./data/cleaned_weather.csv", help="Path to load data")

    # 모델 관련 설정값
    p.add_argument('--seq_length', type=int, default=24, help="Input sequence length")
    p.add_argument('--forecast_horizon', type=int, default=24, help="Forecast horizon")
    p.add_argument('--hidden_size', type=int, default=256, help="LSTM hidden layer size")
    p.add_argument('--dropout', type=float, default=0.2, help="Dropout rate")
    p.add_argument('--n_layers', type=int, default=2, help="Number of LSTM layers")

    # 학습 관련 설정값
    p.add_argument('--n_epochs', type=int, default=50, help="Number of total epochs")
    p.add_argument('--batch_size', type=int, default=64, help="Batch size")
    p.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    p.add_argument('--verbose', type=int, default=1, help="Verbose level (0: no log, 1: default, 2: debug)")

    config = p.parse_args()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(config.model_dir, exist_ok=True)

    config.model_fn = generate_model_filename(config)

    return config

def main(config):
    features = ['rh', 'p', 'wv']    # 입력 변수
    target = 'T'                    # 예측 대상 변수

    # Load data and Preprocess
    df = load_data(config, features, target)
    (X_train, y_train), (X_test, y_test) = prepare_data(df, config, features, target)   

    # Model Initialize
    input_shape = X_train.shape[2]
    output_shape = config.forecast_horizon

    model = WeatherPredicter(
        input_shape=input_shape,
        output_shape=output_shape,
        hidden_size=config.hidden_size,
        n_layers=config.n_layers,
        dropout_p=config.dropout
    ).to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    crit = nn.MSELoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

    # Train
    trainer = Trainer(model, optimizer, crit)
    
    history = trainer.train(
        train_data=(X_train, y_train),
        valid_data=(X_test, y_test),
        config=config
    )

    trainer.plot_history(history)

    # Save best model
    torch.save({
        'model': model.state_dict(),
        'config': config,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
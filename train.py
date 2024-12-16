import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import WeatherPredicter
from utils import load_data
from utils import prepare_data
from trainer import Trainer

def define_argparser():
    '''
    명령줄 인수를 정의하고 파싱하는 함수
    '''
    p = argparse.ArgumentParser()

    # Model File path
    p.add_argument('--model_fn', required=True)

    # Data Setttings
    p.add_argument('--data_path', required=True)
    p.add_argument('--seq_length', type=int, default=24)
    p.add_argument('--forecast_horizon', type=int, default=24)

    # Training Settings
    p.add_argument('--n_epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--learning_rate', type=float, default=0.001)

    # etc
    p.add_argument('--verbose', type=int, default=1)

    config = p.parse_args()

    return config

def main(config):
    # CUDA or CPU
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features = ['rh', 'p', 'wv']    # 입력 변수
    target = 'T'                    # 예측 대상 변수

    # Load data and Preprocess
    df = load_data(config, features, target)    
    (X_train, y_train), (X_test, y_test) = prepare_data(df, config, features, target)   

    # Model Initialize
    input_shape = X_train.shape[2]
    output_shape = config.forecast_horizon

    model = WeatherPredicter(
        input_shape,
        output_shape,
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
        'optimizer': optimizer.state_dict(),
        'config': config,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
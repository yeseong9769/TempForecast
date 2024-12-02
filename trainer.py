from sklearn.metrics import mean_squared_error
import torch
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler

class Trainer():
    """
    모델 학습 및 검증을 위한 클래스

    Args:
        model (nn.Module): 학습할 PyTorch 모델
        optimizer (torch.optim.Optimizer)
        crit (torch.nn.Module): 손실 함수
    """
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    def _batchify(self, x, y, batch_size):
        """
        데이터를 배치 단위로 나눔

        Args:
            x (Tensor): 입력 데이터
            y (Tensor): 타겟 데이터
            batch_size (int): 배치 크기

        Returns:
            Tuple[List[Tensor], List[Tensor]]: 배치로 나눠진 데이터
        """
        x = x.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)

        return x, y
    
    def _train(self, x, y, config):
        """
        한 Epoch 동안 모델 학습

        Args:
            x (Tensor): 학습 입력 데이터
            y (Tensor): 학습 타겟 데이터
            config (Namespace): 학습 설정

        Returns:
            float: 평균 학습 손실 값
        """
        self.model.train()  # 학습 모드로 변경
        total_loss = 0

        x, y = self._batchify(x, y, config.batch_size)  # 배치 단위로 데이터를 나눔

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            pred = self.model(x_i)      # 모델 예측 수행
            loss = self.crit(pred, y_i.squeeze())   # 손실 계산

            self.optimizer.zero_grad()  # 기울기 초기화
            loss.backward()             # 기울기 계산
            self.optimizer.step()       # 파라미터 업데이트

            if config.verbose >= 2:
                print("Train Iteration(%d/%d): Loss=%.4e" % (i + 1, len(x), float(loss)))

            total_loss += loss.item()   # 누적 손실

        return total_loss / len(x)      # 손실 평균 반환
    
    def _validate(self, x, y, config):
        """
        검증 데이터셋에 대한 모델 성능 평가

        Args:
            x (Tensor): 검증 입력 데이터
            y (Tensor): 검증 타겟 데이터
            config (Namespace)

        Returns:
            float
        """
        self.model.eval()   # 검증 모드로 변경

        with torch.no_grad():   # 기울기 계산 비활성화
            x, y = self._batchify(x, y, config.batch_size)
            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                pred = self.model(x_i)  # 모델 예측
                loss = self.crit(pred, y_i.squeeze())   # 손실 계산

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss)))

                total_loss += loss.item()
            
            return total_loss / len(x)
        
    def train(self, train_data, valid_data, config):
        """
        모델 학습 및 검증 루프.

        Args:
            train_data (Tuple[Tensor, Tensor]): 훈련 데이터 (X_train, y_train).
            valid_data (Tuple[Tensor, Tensor]): 검증 데이터 (X_test, y_test).
            config (Namespace)

        Returns:
            None
        """
        train_losses = []
        valid_losses = []

        lowest_loss = np.inf
        best_model = None

        for epoch in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)      # 학습 수행
            valid_loss = self._validate(valid_data[0], valid_data[1], config)   # 검증 수행

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if valid_loss < lowest_loss:    # 새로운 최적 모델인 경우
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e valid_loss=%.4e, lowest_loss=%.4e" % (
                epoch + 1, 
                config.n_epochs, 
                train_loss, 
                valid_loss,
                lowest_loss
            ))
        
        self.model.load_state_dict(best_model)  # Restore to best model
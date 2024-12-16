from copy import deepcopy
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import numpy as np
import matplotlib.pyplot as plt

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

    def _calc_metrics(self, pred, y, loss=None):
        pred_np = pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        # loss 값을 MSE로 재사용
        mse = loss.item() if loss is not None else mean_squared_error(y_np, pred_np)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_np, pred_np)
        
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}

    def _train(self, x, y, config):
        self.model.train()
        total_metrics = {'MSE': 0, 'RMSE': 0, 'MAE': 0}
        
        x, y = self._batchify(x, y, config.batch_size)
        
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            pred = self.model(x_i)
            loss = self.crit(pred, y_i.squeeze())  # MSE Loss 계산
            
            # loss 값을 이용해서 메트릭 계산
            metrics = self._calc_metrics(pred, y_i.squeeze(), loss)
            for k, v in metrics.items():
                total_metrics[k] += v
                
            self.optimizer.zero_grad()
            loss.backward()  # loss로 backward
            self.optimizer.step()
            
            if config.verbose >= 2:
                print("Train Iteration(%d/%d): Loss=%.4e" % (i + 1, len(x), float(loss)))
        
        avg_metrics = {k: v / len(x) for k, v in total_metrics.items()}
        return avg_metrics
    
    def _validate(self, x, y, config):
        self.model.eval()   # 검증 모드로 변경
        total_metrics = {'MSE': 0, 'RMSE': 0, 'MAE': 0}

        with torch.no_grad():   # 기울기 계산 비활성화
            x, y = self._batchify(x, y, config.batch_size)

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                pred = self.model(x_i)  # 모델 예측
                loss = self.crit(pred, y_i.squeeze())   # 손실 계산

                metrics = self._calc_metrics(pred, y_i.squeeze(), loss)
                for k, v in metrics.items():
                    total_metrics[k] += v

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss)))
            
            avg_metrics = {k: v / len(x) for k, v in total_metrics.items()}
            return avg_metrics
        
    def train(self, train_data, valid_data, config):
        train_histroy = {
            'train_mse': [], 'train_rmse': [], 'train_mae': [],
            'valid_mse': [], 'valid_rmse': [], 'valid_mae': []
        }

        lowest_loss = np.inf
        best_model = None

        for epoch in range(config.n_epochs):
            train_metrics = self._train(train_data[0], train_data[1], config)      # 학습 수행
            valid_metrics = self._validate(valid_data[0], valid_data[1], config)   # 검증 수행

            train_histroy['train_mse'].append(train_metrics['MSE'])
            train_histroy['train_rmse'].append(train_metrics['RMSE'])
            train_histroy['train_mae'].append(train_metrics['MAE'])
            train_histroy['valid_mse'].append(valid_metrics['MSE'])
            train_histroy['valid_rmse'].append(valid_metrics['RMSE'])
            train_histroy['valid_mae'].append(valid_metrics['MAE'])

            if valid_metrics['MSE'] < lowest_loss:    # 새로운 최적 모델인 경우
                lowest_loss = valid_metrics['MSE']
                best_model = deepcopy(self.model.state_dict())

            print("[Epoch %d/%d]" % (epoch + 1, config.n_epochs))
            print("  Train => MSE: %.4e, MAE: %.4f, RMSE: %.4f" % 
                  (train_metrics['MSE'], train_metrics['RMSE'], train_metrics['MAE']))
            print("  Valid => loss: %.4e, MAE: %.4f, RMSE: %.4f" % 
                  (valid_metrics['MSE'], valid_metrics['MAE'], valid_metrics['RMSE']))
        
        self.model.load_state_dict(best_model)  # Restore to best model
        return train_histroy
    
    def plot_history(self, history):
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(history['train_mse'], label='Train')
        plt.plot(history['valid_mse'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # MAE plot
        plt.subplot(1, 3, 2)
        plt.plot(history['train_mae'], label='Train')
        plt.plot(history['valid_mae'], label='Validation')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        # RMSE plot
        plt.subplot(1, 3, 3)
        plt.plot(history['train_rmse'], label='Train')
        plt.plot(history['valid_rmse'], label='Validation')
        plt.title('Root Mean Square Error')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('images/training_history.png')
        plt.close()
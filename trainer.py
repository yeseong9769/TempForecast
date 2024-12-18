from copy import deepcopy
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import numpy as np
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

    def _batchify(self, x, y, batch_size):
        x = x.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)
        return x, y

    def _calc_metrics(self, pred, y, loss=None):
        pred_np = pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
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
            loss = self.crit(pred, y_i.squeeze())
            
            metrics = self._calc_metrics(pred, y_i.squeeze(), loss)
            for k, v in metrics.items():
                total_metrics[k] += v
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if config.verbose >= 2:
                print(f"Train Iteration({i+1}/{len(x)}): MSE={metrics['MSE']:.4e}")
        
        return {k: v/len(x) for k, v in total_metrics.items()}
    
    def _validate(self, x, y, config):
        self.model.eval()
        total_metrics = {'MSE': 0, 'RMSE': 0, 'MAE': 0}

        with torch.no_grad():
            x, y = self._batchify(x, y, config.batch_size)

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                pred = self.model(x_i)
                loss = self.crit(pred, y_i.squeeze())

                metrics = self._calc_metrics(pred, y_i.squeeze(), loss)
                for k, v in metrics.items():
                    total_metrics[k] += v

                if config.verbose >= 2:
                    print(f"Valid Iteration({i+1}/{len(x)}): MSE={metrics['MSE']:.4e}")
            
            return {k: v/len(x) for k, v in total_metrics.items()}
        
    def train(self, train_data, valid_data, config):
        best_model = None
        best_loss = float('inf')
        history = {
            'train_mse': [], 'train_rmse': [], 'train_mae': [],
            'valid_mse': [], 'valid_rmse': [], 'valid_mae': []
        }

        for epoch in range(config.n_epochs):
            train_metrics = self._train(train_data[0], train_data[1], config)
            valid_metrics = self._validate(valid_data[0], valid_data[1], config)

            # Update history
            for k in train_metrics.keys():
                history[f'train_{k.lower()}'].append(train_metrics[k])
                history[f'valid_{k.lower()}'].append(valid_metrics[k])

            if valid_metrics['MSE'] < best_loss:
                best_loss = valid_metrics['MSE']
                best_model = deepcopy(self.model.state_dict())

            if config.verbose >= 1:
                print(f"[Epoch {epoch+1}/{config.n_epochs}]")
                print(f"  Train => MSE: {train_metrics['MSE']:.4e}, MAE: {train_metrics['MAE']:.4f}, RMSE: {train_metrics['RMSE']:.4f}")
                print(f"  Valid => MSE: {valid_metrics['MSE']:.4e}, MAE: {valid_metrics['MAE']:.4f}, RMSE: {valid_metrics['RMSE']:.4f}")

        self.model.load_state_dict(best_model)
        return history
    
    def plot_history(self, history):
        plt.figure(figsize=(15, 5))
        
        metrics = ['mse', 'mae', 'rmse']
        titles = ['Mean Squared Error', 'Mean Absolute Error', 'Root Mean Squared Error']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            plt.subplot(1, 3, i+1)
            plt.plot(history[f'train_{metric}'], label='Train')
            plt.plot(history[f'valid_{metric}'], label='Validation')
            plt.title(title)
            plt.xlabel('Epoch')
            plt.ylabel(metric.upper())
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('images/training_history.png')
        plt.close()
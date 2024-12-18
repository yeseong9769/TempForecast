import os
import torch
import matplotlib.pyplot as plt
from utils import load_data
from model import WeatherPredicter
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 최신 모델 파일 찾기
model_dir = "./models"
latest_model = max([f for f in os.listdir(model_dir) if f.endswith('.pth')])
model_path = os.path.join(model_dir, latest_model)

# Device 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 모델 로드
d = torch.load(model_path, map_location=device)
model_dict = d['model']
config = d['config']

features = ['rh', 'p', 'wv']
target = 'T'

df = load_data(config, features, target)
print(df.head())

# 데이터 준비
df = df.iloc[-24:]

ss = StandardScaler()
mm = MinMaxScaler()

df[features] = ss.fit_transform(df[features])
df[target] = mm.fit_transform(df[target].values.reshape(-1, 1))
df.drop(target, inplace=True, axis=1)

print(df.head())
print(df.shape)

# NumPy 배열로 변환 후 PyTorch 텐서로 변환
seq_tensor = torch.tensor(df.to_numpy(), dtype=torch.float32)
print(seq_tensor.shape)
seq_tensor = seq_tensor.reshape(1, 24, 3)

input_size = len(features)
model = WeatherPredicter(
    input_shape=input_size,
    output_shape=config.forecast_horizon,
    hidden_size=config.hidden_size,
    n_layers=config.n_layers,
    dropout_p=config.dropout
).to(device)

model.load_state_dict(model_dict)
model.eval()

with torch.no_grad():
    y_pred = model(seq_tensor)

prediction = mm.inverse_transform(y_pred)

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(len(prediction[0])), prediction[0], label='Temperature Forecast', marker='o')
plt.title('24-Hour Temperature Forecast')
plt.xlabel('Hours from Now')
plt.ylabel('Temperature')
plt.grid(True)
plt.legend()
plt.savefig('images/temperature_forecast.png')
plt.close()

# 예측 결과 출력
print("\nTemperature forecast for next 24 hours:")
for hour, temp in enumerate(prediction[0]):
    print(f"Hour {hour+1}: {temp:.2f}")
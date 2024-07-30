#GRU 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# 배터리 ID 목록
train_battery_ids = ['B0006', 'B0005']

test_battery_ids = [
    "B0005", "B0006", "B0007", "B0018", "B0025", "B0026", "B0027", "B0028",
    "B0029", "B0030", "B0031", "B0032", "B0033", "B0034", "B0036", "B0038",
    "B0039", "B0040", "B0041", "B0042", "B0043", "B0044", "B0045", "B0046",
    "B0047", "B0048", "B0049", "B0050", "B0051", "B0052", "B0053", "B0054",
    "B0055", "B0056"
]

# 훈련 데이터와 테스트 데이터 분리
train_data = combined_data[combined_data['battery_id'].isin(train_battery_ids)]
test_data = combined_data[combined_data['battery_id'].isin(test_battery_ids)]

# 특징과 목표 변수 분리
features = ['Voltage_measured', 'Current_measured', 'Temperature_measured']
target = 'SOH'

X_train = train_data[features].values
y_train = train_data[target].values

X_test = test_data[features].values
y_test = test_data[target].values

# 데이터 정규화
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# GRU 입력 형태로 변환 (samples, timesteps, features)
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# GRU 모델 구성
model = Sequential()
model.add(GRU(50, activation='relu', input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# 모델 학습
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=2, shuffle=False)

# 예측
y_pred = model.predict(X_test_scaled)

# 결과 출력
for i in range(len(y_test)):
    battery_id = test_data.iloc[i]['battery_id']
    print(f"Battery ID: {battery_id}, Actual: {y_test[i]}, Predicted: {y_pred[i][0]}")
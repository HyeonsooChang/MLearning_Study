from sklearn.datasets import load_wine
data = load_wine()
type(data)

print(data)

data.keys()

data.data.shape
data.data.ndim #차원 확인

import pandas as pd
pd.DataFrame(data.data, columns = data.feature_names)

#머신러닝 모델을 만들고 예측
X = data.data
y = data.target

#모델 생성 및 훈련
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X,y)

#예측
y_pred = model.predict(X)

#성능 평가
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print(classification_report(y,y_pred))
print("accuracy = ", accuracy_score(y,y_pred))


#train_test_split()을 활용하여 훈련 데이터와 테스트 데이터 직접 분리하기

from sklearn.model_selection import train_test_split
result = train_test_split(X,y,test_size=0.2, random_state=42)

print(len(result))

result[0].shape
result[1].shape

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state =42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("정답률 =", accuracy_score(y_test,y_pred))
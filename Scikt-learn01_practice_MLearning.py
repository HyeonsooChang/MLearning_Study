import numpy as np
import matplotlib.pyplot as plt
r = np.random.RandomState(10)
x = 10 * r.rand(100)
y = 2*x -3 *r.rand(100)
plt.scatter(x,y)

#입력 데이터 x와 정답 데이터 y의 모양을 확인

x.shape
y.shape

#모델 객체 생성
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model

#모델 훈련
model.fit(x,y) # 에러 발생
X = x.reshape(100,1)
model.fit(X,y)

#예측
x_new = np.linspace(-1,11,100)
X_new = x_new.reshape(100,1)
y_new = model.predict(X_new)

X_ = x_new.reshape(-1,1)
X_.shape

#회귀모델이 잘 예측했는지 성능 평가

from sklearn.metrics import mean_squared_error
error = np.sqrt(mean_squared_error(y,y_new))

plt.scatter(x,y, label='input data')
plt.plot(X_new, y_new, color = 'red', label='regression line')

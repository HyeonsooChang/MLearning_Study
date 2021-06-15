import pandas as pd
fish = pd.read_csv('http://bit.ly/fish_csv')
fish.head()

#species열에서 고유한 값 추출
print(pd.unique(fish['Species']))
fish.columns

#species열을 타깃으로 만들고 나머지를 input값으로 설정.
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_input[:5])

fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

#사이킷런의 StandardScaler 클래스를 사용하여 훈련세트와 테스트 세트를 표준화 전처리
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)   #훈련 세트의 통계값으로 테스트 세트를 변환해야하기 때문에 ss에 훈련세트 적용
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

#k-최근접 이웃 분류기의 확률 예측
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors = 3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

print(kn.classes_)
print(kn.predict(test_scaled[:5]))

import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals =4))

distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])


#로지스틱 회귀로 이진 분류 수행하기

bream_smelt_indexes = (train_target =='Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))

print(lr.coef_, lr.intercept_)

#로지스틱 회귀로 다중 분류 수행하기

lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.predict(test_scaled[:5]))

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
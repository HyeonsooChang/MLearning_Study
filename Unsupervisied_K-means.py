%matplotlib inline
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#중심점이 5개인 100개의 점 데이터를 무작위 생성
points, labels = make_blobs(n_samples=100, centers =5, n_features=2, random_state=135)

print(points.shape, points[:10])
print(labels.shape, labels[:10])

#축그리기
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

#위에서 생성한 점 데이터들을 dateframe으로 변환
points_df = pd.DataFrame(points, columns = ['X', 'Y'])
display(points_df.head())

#점 데이터를 X,Y, grid에 시각화
ax.scatter(points[:,0], points[:,1], c='black', label = 'random generated data')

#축 이름에 라벨을 달고, 점 데이터 그리기
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid()

from sklearn.cluster import KMeans

# 1), 2) 위에서 생성한 무작위 점 데이터(points)에 클러스터의 수(K)가 5인 K-means 알고리즘을 적용
kmeans_cluster = KMeans(n_clusters=5)

# 3) ~ 6) 과정이 전부 함축되어 있는 코드입니다. points에 대하여 K가 5일 때의 K-means iteration을 수행

kmeans_cluster.fit(points)
print(type(kmeans_cluster.labels_))
print(np.shape(kmeans_cluster.labels_))
print(np.unique(kmeans_cluster.labels_))

# n 번째 클러스터 데이터를 어떤 색으로 도식할 지 결정하는 color dictionary
color_dict = {0: 'red', 1: 'blue', 2:'green', 3:'brown', 4:'indigo'}

#점 데이터를 X-Y grid에 시각화합니다.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# K-means clustering의 결과대로 색깔별로 구분하여 점에 색칠한 후 도식
for cluster in range(5):
    cluster_sub_points = points[kmeans_cluster.labels_ == cluster] # 전체 무작위 점 데이터에서 K-means 알고리즘에 의해 군집화된 sub data를 분리합니
    ax.scatter(cluster_sub_points[:, 0], cluster_sub_points[:, 1], c=color_dict[cluster], label='cluster_{}'.format(cluster))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid()
import itertools
from typing import Any, List

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np


class MyKMeans:
    def __init__(self, n_clusters: int, max_iter: int = 1000, random_seed: int = 0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_seed)

    def fit(self, X: Any):
        # 指定したクラスター数分のラベルを繰り返し作成するジェネレータを生成（0,1,2,0,1,2,0,1,2...みたいな感じ）
        cycle = itertools.cycle(range(self.n_clusters))
        # 各データポイントに対してクラスタのラベルをランダムに割り振る
        self.labels_ = np.fromiter(itertools.islice(cycle, X.shape[0]), dtype=np.int)
        self.random_state.shuffle(self.labels_)
        labels_prev = np.zeros(X.shape[0])
        count = 0
        self.cluster_centers_ = np.zeros((self.n_clusters, X.shape[1]))

        # 各データポイントが属しているクラスターが変化しなくなった、又は一定回数の繰り返しを越した場合は終了
        while (not (self.labels_ == labels_prev).all() and count < self.max_iter):
            # その時点での各クラスターの重心を計算する
            for i in range(self.n_clusters):
                XX = X[self.labels_ == i, :]
                self.cluster_centers_[i, :] = XX.mean(axis=0)
            # 各データポイントと各クラスターの重心間の距離を総当たりで計算する
            dist = ((X[:, :, np.newaxis] - self.cluster_centers_.T[np.newaxis, :, :]) ** 2).sum(axis=1)
            # 1つ前のクラスターラベルを覚えておく。1つ前のラベルとラベルが変化しなければプログラムは終了する。
            labels_prev = self.labels_
            # 再計算した結果、最も距離の近いクラスターのラベルを割り振る
            self.labels_ = dist.argmin(axis=1)
            count += 1

    def predict(self, X: Any):
        dist = ((X[:, :, np.newaxis] - self.cluster_centers_.T[np.newaxis, :, :]) ** 2).sum(axis=1)
        labels = dist.argmin(axis=1)
        return labels


def implement():
    pass


def main():
    # Blob データ生成
    N_CLUSTERS = 5
    dataset = datasets.make_blobs(centers=N_CLUSTERS)
    # print(dataset)

    # 特徴データ
    features: List[Any] = dataset[0]
    # print(features)
    # targets = dataset[1] # 正解ラベルは使わない

    # クラスタリングする
    cls = KMeans(n_clusters=N_CLUSTERS)
    pred = cls.fit_predict(features)
    # print(pred)

    # 各要素をラベルごとに色付けして表示する
    for i in range(N_CLUSTERS):
        labels = features[pred == i]
        plt.scatter(labels[:, 0], labels[:, 1])

    # クラスタのセントロイド (重心) を描く
    centers = cls.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], s=100,
                facecolors='none', edgecolors='black')

    plt.show()


if __name__ == '__main__':
    main()

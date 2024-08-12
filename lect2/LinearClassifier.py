import numpy as np;
from sklearn.datasets import make_classification


class LinearClassifier:
    def __init__(self, dataset):
        self.success = 0

        self.dataset = dataset
        # len 15 array
        self.X = np.array(self.dataset[0])

        #0 ~ 9
        self.Y = np.array(self.dataset[1])

        self.trainX = np.array(self.X[:35000])
        self.trainY = np.array(self.Y[:35000])

        self.validateX = np.array(self.X[35001:45000])
        self.validateY = np.array(self.Y[35001:45000])

        self.testX = np.array(self.X[45001:])
        self.testY = np.array(self.Y[45001:])

        #15*10
        self.weight = np.zeros((15*10))

    #input is len 15 array
    def run(self, k, input):

        #각 데이터로부터 거리 구하기
        distance = np.sqrt(np.sum((self.trainX - self.testX[input])**2, axis=1))
        #다른 정도가 제일 작은 데이터 k개 구하기
        nearest_neighbor = self.trainY[np.argsort(distance)[:k]]

        #예측값
        prediction = np.argmax(np.bincount(nearest_neighbor))
        if (prediction == self.testY[input]):
            self.success += 1


linear_classifier = LinearClassifier(make_classification(n_samples=50000, n_features=15, n_informative=5, n_classes=10, random_state=40))

for i in range(len(linear_classifier.testX)):
    linear_classifier.run(50, i);

print("%s %%" % (linear_classifier.success / 5000 * 100))

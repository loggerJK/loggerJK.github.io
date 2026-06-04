---
title: 'CS231n Assignments 1 : kNN'
date: 2020-12-10
categories: DeepLearning
tags: [cs231n, deep learning, kNN]
toc: true
toc_sticky: true
---
# 0. 개요



# 1. kNN 구현

- - -
numpy와 파이썬에 익숙해지는 것이 Assignments 1의 목적인 것 같다.

아직까지는 파이썬과 numpy를 잘 활용하지 못해서 코드도 길어지고, 무언가 C스타일로 파이썬 같다....

numpy와 python을 제대로 활용하면 코드가 매우 간결해진다.

numpy의 함수, 브로드캐스팅 등에 익숙해질 필요를 느꼈다

[다른 분의 블로그](https://codestudyisdiff.tistory.com/10)를 참고해보자
- - -
- 깨달은 점 
  - numpy의 함수와 브로드캐스팅 기능을 활용하자
    - Python코드보다 속도가 훨씬 빠르다

## 1-1. 에러 해결
   

- 오래된 파이썬 자료라서 그런지 import error가 난다
- cs231n/data_utils.py
```
ImportError: cannot import name 'imread' from 'scipy.misc' (/home/jiwon/anaconda3/envs/cs231n/lib/python3.7/site-packages/scipy/misc/__init__.py)
```
import를 다음과 같이 고쳐주도록 하자
```
from matplotlib.pyplot import imread
```

## 1-2. compute_distances_two_loop 함수 작성
<details>
  <summary>접기/펼치기 버튼</summary>
  <div markdown="1">

```python
def compute_distances_two_loops(self, X):
        """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            for j in xrange(num_train):
                # 데이터 로드
                ith_te = X[i]
                jth_tr = self.X_train[j]

                # L2 distance 계산
                value = ith_te - jth_tr
                data = np.sum(np.square(value))
                dists[i, j] = data

        return dists
```
  
  </div>
</details>

## 1-3. predict_labels 함수 작성
- numpy.argsort() 함수를 이용해야 한다
  - 인덱스를 담은 ```list[]```를 반환한다
  - ```list[]```에 담긴 인덱스 순서대로 방문한다면, 오름차순으로 데이터를 접근할 수 있다.

<details>
  <summary>접기/펼치기 버튼</summary>
  <div markdown="1">

```python
def predict_labels(self, dists, k=1):
        """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            # i번째 test example
            # A list of length k he k nearest nestoring the labels of tighbors to
            # the ith test point.
            closest_y = []

            # 방문할 인덱스를 순서대로 담고 있는 배열 선언
            idx = np.argsort(dists[i])

            # k번만큼 방문하여 closest_y에 해당 example의 label을 추가한다
            for j in range(k):
                index = idx[j]
                closest_y.append(self.y_train[index])

            dict = {}

            # label의 빈도 조사
            for j in range(len(closest_y)):
                if closest_y[j] in dict:
                    dict[closest_y[j]] += dict[closest_y[j]]
                else:
                    dict[closest_y[j]] = 1

            # 가장 작은 빈도가 나온 label을 고른다
            # 최대 빈도는 2k를 넘을 수 없다
            max = 0
            max_label = 0
            for j in list(dict.keys()):
                # 기존보다 큰 라벨을 발견한다면
                if max < dict[j]:
                    max = dict[j]
                    max_label = j

            # max_label 저장
            y_pred[i] = max_label

        return y_pred
```

  </div>
</details>

## 1-4. compute_distances_one_loop 함수 작성


<details>
  <summary>접기/펼치기 버튼</summary>
  <div markdown="1">

```python
def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            value = self.X_train - X[i]
            value = np.square(value)
            value = np.sum(value, axis=1)
            value = np.sqrt(value)
            value = np.reshape(value, [1, num_train])
            dists[i] = value

        return dists
```

  </div>
</details>


## 1-5. compute_distances_no_loop

<details>
  <summary>접기/펼치기 버튼</summary>
  <div markdown="1">

```python
 def compute_distances_no_loops(self, X):
   """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        value = []
        value.append(np.sum(np.square(X), axis=1).reshape([num_test, 1]))
        value.append(np.sum(np.square(self.X_train), axis=1).reshape([1, num_train]))
        value.append(np.matmul(X, self.X_train.T) * (-2))
        dists = np.sqrt(np.sum(value))
        return dists
```

  </div>
</details>

## 1-6. Cross Validation 구현

<details>
  <summary>코드 보기</summary>
  <div markdown="1">

```python
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
pass
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)


# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
pass
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# 하이퍼 파라미터 = K
# 총 k개의 모델을 테스트한다
for k in k_choices:
    # num_folds만큼 K-fold Cross-Validation을 실행한다
    folds_acc = []
    for fold_num in range(num_folds):
        accuracy = 0
        # validation set 설정
        X_tr = np.array(X_train_folds[:fold_num] + X_train_folds[fold_num + 1 :])
        y_tr = np.array(
            np.array(y_train_folds[:fold_num] + y_train_folds[fold_num + 1 :])
        )
        X_tr = X_tr.reshape(X_tr.shape[0] * X_tr.shape[1], X_tr.shape[2])
        y_tr = y_tr.flatten()
        X_val = X_train_folds[fold_num]
        y_val = y_train_folds[fold_num].flatten()
        # print("k = ", k, " fold_num = ", fold_num)
        # print("X_tr = ", X_tr.shape)
        # print("y_tr = ", y_tr.shape)
        # print("X_val = ", X_val.shape)
        # print("y_val = ", y_val.shape)

        # classifier 로딩
        classifier = KNearestNeighbor()
        # train데이터 입력
        classifier.train(X_tr, y_tr)
        # Validation Set으로 Predict
        y_pred = classifier.predict(X_val, k=k, num_loops=0)
        # Accuracy를 계산한다
        num_correct = np.sum(y_pred == y_val)
        accuracy += float(num_correct) / (X_val.shape[0])
        folds_acc.append(accuracy)

    # k_to_accuracies 업데이트
    k_to_accuracies[k] = folds_acc



# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

```
  </div>
</details>

## 1-7. Cross Validation 결과 분석
![graph](https://user-images.githubusercontent.com/12046879/102012992-a5bacd80-3d90-11eb-873d-57267614fe0e.png)

그래프를 보면 ```k=10```부근에서 정확도가 제일 높은 것을 볼 수 있다.

```k=9```로 predict를 진행한 결과는 다음과 같다.

```python
# Based on the cross-validation results above, choose the best value for k,   
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.
best_k = 9

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
```
```text
Got 145 / 500 correct => accuracy: 0.290000
```


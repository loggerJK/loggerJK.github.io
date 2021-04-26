---
title : 'CS231n Assignments 1 : SVM'
date : 2021-04-25
categories : DeepLearning
tags: [cs231n, deep learning, svm, SVM]
toc: true
toc_sticky: true
---

# 1. svm_loss_naive 구현하기

cs231n.classifiers.linear_svm 안에 있는 svm_loss_naive를 구현하는 문제이다

## 1-1. svm_loss_naive : gradient를 어떻게 구할 것인가?
과제를 구현하기 위해서는 SVM Loss를 직접 미분해서 $$\frac{dL}{dW}$$를 구해야한다.

이 gradiet를 계산하는데 애를 먹어서 정리해보고자 한다.

- 참고한 웹 사이트 :
  - [https://cs231n.github.io/optimization-1/#gradcompute](https://cs231n.github.io/optimization-1/#gradcompute)

들어가기에 앞서 이용되는 변수 및 수식을 정리하자

- $$ N,\,D,\,C\, =  \text{(배치개수), (차원), (분류하는 class 개수)}$$
- $$X_i$$는 $$\text{i-th column vector of X}$$
- $$X^T_j$$는 $$\text{j-th row vector of X}$$이다.
- W : weights
  - shape : $$ (D,C) $$
- X : batch data
  - shape : $$(N,D)$$
  - ![image](https://user-images.githubusercontent.com/12046879/115996639-08638600-a61b-11eb-85c8-e008e93cad22.png){: width='30%', height='30%'}
- S : scores를 저장하는 matrix
  - shape : $$(N,C)$$
  - $$ S = XW$$

먼저 하나의 data point vector $$x_i$$에 대한 SVM Loss는 $$L_i$$라는 하나의 스칼라 값이며, 다음과 같이 나타낼 수 있다. signle data point vector는 $$X$$의 row vector이므로 $$x_i = X^T_i$$ 로 표기했다.  

$$ L_i = \sum_{j_i \neq y_i} max (0, \, X^T_i W_j - X^T_i W_{y_i} + \Delta)$$

그림으로 이해해보자

Single data point $$x_i$$에 대한 score를 저장하는 matrix $$S$$는 다음과 같을 것이다

![image](https://user-images.githubusercontent.com/12046879/115980772-310f5f80-a5ca-11eb-94d1-8db1d3ea0891.png){: width='30%' height = '30%'}

이 $$S$$ matrix에서 '한 칸'마다 score가 저장되어 있고, 총 칸의 개수는 $$C$$이다. 그림에서 정답 레이블에 해당하는 '$$y_i$$번째 칸' $$= S_{y_i}$$ 이 색칠해져 있다. $$S$$의 각 칸은 스칼라 값으로, 다음의 수식적 연관성을 가지고 있다는 점을 이해하자. 

$$ \begin{cases} S_{y_i} = W^T_{y_i} x_i \\ S_j =  W^T_j x_i \end{cases}$$

$$ W^T_j \in \mathbb{R}^{1 \times D}$$

$$\, x_i = X^T_i \in \mathbb{R}^{D \times 1}$$

풀어쓰자면, $$\text{(j번째 스코어) = (single data point = row vector)} \times \text{(W의 j번째 colmun vector = j번째 class에 대한 가중치)}$$

이 각각의 스코어를 모두 더한 것이 Loss 값 $$L_i$$인데, 시그마에서 알 수 있듯이
- 총 C-1개의 항으로 구성되어 있다 (정답 레이블에 해당하는 칸은 계산하지 않는다)
- 이 항들은
  - 0이거나 
  - $$margin = X^T_i W_j - X^T_i W_{y_i} + \Delta$$ 값이다.


### **$$\frac{dL}{dW_{y_i}}$$는 어떻게 계산할까?**

1. 항의 값이 0인 경우 : 이 경우의 미분값은 당연히 0이다
2. 항의 값이 $$margin = X^T_i W_j - X^T_i W_{y_i} + \Delta$$인 경우 : 다음의 그림을 참고하자  
  
![image](https://user-images.githubusercontent.com/12046879/115993311-304bed00-a60d-11eb-8c82-2ede0827bf52.png)

위 그림과 같이 loss $$L_i$$의 값이 표현이 되고, 결국 
$$\frac{dL}{dW_{y_i}} = -\text{(margin 항의 개수)} \times x_i$$
 로 나타낼 수 있다.

 ( (margin 항의 개수)만큼 vector $$x_i$$를 scailing한 값이라고 할 수 있다)

### **$$\frac{dL}{dW_j}$$ 는 어떻게 계산할까?**

1. 항의 값이 0인 경우 : 이 경우의 미분값은 당연히 0이다
2. 항의 값이 $$margin = X^T_i W_j - X^T_i W_{y_i} + \Delta$$인 경우 :
   1. $$margin$$ 항을 $$W_j$$로 미분한 값이 $$x_i$$이므로
   2. $$W_{y_i}$$의 경우와 같이 $$\frac{dL}{dW_{y_i}} = \text{(margin 항의 개수)} \times x_i$$ 라고 할 수 있다

### 함수 svm_loss_naive의 구현

- Score Matrix $$S$$를 순회하는데
- 첫번재 for문을 이용해서 data point $$x_i$$에 대한 score vector를 순회하고 (즉, row 순회)
- 두번째 for문을 이용해서 $$x_i$$에 해당하는 C개의 score를 순회하면서 (column 순회, 정답 레이블에 해당하는 칸은 제외한다)
  - 만약 $$margin > 0$$이라면
    - loss 값을 업데이트한다
    - $$\frac{dL}{dW_{y_i}}$$ 값을 업데이트 한다
    - $$\frac{dL}{dW_j}$$ 값을 업데이트 한다

<details>
  <summary>코드 보기</summary>
  <div markdown="1">

```python
def svm_loss_naive(W, X, y, reg):
    """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        # dW = X[i].T.dot(scores)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, y[i]] -= X[i]
                dW[:, j] += X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW
```

  </div>
</details>

## 1-2. svm_loss_vectorize : 벡터 연산을 이용하여 작성하기

아마 SVM과제 중에서 제일 애를 먹었던 부분이 아닌가 싶다. Matrix Equation에 대한 직관이 부족해서 vectorize를 하는데 시간이 꽤 오래 걸리는 편인데... 이렇게 공부하면서라도 실력이 늘었으면 좋겠다3

### loss 값 코드의 vectorize
loss 코드의 vectorize는 비교적 쉬운 편이다

- 전체 batch data를 담고 있는 matrix $$X$$와 weight $$W$$간에 matrix multiplication을 수행한다. 그러면 전체 batch data에 대해서, 각 data마다 C개의 score 정보를 저장하고 있는 $$(N,C)$$ shape의 matrix $$S$$를 얻을 수 있다

![image](https://user-images.githubusercontent.com/12046879/115995412-0c40d980-a616-11eb-9c4e-89760963684e.png){: width='30%',height='30%'}

< $$3 \times 3 $$ matrix $$S$$의 예시. 정답 레이블에 해당하는 부분은 빨간색으로 동그라미를 쳐놨다 >


- '정답 레이블이 아닌 scores'들에 대해서 margin을 구하는 과정, $$S_j - S_{y_i} + \Delta$$를 수행한다. (여기서 $$\Delta$$의 값은 임의로 1이라 하였다' 결과는 아래 그림과 같이 나타날 것이다.
- 앞으로 이 matrix를 $$margin$$ matrix라 한다

![image](https://user-images.githubusercontent.com/12046879/115995662-1fa07480-a617-11eb-84bf-eae506afdedc.png){: width='30%', height = '30%'}

<margin matrix의 예시. 0보다 큰 margin이 발생한 case에 대해서 파란색 동그라미를 쳐놓았다.>

- 이제 정답이 아닌 레이블들에 대해 $$max (0, margin) $$ 작업을 수행해주자. 그 결과는 아래 그림과 같다.(주의, 정답 레이블의 경우는 loss계산에서 반영이 되지 않기에 0으로 그 값들을 바꾸어주었다)

![image](https://user-images.githubusercontent.com/12046879/115995896-15cb4100-a618-11eb-87bf-3f99ff258795.png){: width='30%', height = '30%'}

- 다 되었다. 이제 이 값들을 모두 더하고 num_train(=3)으로 나누어주어 평균 loss를 계산, 반환해주면 된다. 이것으로 loss의 계산은 끝이 난다

### $$dW_j$$ 코드의 vectorize

기존의 $$margin$$ matrix에서 1.양수였던 값들을 일괄적으로 1로 변경하고 2. 정답 레이블에 해당하는 값들은 0으로 변경해준 matrix를 $$S^\prime$$이라고 하자. 이때, $$S^\prime \in \mathbb{R}^{N \times C}$$

(정답 레이블에 해당하는 값들을 0으로 바꾸어준 이유는 추후 계산의 용이성 때문이다)

![image](https://user-images.githubusercontent.com/12046879/116008130-f5b67480-a64d-11eb-93d2-7a96ff179047.png){: width='30%', height='30%'}


svm_loss_naive가 $$dW_j$$에 하는 일은 결과적으로 다음 수식과 같이 나타낼 수 있다. 단, $$S_{i,j}$$는 정답레이블이 아니다

$$\bullet \,\, {dW}_j = \sum^N_{i=1} S^\prime_{i,j} X^T_i $$

$$ \begin{align} \bullet \,\, dW_j & =  S^\prime_{1,j} X^T_1 + S^\prime_{2,j} X^T_2 + S^\prime_{3,j} X^T_3 + \cdots + S^\prime_{N,j} X^T_N \\ & = \begin{bmatrix} X^T_1 & X^T_2 & X^T_3 \cdots X^T_n \end{bmatrix} \begin{bmatrix}  S^\prime_{1,j} \\  S^\prime_{2,j} \\  S^\prime_{3,j}  \\ S^\prime_{N,j}\end{bmatrix} \\ & = X^T S^\prime_j \,\,\, \text{for i,j such that   } y_i \neq \text{j} \end{align}$$

이제 이 $$dW_j$$에 대한 식을 $$dW$$ 전체로 확장해서 생각할 수 있다.

- $$dW = X^T S_j$$

### $$dW_{y_i}$$ 코드의 vectorize

- 놓치지 말아야 할 포인트 : $$ y_i $$는 $$ i $$에 고유하게 대응되는 상수값이다. 즉, $$ i $$가 변하면 $$ y_i $$도 변한다. 

- $$dW_{y_i} = -(S^\prime \text{의 i번째 행 margin의 개수}) \times X^T_i$$


- 정답 label에 해당하는 margin 값의 위치는 $$S^\prime_{i,y_i}$$로 나타낼 수 있다. 

- $$ row \,\, index = i, \, column \,\, index = y_i$$ 라는 점을 이용하여, 

1. $$S^\prime_{i,y_i}$$ =  $$ - $$($$S^\prime$$의 $$i$$번째 행에서 margin의 개수) 
2. 나머지 = 0

인 martix $$S^{\prime\prime}$$을 생각해보자. 이 matrix를 관찰해보면 다음 그림과 같은 대응 관계를 가진다.

![Imgur](https://i.imgur.com/xaqXXMa.png){: width='50%', height='50%'}

- 우리가 최종적으로 원하는 것은 $$ dW = \begin{bmatrix} dW_1 \quad dW_2 \quad  dW_3 \quad  \cdots \quad  dW_C \end{bmatrix} $$ 으로, 다음 vector equation을 이용해 한번에 구할 수 있다.

- $$ dW = X^T S^{\prime\prime} $$

### 두 코드의 통합

두 matrix $$ S^\prime $$과 $$ S^{\prime\prime} $$을 살펴보자.
- 한 쪽이 의미있는 값을 가지는 경우에, 대응되는 다른 쪽은 값이 0이고,
- 결과적으로 똑같이 $$ X^T $$와 행렬 곱을 수행했을 때, 그 결과값으로 dW를 출력한다
- 이 두 dW는 결국 나중에 서로 합쳐진다

$$ \therefore $$ 두 matrix를 합쳐서 한번에 계산할 수 있다는 결론이 나온다.

![alt](https://i.imgur.com/JxmvEwW.png)

$$ \therefore \,\, dW = X^T M $$

### 코드 구현 : 평균과 Regularization까지

<details>
  <summary>코드 보기</summary>
  <div markdown="1">

```python
def svm_loss_vectorized(W, X, y, reg):
    """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # Compute the loss
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    correct_class_scores = scores[np.arange(num_train), y]
    # print(correct_class_scores.shape)
    correct_class_scores = correct_class_scores.reshape(num_train, 1)
    # print(correct_class_scores.shape)
    margin = np.maximum(0, scores - correct_class_scores + 1)
    margin[np.arange(num_train), y] = 0  # 정답 라벨은 0으로 만든다
    loss += margin.sum()
    loss /= num_train
    loss += reg * np.sum(W * W)

    margin[margin > 0] = 1
    valid_margin_counts = margin.sum(axis=1)

    # print(margin.shape)
    # print(valid_margin_counts.shape)

    margin[np.arange(num_train), y] -= valid_margin_counts

    dW += X.T.dot(margin)
    dW /= num_train
    dW += reg * 2 * W

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
```
  </div>
</details>






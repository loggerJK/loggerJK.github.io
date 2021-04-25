---
title : 'CS231n Assignments 1 : SVM'
date : 2021-04-25
categories : DeepLearning
tags: [cs231n, deep learning, kNN]
toc: true
toc_sticky: true
---

# 1. svm_loss_naive 구현하기

cs231n.classifiers.linear_svm 안에 있는 svm_loss_naive를 구현하는 문제이다

## 1-1. gradient를 어떻게 구할 것인가?
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
- S : scores를 저장하는 matrix
  - shape : $$(N,C)$$
  - $$ S = XW$$

먼저 하나의 data point vector $$x_i$$에 대한 SVM Loss는 $$L_i$$라는 하나의 스칼라 값이며, 다음과 같이 나타낼 수 있다
signle data point vector는 $$X$$의 row vector이므로 $$x_i = X^T_i$$ 로 표기했다.  

$$ L_i = \sum_{j_i \neq y_i} max (0, \, X^T_i W_j - X^T_i W_{y_i} + \Delta)$$

그림으로 이해해보자

Single data point $$x_i$$에 대한 score를 저장하는 matrix $$S$$는 다음과 같을 것이다
![image](https://user-images.githubusercontent.com/12046879/115980772-310f5f80-a5ca-11eb-94d1-8db1d3ea0891.png){: width='75%' height = '75%'}

이 $$S$$ matrix에서 '한 칸'마다 score가 저장되어 있고, 총 칸의 개수는 $$C$$이다. 그림에서 정답 레이블에 해당하는 '$$y_i$$번째 칸' $$= S_{y_i}$$ 이 색칠해져 있다. $$S$$의 각 칸은 스칼라 값으로, 다음의 수식적 연관성을 가지고 있다는 점을 이해하자. 

$$S_{y_i} = W^T_{y_i} x_i$$

$$S_j =  W^T_j x_i$$

$$ W^T_j \in \mathbb{R}^{1 \times D}$$

$$\, x_i = X^T_i \in \mathbb{R}^{D \times 1}$$

풀어쓰자면, $$\text{(j번째 스코어) = (single data point = row vector)} \times \text{(W의 j번째 colmun vector = j번째 class에 대한 가중치)}$$

이 각각의 스코어를 모두 더한 것이 Loss 값 $$L_i$$인데, 시그마에서 알 수 있듯이
- 총 C-1개의 항으로 구성되어 있다 (정답 레이블에 해당하는 칸은 계산하지 않는다)
- 이 항들은
  - 0이거나 
  - $$margin = X^T_i W_j - X^T_i W_{y_i} + \Delta$$ 값이다.


**$$\frac{dL}{dW_i}$$는 어떻게 계산할까?**







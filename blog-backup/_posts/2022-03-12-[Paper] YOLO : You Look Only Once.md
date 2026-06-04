---
title : '[Paper] YOLO : You Look Only Once'
date : 2022-03-12
categories : DeepLearning
tags : [yolo, vision, you look only once, deeplearning, object detection]
toc : true
toc_sticky : true
---

# Abstract
- 기존 : classifier를 detection을 수행하도록 바꾸는 방식 ➡️ YOLO는 object detection을 regression 문제로 접근함.
- **단일 네트워크가 bounding box와 class probability 예측을 한번에 수행**
- 모든 pipeline이 단일 네트워크이고, 따라서 end-to-end 학습이 가능함
- **빠르다**
    - base모델은 초당 45프레임, fast 모델은 초당 150프레임 (NVIDIA TITAN X 기준)
    - fast모델마저 다른 real-time detector보다 두배 높은 mAP 성능
- Localization error는 타 모델에 비해 더 많지만, background에 대한 false positive (배경을 물체로 검출하는 에러)는 더 적다.
- 물체의 일반적인 표현을 학습한다
    - 예술작품과 같은 다른 도메인에 일반화할 때도 다른 모델들 (DPM, R-CNN, ...)을 압도하는 성능을 보인다.

# 1. Introduction
![Figure 1](https://i.imgur.com/l3y3LnH.png)

YOLO는 기존의 모델에 비해 매우매우(!) 간단하다. Single Convolution Network가 여러개의 bounding box를 예측하고 이에 대한 class probability까지 예측한다. 기존에 각 부분을 따로 따로 학습해야 했던 DPM, R-CNN에 비하면 엄청난 장점이다.

장점 1.(다시 한번) YOLO는 빠르다. 25ms 이하의 레이턴시로 동영상 실시간 스트리밍이 가능할 정도이다. 게다가 다른 real-time 시스템에 비해서 mAP는 두배가 더 높다. 이는 [데모 사이트](http://pjreddie.com/yolo/)에서 확인할 수 있다.

장점 2. YOLO는 training과 prediction을 진행할 때 이미지 전체를 본다. 그렇기 때문에 (sliding window 방식과 다르게) 이미지의 contextual 정보까지 얻을 수 있다. Fast R-CNN은 배경을 물체로 인식하는 실수를 잘 범하는데, larger context를 못봐서 그렇다. YOLO는 Fast R-CNN에 비해서 배경 실수가 절반밖에 안된다.

장점 3. 일반화가 잘된다. 일반적인 (natural)이미지로 학습하고 artwork로 테스트 했을때, YOLO는 다른 DPM, R-CNN 등의 모델을 큰 차이로 이겼다. 결과적으로 다른 도메인이나 예상치 못한 입력에도 잘 대응할 수 있다는 말이 된다.

단점. YOLO는 여전히 (당시의) SOTA 모델들보다 정확도가 떨어진다. 또한 어떤 물체를 정확히 localize하는 것에는 어려움을 겪고, 작은 물체일수록 심하다. 

# 2. Unified Detection
먼저 이미지를 $S \times S$ grid로 나눈다. 먄약 object의 중심 좌표가 어떠한 grid cell에 위치한다면, 그 grid cell은 object를 검출해야 한다.

각각의 grid cell은 (1) B개의 bounding box (2) 각각의 bounding box에 대한 confidence score를 에측한다. Confidence score는 다음을 반영한다. (1) 모델이 box 안에 object가 있음을 확신하는 정도 (2) 예측한 bounding box가 ground truth bounding box와 얼마나 일치하는가. 따라서 confidence는 다음과 같이 정의된다.

$$\text{confidence = Pr(Object)} \ * \text{IOU}_{\text{pred}}^{\text{truth}}$$


만약 object가 cell 안에 없다면, confidence score는 0이어야 한다. 반대로 object가 cell 안에 있다면, confidence score는 예측한 box와 ground truth box 간의 IoU 값이 될 것이다.

결과적으로 각각의 bounding box는 5개의 예측값 $(x,y,w,h,\text{confidence})$를 가진다. $(x,y)$는 grid cell에 대한 중심 좌표의 *상대적* 위치를 나타낸다. $(w,h)$는 각각 이미지 전체에 대해 상대적인 width와 height 값을 나타낸다. 마지막으로 conficence는 예측 box와 ground-truth box 간의 IOU 값을 예측한다. 


Bounding box의 개수 $B$와는 상관 없이, 각각의 grid cell은 $C$개의 조건부 클래스 확률 $\text{Pr(class}_i \| \text{Object})$을 계산한다. Test time에서는 조건부 클래스 확률과 box confidence 값을 곱해서 box마다 클래스별 confidence score를 얻는다. 이 confidence score는 (1) 해당 box 안에 해당 class의 물체가 있을 확률 (2) 에측한 bounding box가 정확할 확률을 의미한다.


> Bounding Box는 단순히 **물체가 있을 확률**만을 고려한 confidence score를 가진다. 어느 class일지는 각각의 grid cell이 고려한다.

![Figure 2](https://i.imgur.com/6kb2pp3.png)

결과적으로, prediction은 $S \times S \times (B * 5 + C)$ 텐서로 인코딩된다.

PASLCAL VOC 데이터셋에서 YOLO는 $S = 7, B = 2, C = 20$으로, 결과적으로는 $7 \times 7 \times 30$의 텐서를 가진다

## 2.1 Network Design
GoogLeNet에서 영감을 받은 네트워크 디자인. 24개의 convolutional layer와 2개의 fully connected layer로 구성된다. GoogLeNet의 Inception 모듈 대신 $1 \times 1$ reduction layer와 $3 \times 3$ convolutional layer의 조합을 사용한다. 전체 네트워크 디자인은 다음과 같다.

![Figure 3](https://i.imgur.com/demhMBg.png)

Fast YOLO는 24개 대신 9개의 convolutional layer만 사용하고, 레이어의 filter 가 더 적다.

## 2.2 Training
1k ImageNet 데이터로 처음 20개 convolution network의 pretraining을 진행한다. (저자들은 일주일동안 학습을 진행해서 GoogLeNet에 필적하는 single crop top-5 accuracy를 달성했다.) 이후 4개의 convolutional layer와 2개의 fully connected layer를 이어붙인다. Detection에서는 이미지 품질이 중요하므로 input resolution을 $224 \times 224$에서 $448 \times 448$로 끌어올렸다.

마지막 layer는 class probability와 bounding box 좌표를 둘 다 예측한다. 이 때, width와 height는 normalize하여 [0,1]이 되도록 한다. 또한 $(x,y)$는 특정 grid cell location의 offset이 되도록 하기 위해서 역시 [0,1]의 값이 되도록 한다.

마지막 layer에 대한 linear activation function으로는 leaky rectified linear activation을 사용한다.

$$\phi(x) = 
\begin{cases} 
x, & \text{if}\quad x > 0 
\\ 0.1x, & \text{otherwise}
\end{cases}$$

YOLO는 최적화하기가 쉽기 때문에 loss 함수로 sum-squared error를 사용한다. 하지만 이 loss 함수가 'AP 최대화'라는 우리의 목표와 완전히 일치하는 것은 아니다. 왜냐면 **localization error와 classification error를 동일하게 취급하기 때문**이다. 대다수의 object를 포함하지 않는 cell에서 confidence score는 0일 것이고, object를 포함하는 다른 cell들의 gradient 영향력이 지나치게 크도록 하는 문제를 낳는다. 이는 모델의 불안정성 및 발산으로 이어진다.

이를 해결하기 위해, 우리는 bounding box 좌표에 대한 loss는 증가시키는 한편, object를 포함하지 않는 box의 confidence 예측에 대한 loss는 감소시킨다. 이를 위해서 $\lambda_{\text{coord}} = 5, \lambda_{\text{noobj}} = 0.5$라는 파라미터를 이용한다.

같은 small deviation이더라도, large box보다는 small box에서의 small deviation이 더 영향이 크다. 하지만 sum-squared error는 이러한 특성을 제대로 반영하지 못한다. 논문에서는 이 문제를 커버하기 위해서 width와 height 값에 루트를 취하는 방식으로 접근을 시도했다.

YOLO는 그리드 셀마다 여러개의 bounding box를 예측하는데, 학습 시에는 오직 하나의 bounding box만이 하나의 object를 예측하도록 한다.(이때, 이 boudning box가 해당 object에 대해 *Responsible* 하다고 한다.) 이것은 predictor의 specialization으로 이어지고 각각의 predictor는 어떠한 크기, 비율, 물체의 클래스 등등으로 더 잘 예측하게 되어 결과적으로는 전체적인 재현율(recall)을 높인다.

Loss function은 다음과 같다.

$$ \displaystyle
\lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] 
\\
+ \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right] 
\\
+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2
\\
+ \lambda_{\text{noob}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2
\\
+ \sum_{i=0}^{S^2} \mathbb{1}_i^{\text{obj}} \sum_{c \ \in \ \text{classes}} (p_i(c) - \hat{p}_i(c))^2
$$

- $\mathbb{1}_i^{\text{obj}}$ : object가 cell $i$에 있는 경우
- $\mathbb{1}_{ij}^{\text{obj}}$ : cell $i$의 bounding box $j$가 해당 예측에 Responsible 한 경우
- line 1~4는 bounding box에 대한 loss
    - line 1~3 : object를 포함하는 bounding box인 경우
        - line 1~2 : localization error
        - 3 : confidence error
    - line 4 : object를 포함하지 않는 boudning box인 경우, confidence error

Loss function에 두가지 특징이 있다. (1) line 4를 보면, grid cell에 object가 위치한 경우에만 페널티를 주고 있다. (2) line 1~2에서는 ground-truth box에 대해 Responsible한 (예를 들어, IoU가 제일 높은) box의 좌표 error에만 페널티를 주고 있다.

## 2.4 Limitations of YOLO
### Spatial Constraints
YOLO는 하나의 grid cell이 2개의 bounding box, 하나의 class만 예측할 수 있다는 단점을 가지고 있다. 이러한 spatial constraint로 인해 물체가 서로 근접해 있는 경우, 물체가 작은 경우에 잘 검출하지 못하는 경우가 있다.
### Generalization
모델이 데이터로부터 bounding box를 학습하기 때문에, 새로운 비율에 일반화하는 능력은 떨어진다. 또한 bounding box를 예측할 때 downsampling layer를 통해 얻어진 저품질의 feature를 이용한다는 것도 모델의 전체적인 능력을 저하한다.
### Loss Function
손실 함수가 bounding box의 크기와 관계없이 모두 error를 동일하게 대한다. large box에서의 small error는 일반적으로 무시할만한 반면, small box에서의 small error는 IOU에서 훨씬 영향이 크다. YOLO의 error의 대부분은 부정확한 localization 때문이다.


---
title : 행렬곱의 미분 (Chain rule)
date : 2021-04-30
categories : DeepLearning
tags :
toc : true
toc_sticky : true
---

$$ A = BC \\ L = f(A) $$ 
라고 하자.

$$\frac{dL}{dB} = C^T \frac{dL}{dA}$$
$$\frac{dL}{dC} = \frac{dL}{dA} B^T$$

증명 : 나중에 정리하는 걸로...
왜 저렇게 되는지 아직은 이해를 못하겠다

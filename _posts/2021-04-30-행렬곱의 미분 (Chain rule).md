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

$$\frac{dL}{dB} = \frac{dL}{dA} C^T $$


$$\frac{dL}{dC} =  B^T \frac{dL}{dA}$$

증명 : 나중에 정리하는 걸로...
왜 저렇게 되는지 아직은 이해를 못하겠다

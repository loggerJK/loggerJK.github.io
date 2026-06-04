---
title: Transformer를 구현하며 배운 것들
date: 2022-01-07
categories: DeepLearning
tags: [Pytorch, torch, transformer]
toc: true
toc_sticky: true
---

# Pytorch Layer Initialization

## 첫번째 : apply() 함수 이용하기

```python
def apply_xavier(layer):
    if hasattr(layer, 'weight'):
        print(layer)
        torch.nn.init.xavier_uniform_(layer.weight)

encoder_layer=EncoderLayer(128,8,2048)
encoder_layer.apply(apply_xavier)
```

```
Output:

Linear(in_features=128, out_features=128, bias=True)
Linear(in_features=128, out_features=128, bias=True)
Linear(in_features=128, out_features=128, bias=True)
Linear(in_features=128, out_features=128, bias=True)
Linear(in_features=128, out_features=2048, bias=True)
Linear(in_features=2048, out_features=128, bias=True)
LayerNorm((128,), eps=1e-05, elementwise_affine=True)
```

## 두번째 : named_parameters() 함수 이용하기

named_parameters() 함수는 (param_name, param_weight) 형태의 튜플을 반환한다.

```python
for param in encoder_layer.named_parameters():
  print(param[0])
```

```
Output:

multiheadattention.fcQ.weight
multiheadattention.fcQ.bias
multiheadattention.fcK.weight
multiheadattention.fcK.bias
multiheadattention.fcV.weight
multiheadattention.fcV.bias
multiheadattention.fcOut.weight
multiheadattention.fcOut.bias
ffn.fc1.weight
ffn.fc1.bias
ffn.fc2.weight
ffn.fc2.bias
layerNorm.weight
layerNorm.bias
```

Xavier Uniform Initilization을 이용하고자 한다면 다음과 같이 조건식을 추가하여 초기화하면 된다. (bias와 nn.layerNorm()은 초기화 대상이 아니므로 제외해준 모습을 볼 수 있다.)

```python
for layer in encoder_layer.named_parameters():
    if 'weight' in layer[0] and 'layerNorm' not in layer[0]:
        print(layer[0])
        torch.nn.init.xavier_uniform_(layer[1])
```

```
Output:

multiheadattention.fcQ.weight
multiheadattention.fcK.weight
multiheadattention.fcV.weight
multiheadattention.fcOut.weight
ffn.fc1.weight
ffn.fc2.weight
```

# Pytorch에서 List 형식으로 layer 선언하기

처음에 작성했던 코드는 다음과 같다.

```python
class Decoder(nn.Module):
    def __init__ (self, N, hidden_dim, num_head, inner_dim):
        super().__init__()

        self.dec_layers = []
        for i in range(N):
            self.dec_layers.append(DecoderLayer(hidden_dim, num_head, inner_dim))

```

이 코드는 문제가 있는 코드이다. 왜일까? 저렇게 단순히 Python 리스트에 레이어를 집어넣어서 사용하면 Pytorch가 layer를 제대로 인식하지 못하는 상황이 벌어지기 때문이다. 이 말인 즉, 상위 layer에서 `children()`을 호출해도 저 `self.dec_layers`안의 layer들은 호출되지 않는다는 이야기이다. 당연히 `model.parameters()`를 호출해도 저 layer들의 parameter들은 누락되게 되고, 학습을 해도 optimizer가 최적화하지 않는 치명적인(!) 상태가 된다. (당연하다, optimizer에 parameter가 등록되어 있지 않으니까.)

그러면 어떻게 해야할까?

## nn.ModuleList

이를 위해서, Python은 `nn.ModueList`를 제공한다. 사용법은 다음과 같다

```python
self.dec_layers = nn.ModuleList([DecoderLayer(hidden_dim, num_head, inner_dim) for _ in range(N)])
```

그러면 Pytorch에서 정상적으로 layer들을 인식한다. 쓸 때는 일반적인 반복문처럼 `for layer in self.dec_layers:`으로 사용하면 된다.

# Pytorch에서 += 연산자의 위험성

[참고 사이트](https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/4)

Pytorch에서 layer를 짤 때 다음과 같은 코드는 조심해야 한다.

```python
    def forward(self, input, mask = None):

        # input : (bs, seq_len, hidden_dim)

        # encoder attention
        # uses only padding mask
        output = self.multiheadattention(srcQ= input, srcK = input, srcV = input, mask = mask)
        output = self.dropout1(output)
        output += input
        output = self.layerNorm(output)

        output_ = self.ffn(output)
        output_ = self.dropout2(output_)
        output += output
        output = self.layerNorm(output)

        # output : (bs, seq_len, hidden_dim)
        return output
```

왜냐? 이 `+=` 연산자가 바로 inplace 연산자이기 때문이다. 따라서 이를 이용해서 layer를 짜고 `loss.backward()`를 하면 Pytorch가 `One of the variables needed for gradient computation has been modified by an inplace operation` 에러를 내뿜게 된다. 디버깅하기 힘드니까 조심하자.... Pytorch를 사용하면서 느끼는 건 잘 모르면 진짜 그냥 안전하게 짜는게 에러 안나고 베스트라는 것이다. 코드 길이 줄이겠다고 `+=` 썼다가 에러 잡느라 몇시간을 날렸다....

# 모델 코드가.... detach()...?

```python
class Transformer(nn.Module):
    def __init__(self, N = 3, hidden_dim = 256, num_head = 8, inner_dim = 512):
        super().__init__()
        self.encoder = Encoder(N, hidden_dim, num_head, inner_dim)
        self.decoder = Decoder(N, hidden_dim, num_head, inner_dim)

    def forward(self, enc_src, dec_src):
        # enc_src : (bs, seq_len)
        # dec_src : (bs, seq_len)

        # print(f'enc_src : {enc_src.shape}')
        # print(f'dec_src : {dec_src.shape}')

        enc_output = self.encoder(enc_src)
        output, logits = self.decoder(dec_src, enc_src, enc_output.detach())
        # logits = (bs, seq_len, VOCAB_SIZE)

        return output, logits
```

문제의 코드(17).... `enc_output.detach()`가 보이는가? 도대체 무슨 생각을 저 코드를 넣었던 건지 모르겠다....
저러면 당연히 encoder가 학습이 될리가 없는데 진짜 멍청함을 느끼는 순간이었다.

![training image](https://i.imgur.com/uWsUoFc.png)

파란색 곡선이 보이는가...? 아 어이없어... 저거 떼자마자 귀신같이 학습이 잘된다 ^^ 마스크 수정하러 가자

# Masking Function

막판까지 속썩였던 애들 중 한명이다.... 논문에서는 `query_len`과 `key_len`을 동일하게 놓고 진행한다. 그래서 처음에 `padding mask`와 `lookahead mask` 둘 다 `input (bs, seq_len)`을 받으면 `(bs, 1, seq_len, seq_len)`을 반환하도록 구현했다. (중간에 1 부분은 `num_head` broadcasting을 위한 것). Training 할때도 어차피 encoder와 seq_len과 decoder의 seq_len이 같기 때문에 문제가 되지 않았다.

문제는 Inference할 때의 `padding mask`였다. Inference할 때에는 decoder의 입력으로 `<SOS>` 토큰 하나만 들어가기 때문에, decoder의 입력은 1이다. 반대로 encoder는 문장 하나가 input으로 들어가기 때문에 일정한 seq_len을 가진다. Encoder-Decoder Self Attention 부분을 살펴보자. decoder의 입력이 `srcQ`가 되고, encoder의 출력이 `srcK`가 된다. 따라서 `query_len`과 `key_len`이 다른 상황을 마주하게 되는데, 이를 고려하지 못했던 것이다.

- batch size = 1
- dec_src의 seq_len = 2
- enc_src, enc_ouput의 seq_len = 10
- num_head = 8
- hidden_dim = 256을 고려하면 다음과 같다

![img](https://i.imgur.com/SOTgsaA.png)

아무튼 그래서

- `pading mask` : `(bs, 1, 1, k_len)`
- `lookahead mask` : `(bs, 1, k_len, k_len)`

이렇게 수정하는 것으로 원만한 합의를 볼 수 있었다.

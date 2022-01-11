---
title: "[PyTorch] Transformer 구현"
date: 2022-01-11
categories: PyTorch
tags: [transformer, pytorch]
toc: true
toc_sticky: true
---

# 계기

작년부터 Transformer가 성능이 그렇게 좋다더라... 이제 RNN이고 LSTM이고 다 필요없고 이미 NLP는 저걸로 천하통일이라더라... 하는 소식을 듣고 있었다. 중간에 강화학습 공부 좀 찍먹해보다가 다시 캐글에서 [PetFinder](https://www.kaggle.com/c/petfinder-pawpularity-score) 이미지 비전 태스크 공부로 선회한 상태였는데, 내가 주로 참고했던 노트북은 [[Pytorch] Hybrid Swin Transformer + CNN](https://www.kaggle.com/debarshichanda/pytorch-hybrid-swin-transformer-cnn) 이었다.

![Hybird Swin Transformer](https://i.imgur.com/Io5ZdFB.png)

이 노트북에서 사용된 Hybrid Swin Transformer는 CNN (EfficeintNet)을 이용해서 패치를 뽑아낸 다음, Swin Transformer의 입력으로 넣어주는 방식이다. 그런데 이때까지도 나는 Transformer의 이름만 들어봤지 아직 제대로는 공부해보지 못한 상태여서, Trnasformer라는 구조에 대해서 제대로 공부해봐야겠다고 생각했다.

# Youtube 강의 + 논문

나동빈이라는 분이 유튜브에 업로드한 [[딥러닝 기계 번역] Transformer: Attention Is All You Need (꼼꼼한 딥러닝 논문 리뷰와 코드 실습)](https://www.youtube.com/watch?v=AA621UofTUA&t=2711s) 유튜브 강의가 큰 도움이 되었다. 먼저 논문을 읽기 전에 이 유튜브를 들었는데, 덕분에 큰 그림을 잡을 수 있었다. 그리고 논문 읽어본 거, 처음이었는데 생각보다 별거 아니라는 생각이 들었다. 영어가 좀 빡세다는 점, 그리고 이게 뭐지 싶은 구조들이 참조 논문으로 우수수 튀어나오는 것만 제외하면... 사실 논문 하나 대충 읽는 건 별로 안걸리지만 만약 대학원생이 되어서 참조 논문까지 싹 다 이해하면서 논문을 읽고 그 분야를 섭렵해나가야 한다? 진짜 쉽지 않을 것 같다는 생각을 했다.

아무튼 논문까지 읽고, PyTorch 라이브러리에 대한 복습도 좀 할 겸 Transformer 자체를 코드로 구현해봐야겠다는 생각이 들었다. 사실 그때까지 이해가 좀 가지 않는 부분도 있어서 코드로 구현하면 확실히 이해가 될 것 같다는 생각도 들었고, 이 구조를 베이스로 한 모델들이 지금 NLP랑 Vision 분야를 휩쓸고 있는 상태에서, 베이스를 이해하지 않으면 쉽지 않겠다는 생각이 들어서이다.

# Transformer 구현

링크는 [Github](https://github.com/loggerJK/transformer-implementation)에 올려놓았다. 주요 코드는 아래와 같다.

## Mask Function

<details>
  <summary>코드 보기</summary>
  <div markdown="1">

```python
'''
Mask 행렬을 반환하는 Mask Function
Masking은 QK_T 중 srcK 의 seq_len을 중심으로 한다는 점을 알아두자!!

Input
- Tensor
    shape (bs, srcK seq_len)

Args
- Option
    If option is 'padding', function returns padding mask
    If option is 'lookahead', function returns lookahead mask

Output
- Tensor (option = 'padding' )
    shape (bs, 1, 1, srcK seq_len)


* shape 중 (1, 1) 부분은 broad casting을 위한 것이다.
'''


def makeMask(tensor, option: str) -> torch.Tensor:
    '''
    tensor (bs, seq_len)
    '''
    if option == 'padding':
        tmp = torch.full_like(tensor, fill_value=PAD_IDX).to(device)
        # tmp : (bs,seq_len)
        mask = (tensor != tmp).float()
        # mask : (bs, seq_len)
        mask = rearrange(mask, 'bs seq_len -> bs 1 1 seq_len ')

        # mask(bs, 1, seq_len,seq_len)

        '''
        Example of mask
        tensor([[
         [1., 1., 1., 1., 0., 0., 0., 0.]]])
        '''

    elif option == 'lookahead':
        # srcQ의 seq_len과 srcK의 seq_len이 동일하다고 가정한다
        # tensor : (bs, seq_len)

        padding_mask = makeMask(tensor, 'padding')
        padding_mask = repeat(
            padding_mask, 'bs 1 1 k_len -> bs 1 new k_len', new=padding_mask.shape[3])
        # padding_mask : (bs, 1, seq_len, seq_len)

        '''
        Example of padding_mask
        tensor([[
         [1., 1., 1., 1., 0., 0., 0., 0.]
         [1., 1., 1., 1., 0., 0., 0., 0.]
         [1., 1., 1., 1., 0., 0., 0., 0.]
         [1., 1., 1., 1., 0., 0., 0., 0.]
         [1., 1., 1., 1., 0., 0., 0., 0.]
         [1., 1., 1., 1., 0., 0., 0., 0.]
         [1., 1., 1., 1., 0., 0., 0., 0.]
         [1., 1., 1., 1., 0., 0., 0., 0.]]])
        '''
        mask = torch.ones_like(padding_mask)
        mask = torch.tril(mask)

        '''
        Example of 'mask'
        tensor([[
        [1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1.]]])
        '''

        mask = mask * padding_mask
        # ic(mask.shape)

        '''
        Example
        tensor([[
         [1., 0., 0., 0., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0., 0., 0., 0.],
         [1., 1., 1., 0., 0., 0., 0., 0.],
         [1., 1., 1., 1., 0., 0., 0., 0.],
         [1., 1., 1., 1., 0., 0., 0., 0.],
         [1., 1., 1., 1., 0., 0., 0., 0.],
         [1., 1., 1., 1., 0., 0., 0., 0.],
         [1., 1., 1., 1., 0., 0., 0., 0.]]])
        '''

    return mask
```

  </div>
</details>

## Multihead Self Attention

<details>
  <summary>코드 보기</summary>
  <div markdown="1">

```python
class Multiheadattention(nn.Module):
    def __init__(self, hidden_dim: int, num_head: int):
        super().__init__()

        # embedding_dim, d_model, 512 in paper
        self.hidden_dim = hidden_dim
        # 8 in paper
        self.num_head = num_head
        # head_dim, d_key, d_query, d_value, 64 in paper (= 512 / 8)
        self.head_dim = hidden_dim // num_head
        self.scale = torch.sqrt(torch.FloatTensor()).to(device)

        self.fcQ = nn.Linear(hidden_dim, hidden_dim)
        self.fcK = nn.Linear(hidden_dim, hidden_dim)
        self.fcV = nn.Linear(hidden_dim, hidden_dim)
        self.fcOut = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(0.1)


    def forward(self, srcQ, srcK, srcV, mask=None):

        ##### SCALED DOT PRODUCT ATTENTION ######

        # input : (bs, seq_len, hidden_dim)
        Q = self.fcQ(srcQ)
        K = self.fcK(srcK)
        V = self.fcV(srcV)

        Q = rearrange(
            Q, 'bs seq_len (num_head head_dim) -> bs num_head seq_len head_dim', num_head=self.num_head)
        K_T = rearrange(
            K, 'bs seq_len (num_head head_dim) -> bs num_head head_dim seq_len', num_head=self.num_head)
        V = rearrange(
            V, 'bs seq_len (num_head head_dim) -> bs num_head seq_len head_dim', num_head=self.num_head)

        attention_energy = torch.matmul(Q, K_T)
        # attention_energy : (bs, num_head, q_len, k_len)

        if mask is not None :
            '''
            mask.shape
            if padding : (bs, 1, 1, k_len)
            if lookahead : (bs, 1, q_len, k_len)
            '''
            attention_energy = torch.masked_fill(attention_energy, (mask == 0), -1e+4)

        attention_energy = torch.softmax(attention_energy, dim = -1)

        result = torch.matmul(self.dropout(attention_energy),V)
        # result (bs, num_head, seq_len, head_dim)

        ##### END OF SCALED DOT PRODUCT ATTENTION ######

        # CONCAT
        result = rearrange(result, 'bs num_head seq_len head_dim -> bs seq_len (num_head head_dim)')
        # result : (bs, seq_len, hidden_dim)

        # LINEAR

        result = self.fcOut(result)

        return result




```

  </div>
</details>

## Poistionwise Feedforward Network

<details>
  <summary>코드 보기</summary>
  <div markdown="1">

```python
class FFN(nn.Module):
    def __init__ (self, hidden_dim, inner_dim):
        super().__init__()

        # 512 in paper
        self.hidden_dim = hidden_dim
        # 2048 in paper
        self.inner_dim = inner_dim

        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.1)



    def forward(self, input):
        output = input
        output = self.fc1(output)
        output2 = self.relu(output)
        output2 = self.dropout(output)
        output3 = self.fc2(output2)

        return output3
```

  </div>
</details>

## Encoder Layer

<details>
  <summary>코드 보기</summary>
  <div markdown="1">

```python
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_head, inner_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim

        self.multiheadattention = Multiheadattention(hidden_dim, num_head)
        self.ffn = FFN(hidden_dim, inner_dim)
        self.layerNorm1 = nn.LayerNorm(hidden_dim)
        self.layerNorm2 = nn.LayerNorm(hidden_dim)


        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)


    def forward(self, input, mask = None):

        # input : (bs, seq_len, hidden_dim)

        # encoder attention
        # uses only padding mask
        output = self.multiheadattention(srcQ= input, srcK = input, srcV = input, mask = mask)
        output = self.dropout1(output)
        output = input + output
        output = self.layerNorm1(output)

        output_ = self.ffn(output)
        output_ = self.dropout2(output_)
        output = output + output_
        output = self.layerNorm2(output)

        # output : (bs, seq_len, hidden_dim)
        return output
```

  </div>
</details>

## Encoder

<details>
  <summary>코드 보기</summary>
  <div markdown="1">

```python
class Encoder(nn.Module):
    def __init__ (self, N, hidden_dim, num_head, inner_dim,max_length=100):
        super().__init__()

        # N : number of encoder layer repeated
        self.N = N
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim

        self.embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)
        self.enc_layers = nn.ModuleList([EncoderLayer(hidden_dim, num_head, inner_dim) for _ in range(N)])

        self.dropout = nn.Dropout(p=0.1)



    def forward(self, input):

        batch_size = input.shape[0]
        seq_len = input.shape[1]
        # input : (bs, seq_len)

        mask = makeMask(input, option='padding')

        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        # pos: [batch_size, src_len]

        # embedding layer
        output = self.dropout(self.embedding(input) + self.pos_embedding(pos))
        # output : (bs, seq_len, hidden_dim)


        # Positional Embedding
        # output = pos_embed(output)

        # Dropout
        output = self.dropout(output)

        # N encoder layer
        for layer in self.enc_layers:
            output = layer(output, mask)

        # output : (bs, seq_len, hidden_dim)

        return output




```

  </div>
</details>

## Decoder Layer

<details>
  <summary>코드 보기</summary>
  <div markdown="1">

```python
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_head, inner_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim

        self.multiheadattention1 = Multiheadattention(hidden_dim, num_head)
        self.layerNorm1 = nn.LayerNorm(hidden_dim)
        self.multiheadattention2 = Multiheadattention(hidden_dim, num_head)
        self.layerNorm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FFN(hidden_dim, inner_dim)
        self.layerNorm3 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)


    def forward(self, input, enc_output, paddingMask, lookaheadMask):
        # input : (bs, seq_len, hidden_dim)
        # enc_output : (bs, seq_len, hidden_dim)

        # first multiheadattention
        output = self.multiheadattention1(input, input, input, lookaheadMask)
        output = self.dropout1(output)
        output = output + input
        output = self.layerNorm1(output)


        # second multiheadattention
        output_ = self.multiheadattention2(output, enc_output, enc_output, paddingMask)
        output_ = self.dropout2(output_)
        output = output_ + output
        output = self.layerNorm2(output)



        # Feedforward Network
        output_ = self.ffn(output)
        output_ = self.dropout3(output_)
        output = output + output_
        output = self.layerNorm3(output)



        return output

```

  </div>
</details>

## Decoder

<details>
  <summary>코드 보기</summary>
  <div markdown="1">

```python
class Decoder(nn.Module):
    def __init__ (self, N, hidden_dim, num_head, inner_dim, max_length=100):
        super().__init__()

        # N : number of encoder layer repeated
        self.N = N
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim

        self.embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        self.dec_layers = nn.ModuleList([DecoderLayer(hidden_dim, num_head, inner_dim) for _ in range(N)])

        self.dropout = nn.Dropout(p=0.1)

        self.finalFc = nn.Linear(hidden_dim, VOCAB_SIZE)


    def forward(self, input, enc_src, enc_output):

        # input = dec_src : (bs, seq_len)
        # enc_src : (bs, seq_len)
        # enc_output : (bs, seq_len,hidden_dim)

        lookaheadMask = makeMask(input, option= 'lookahead')
        paddingMask = makeMask(enc_src, option = 'padding')

        # embedding layer
        output = self.embedding(input)
        # output = (bs, seq_len, hidden_dim)


        # Positional Embedding
        # output = pos_embed(output)

        # Dropout
        output = self.dropout(output)

        # N decoder layer
        for layer in self.dec_layers:
            output = layer(output, enc_output, paddingMask, lookaheadMask)
        # output : (bs, seq_len, hidden_dim)

        logits = self.finalFc(output)
        # logits : (bs, seq_len, VOCAB_SIZE)
        output = torch.softmax(logits, dim = -1)

        output = torch.argmax(output, dim = -1)
        # output : (bs, seq_len), dtype=int64



        return logits, output




```

  </div>
</details>

## Transformer

<details>
  <summary>코드 보기</summary>
  <div markdown="1">

```python
class Transformer(nn.Module):
    def __init__(self, N = 2, hidden_dim = 256, num_head = 8, inner_dim = 512):
        super().__init__()
        self.encoder = Encoder(N, hidden_dim, num_head, inner_dim)
        self.decoder = Decoder(N, hidden_dim, num_head, inner_dim)

    def forward(self, enc_src, dec_src):
        # enc_src : (bs, seq_len)
        # dec_src : (bs, seq_len)

        # print(f'enc_src : {enc_src.shape}')
        # print(f'dec_src : {dec_src.shape}')

        enc_output = self.encoder(enc_src)
        # enc_output : (bs, seq_len, hidden_dim)
        logits, output = self.decoder(dec_src, enc_src, enc_output)
        # logits = (bs, seq_len, VOCAB_SIZE)

        return logits, output


```

  </div>
</details>

# 후기

확실히 코드로 구현하니까 헷갈리던 부분들이 많이 정리가 된다. 모델 짜는 것 자체는 2~3일 정도밖에 걸리지 않았던 것 같다. 하지만 아직 PyTorch에 익숙하지 않아서 그런지 잔버그들이 좀 많았고, 이를 해결하는데 시행착오를 좀 겪어서 결과적으로는 약 일주일정도만에 완성한 것 같다. Transformer가 (내 생각엔) 다른 구조들보다는 그래도 간단한 편이어서 구현하기 괜찮았던 것 같다. 만약 Swin Transformer의 Shifted Window 같은 거 구현하라고 하면 어우 벌써부터 머리가 깨진다 ^^.....

시행착오는 [여기](https://loggerjk.github.io/deeplearning/Transformer%EB%A5%BC-%EA%B5%AC%ED%98%84%ED%95%98%EB%A9%B0-%EB%B0%B0%EC%9A%B4-%EA%B2%83%EB%93%A4/)에 정리해두었다. (생각보다 많다)

이후에는 BERT, GPT, Swin-T 등 Transformer 기반으로 만들어진 여러 모델들에 대해서 공부하고, 가끔은 구현도 해볼 예정이다. 참, Swin-T VER2 나온 거 같던데 이것도 공부해야지.

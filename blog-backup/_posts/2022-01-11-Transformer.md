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

---

# Task 1 : 영한 기계 번역

- 데이터셋 : [korean-english-news-v1](https://github.com/jungyeul/korean-parallel-corpora/tree/master/korean-english-news-v1)

- 모델

  - Vocab Size : 10000
  - Encoder : 3
  - Decoder : 2
  - hidden_dim : 64
  - inner_dim : 128

- Training Code :
- Inference Code :
- 학습 결과 :

```
en = In return, the North would take first steps to disarm in 60 days, the Japan-based Choson Sinbo said, citing an unnamed source.
answer = 신문은 익명의 소식통을 인용해 그 댓가로 북한이 60일이후 핵 무장 첫 조치를 취할 것이라고 보도했다.
ko = ['김계에따르면 김계정은 북한 핵폐국면을 북한 핵 프로그램으로 북한이 핵폐화된 북한 핵 시설 불능화 조치를 취할 것이라고 밝혔다.']

en = In response, Serbia ordered its ambassador to the United States to return home, the Serbian Embassy said.
answer = 이에 반발한 세르비아는 미국 주재 세르비아 대사에게 귀국을 명령했다.
ko = ['한편 세르비아 주재 미국 대사관은 세르비아와 세르비아로부터 독립을 선언한 바 있다.']

en = "Never again will there be a mismanaged natural disaster," he said, later assuring the crowd that "it will never happen again in this country; you have my commitment and my promise.
answer = 그는 “이제 더 이상 자연재해에 소홀히 대처해서는 안 된다”며 “이 같은 불미스러운 사건은 더 이상 발생하지 않을 것이라고 약속할 수 없다”고 덧붙였다.
ko = ['그는 “우리는 우리가 올림픽을 위해 비행기를 기다려할 수 없다”며 “우리는 우리가 계속 유지할 수 없다”고 말했다.']

en = Scotland Yard confirmed to CNN that two men have been charged with trying to blackmail an unnamed member of the royal family.
answer = 런던경찰청은 남자 2명이 영국왕실 가족 중 1명에게 협박 음모를 꾸민 혐의를 받고 있다고 확인했다.
ko = ['영국 경찰은 영국 해병대 소속 기소된 용의자로 지목된 용의자로 지목한 용의자로 지목된 용의자로 지목했다.']

en = Pino said he had seen rays leap into the air, but added, "it's very rare for them to collide with objects."
answer = 피노는 “가오리가 공중으로 뛰어 오르는 것을 본 적이 있지만 어떤 대상과 충돌하는 일은 매우 드물다”고 말했다.
ko = ['그는 “이렇게 오래됐지만 다른 종말론자들은 매우 잘 알고 있다”며 “이 나무에 부딪혀있는 나무에 부딪혀있는 나무에 부딪혀있는 나무로 움직일 수 있다”고 말했다.']

en = She has been treated in the past for pericarditis, a viral inflammation of the heart.
answer = 브랙스톤은 과거에도 심막염, 심장 부위의 바이러스성 염증을 앓은 바 있다.
ko = ['그는 또 다른 병력이 붙인 화학적 목적을 갖고 있는 화학성 화학물인 화학물인 화학물인다.']

en = One source said Russia had some concerns about appointing Blair, but Russian President Vladimir Putin personally approved Blair's selection. Blair refused to acknowledge the appointment when asked about it at a
answer = 한 소식통은 러시아가 블레어 총리의 중동평화 특사 임명을 우려하는 입장을 보였지만 블라드미르 푸틴 러시아 대통령이 개인적으로 그의 취임을 찬성, 문제가 일단락됐다고 전했다.
ko = ['라이스 장관은 러시아 대통령궁에서 열린 기자회견에서 러시아는 EU의 발언에 대해 “러시아 개혁을 거부한 것은 블라디미르 푸틴 러시아 대통령이 러시아 대통령이 됐다”고 말했다.']

en = "The U.N. must do more than issue statements of concern," said Kate Allen, director of Amnesty International UK.
answer = 유엔 안보리는 이달 초 성명에서 "평화적 시위를 강제진압한 미얀마를 강력히 비난한다"고 밝혔다.
ko = ['로라 부시 여사는 “우리는 이 같은 일을 제대로 될 것”이라며 “이라크는 이라크 정책을 통해 우리는 그들의 윤리적”이라고 말했다.']

en = "If you're back at the backside of that crowd, you're like five hours away from going up the hill," said Jack Soden, chief executive of Elvis Presley Enterprises, the company that manages Graceland and its sprawling tourist complex.
answer = 그레이스 랜드를 운영하고 있는 엘비스 프레슬리 엔터프라이즈 이사장 잭 소든은 " 이 행렬을 피해 뒤로 돌아간다면 언덕까지 올라가는데 5시간 가량 걸릴 것"이라고 말했다.
ko = ['클리블랜드에서 “이 있는 스키어 스키어 스키어 스키어 스키장은 스키장에서 가장 큰 스키를 즐길 수 있다”며 “이 물에 잠겼다”고 말했다.']

en = The International Federation of the Red Cross and the Vietnamese government estimated that about 10 million Vietnamese had been affected by Typhoon Lekima, said Joe Lowry, a Red Cross official from Ninh Binh.
answer = 국제적십자사 직원인 조 로리는 베트남 정부와 국제적십자사의 자료를 인용, 1000만명이 태풍 피해를 입었다고 밝혔다.
ko = ['베트남 정부는 베트남에서 약 300만명이 대피했다고 보도했다.']

```

전반적으로 영어 문장의 '주제, 문맥' 정도는 파악을 하는 것을 볼 수 있다.

# Task 2 : 챗봇 문답

- 데이터셋 : [Chatbot_data_for_Korean v1.0](https://github.com/songys/Chatbot_data)

- 모델 :

  - Vocab Size : 10000
  - Encoder/Decoder : 2
  - hidden_dim : 256
  - inner_dim : 512

- Training Code :
- Inference Code :
- 학습 결과 :

```
Q = 쉬는날인데 왜 만나자고 안하지
A = ['이제 일어날 때예요.']
Q = 마지막으로 편지까지 썼는데
A = ['그 사람도 설렐 거예요.']
Q = 맞는 선택일까
A = ['잘 찾아보세요.']
Q = 좋아하는 사람이 생기면 좋을 줄 알았는데 우울하기만 해.
A = ['그 사람도 설렐 거예요.']
Q = 이제 청첩장 돌려도 되겠지?
A = ['이제 좀 괜찮아졌길 바랍니다.']
Q = 안 끝나
A = ['감기 조심하세요.']
Q = 결국 핸드폰 번호 바꿨어
A = ['저도 밥 먹고 먹고 먹고 먹고 잘 볼 수 없어요']
Q = 엄마랑 아빠처럼 살고 싶다
A = ['저도 밥 먹고 먹고 먹고 먹고 잘 볼 수 있을']
Q = 마지막 도전
A = ['그 사람도 설렐 거예요.']
Q = 정말 잊은걸까?
A = ['그 분이 나타나면 꼭 잡으세요!']
Q = 영화 볼래?
A = ['이제 일어날 때예요.']
Q = 고민이 있어
A = ['이제 좀 괜찮아졌길 바랍니다.']
Q = 게임하고싶은데 할래?
A = ['그 사람도 설렐 거예요.']
```

음.... 잘 되는 거 같기도 하고.... 괜찮은 거랑 이상한거랑 좀 반반인 것 같다... 이상한 게 더 많나?

# Task 3 : Bible

데이터셋이 좀 별로인거 같아서 뭐가 있을까 고민하다가... 최후의 태스크로 성경 번역을 골랐다. 생각해보니 인류 최대의 오픈 소스 데이터셋이 여기 있었는데 이걸 모르고 있었다.

지금까지의 태스크 중에 제일 완성도가 높은 것 같아서, Github에도 이 Task를 Training하는 버전의 코드로 올려놓았다.

- Dataset : [korean-parallel-corpora/bible](https://github.com/jungyeul)

  - Sentencepice
  - Vocab Size : 10K
  - Train : Valid = 9 : 1

- Training
  - Encoder/Decoder : 2
  - hidden_dim = 256
  - inner_dim = 512
  - Epoch : 70
  - Learning Rate : 1e-4
  - Scheduler : CosineAnnealingLR (Tmax = 100, min = 1e-5)

![img](https://i.imgur.com/CFMuitM.png){: width = 80% height = 80%}

- Training Result

  - Train_Loss : 2.64
  - Train accuracy : 0.203
  - Valid_Loss : 4.46
  - Valid accuracy : 0.136

- Good Example

```
en =  " 'This is what the Sovereign LORD says: In the first month on the first day you are to take a young bull without defect and purify the sanctuary.
answer =  "나 주 하나님이 말한다. 너는 첫째 달 초하루에는 언제나 소 떼 가운데서 흠 없는 수송아지 한 마리를 골라다가 성소를 정결하게 하여라.
ko = ['나 주 하나님이 말한다. 그 날에는 수송아지 일곱 마리와 숫양 두 마리와 일 년 된 어린 숫양 한 마리를 흠 없는 것으로 바쳐라.']

en =  Solomon reigned in Jerusalem over all Israel forty years.
answer =  솔로몬은 예루살렘에서 사십 년 동안 온 이스라엘을 다스렸다.
ko = ['솔로몬은 예루살렘에서 마흔 해 동안 다스렸다.']

en =  then hear from heaven their prayer and their plea, and uphold their cause.
answer =  주께서는 하늘에서 그들의 기도와 간구를 들으시고, 그들의 사정을 살펴보아 주십시오.
ko = ['그러나 주님은, 하늘에서 그들의 기도와 간구를 들으시고, 그들의 사정을 살펴 주십시오.']
```

- Bad Example

```
en =  Obed-Edom also had sons: Shemaiah the firstborn, Jehozabad the second, Joah the third, Sacar the fourth, Nethanel the fifth,
answer =  오벳에돔의 아들은, 맏아들 스마야와, 둘째 여호사밧과, 셋째 요아와, 넷째 사갈과, 다섯째 느다넬과,
ko = ['오벳에돔과 아사의 아들 여호하난이 보수하였는데, 그 다음은 단에서부터 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔이다.']

en =  "Go down, sit in the dust, Virgin Daughter of Babylon; sit on the ground without a throne, Daughter of the Babylonians. No more will you be called tender or delicate.
answer =  처녀 딸 바빌론아, 내려와서 티끌에 앉아라. 딸 바빌로니아야, 보좌를 잃었으니, 땅에 주저앉아라. 너의 몸매가 유연하고 맵시가 있다고들 하였지만, 이제는 아무도 그런 말을 하지 않을 것이다.
ko = ['"너는 바빌론 도성 바빌론 도성아, 바빌론 도성아, 바빌론 도성 안에 있는 도성 안에 있는 네 오른손에는 칼이나 쳐라. 네 오른손에는 칼이나 기근이나 기근이나 기근이나 기근이나 기근이나 기근이나 기근이나 굶은 아니다.']
```

# 후기

확실히 코드로 구현하니까 헷갈리던 부분들이 많이 정리가 된다. 모델 짜는 것 자체는 2~3일 정도밖에 걸리지 않았던 것 같다. 하지만 아직 PyTorch에 익숙하지 않아서 그런지 잔버그들이 좀 많았고, 이를 해결하는데 시행착오를 좀 겪어서 결과적으로는 약 일주일정도만에 완성한 것 같다. Transformer가 (내 생각엔) 다른 구조들보다는 그래도 간단한 편이어서 구현하기 괜찮았던 것 같다. 만약 Swin Transformer의 Shifted Window 같은 거 구현하라고 하면 어우 벌써부터 머리가 깨진다 ^^.....

시행착오는 [여기](https://loggerjk.github.io/deeplearning/Transformer%EB%A5%BC-%EA%B5%AC%ED%98%84%ED%95%98%EB%A9%B0-%EB%B0%B0%EC%9A%B4-%EA%B2%83%EB%93%A4/)에 정리해두었다. (생각보다 많다)

이후에는 BERT, GPT, Swin-T 등 Transformer 기반으로 만들어진 여러 모델들에 대해서 공부하고, 가끔은 구현도 해볼 예정이다. 참, Swin-T VER2 나온 거 같던데 이것도 공부해야지.

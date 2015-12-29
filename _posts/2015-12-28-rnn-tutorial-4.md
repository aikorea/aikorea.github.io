---
layout: post
comments: true
title:  "RNN Tutorial Part 4 - GRU/LSTM RNN 구조를 Python과 Theano를 이용하여 구현하기"
date:   2015-12-28
mathjax: true
---

> [WildML](http://www.wildml.com/)의 네 번째 (마지막!) RNN 튜토리얼입니다.
>
> 이전 번역 포스트들과 마찬가지로 [영문 버전](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)을 거의 그대로 옮겨왔고, 번역에 이상한 점을 발견하셨거나 질문이 있으시다면 댓글로 달아주세요!

RNN 튜토리얼 파트 4입니다! [이 포스트에 대한 코드](https://github.com/dennybritz/rnn-tutorial-gru-lstm) 역시 Github에 올라와 있습니다. RNN 튜토리얼 시리즈의 마지막 파트이며, 이전 파트들은:

1. [Recurrent Neural Network (RNN) Tutorial - Part 1](/blog/rnn-tutorial-1)
2. [RNN Tutorial Part 2 - Python, NumPy와 Theano로 RNN 구현하기](/blog/rnn-tutorial-2)
3. [RNN Tutorial Part 3 - BPTT와 Vanishing Gradient 문제](/blog/rnn-tutorial-3)

이 포스트에서는 LSTM (Long Short Term Memory)과 GRU (Gated Recurrent Unit)이라는 모듈에 대해 알아볼 것입니다. LSTM은 [1997년에 스위스의 Sepp Hochreiter와 ürgen Schmidhuber에 의해 처음 제안](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)되었고, 현재 자연어처리 분야에 활용되는 딥러닝 기법 중 가장 널리 사용되고 있는 모델 중의 하나입니다. GRU는 2014년에 처음 사용되었는데 (뉴욕대의 [조경현 교수님께서 처음 제안](http://arxiv.org/pdf/1406.1078.pdf)), 대부분 LSTM과 비슷한 성질을 갖지만 더 간단한 구조를 갖고 있습니다. 여기서는 먼저 LSTM에 대해 살펴본 다음, GRU가 어떻게 LSTM과 다른지 살펴볼 것입니다.


## LSTM 네트워크

[파트 3](/blog/rnn-tutorial-3)에서는 vanishing gradient 문제가 왜 기본 RNN 구조에서 긴 시퀀스를 학습하기 힘든지 살펴보았습니다. LSTM은 이 문제를 몇 가지 게이트를 추가함으로써 해결하도록 디자인 되었습니다. 이것이 무슨 말인지 이해하기 위해서, LSTM이 hidden state \\( s\_t \\)를 어떻게 계산하는지 수식으로 살펴보면 다음과 같습니다. ( \\( \circ \\) 는 element-wise 곱을 의미합니다)

$$
\begin{align}  
i & = \sigma(x\_tU^i + s\_{t-1} W^i) \\\\  
f & = \sigma(x\_t U^f +s\_{t-1} W^f) \\\\  
o & = \sigma(x\_t U^o + s\_{t-1} W^o) \\\\  
g & = \tanh(x\_t U^g + s\_{t-1}W^g) \\\\  
c\_t & = c\_{t-1} \circ f + g \circ i \\\\  
s\_t & = \tanh(c\_t) \circ o  
\end{align}
$$

수식만 보면 상당히 복잡해 보이지만, 실제로는 그렇게 어렵지 않습니다. 먼저, LSTM 레이어(층)는 그냥 hidden state를 계산하기 위한 한 가지 다른 방식이라고 생각하면 됩니다. 이전에는 hidden state를 계산할 때 \\( s\_t = \tanh(Ux\_t + Ws\_{t-1}) \\)처럼 계산했는데, 여기서 \\( x\_t \\)는 현재 시간 스텝 t에서의 입력이고 \\( s\_{t-1} \\)는 이전 시간 스텝의 hidden state, 그리고 새로운 hidden state의 출력값이 \\( s\_t \\)였습니다. LSTM 모듈은 같은 작업을 조금 다르게 수행할 뿐입니다! **이것이 큰 그림을 이해하기 위한 핵심 포인트입니다.** 결국에는 LSTM (또는 GRU) 모듈을 블랙박스로 취급해서, 현재의 입력 벡터와 이전 시간 스텝의 hidden state를 받아 다음 hidden state를 알아서 잘 계산한다고 생각해도 될 것입니다.

![lstm gru 그림](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/10/gru-lstm.png)

이 점을 기억하면서, LSTM이 hidden state를 어떻게 계산하는지에 대한 감을 잡아보도록 하겠습니다. 이에 대한 자세한 내용으로는 구글의 [Chris Olah의 훌륭한 포스트](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)가 이미 존재하기 때문에, 중복되는 포스팅을 피하기 위해 여기서는 내용을 간단히 요약하고 넘어가도록 하겠습니다. 보다 깊은 인사이트를 얻기 위해서, 또는 시각화를 통해 이해하기 위해서는 Chris Olah의 포스트를 꼭 읽어보기 바랍니다. 요약하자면:

- i, f, o 는 각각 입력, 까먹음, 출력 게이트라고 부릅니다. 각 게이트의 수식은 동일한 형태를 띄고 있고, 파라미터 행렬만 다른 것을 확인할 수 있습니다. 이들이 게이트라고 부르는 이유는, sigmoid 함수가 이 벡터들의 값을 0에서 1 사이로 제한시키고 이를 다른 벡터와 elementwise 곱을 취한다면 그 다른 벡터값의 얼마만큼을 통과시킬지를 정해주는 것과 같기 때문입니다. 입력 게이트는 새 hidden state 값을 계산하는데 있어서 입력 벡터값을 얼만큼 사용할 지를 정해줍니다. 까먹음 게이트는 이전 state 값을 얼만큼 기억하고 있을지를 정해주고, 출력 게이트는 현재의 내부 state 값의 얼만큼을 LSTM 모듈의 바깥쪽에서 (더 깊은 레이어나 이후 시간 스텝에서) 볼 수 있을지를 정해줍니다. 모든 게이트들은 \\( d\_s \\)로 hidden state와 같은 차원을 갖게 됩니다.
- g는 현재 입력과 이전 hidden state의 값을 기반으로 계산된 현재 hidden state 값의 "후보"라고 할 수 있습니다. RNN 기본형 모델에서 본 것과 동일한 수식으로 계산되는데, 파라미터 행렬의 이름만 U와 W가 \\( U^g \\)와 \\( W^g \\)로 바뀌었습니다. 그러나 이전처럼 g를 바로 새 hidden state로 정하는 대신, 여기서는 입력 게이트를 사용하여 일부만 사용합니다.
- \\( c\_t \\)는 LSTM 유닛(모듈)의 내부 메모리입니다. 이것은 이전에 저장된 메모리인 \\( c\_{t-1} \\)과 까먹음 게이트의 곱, 그리고 새로 계산된 hidden state g와 입력 게이트의 곱을 합친 형태로 계산됩니다. 따라서, 간단히 말하면 이전 메모리와 현재의 새 입력을 어떻게 합칠까에 대한 부분입니다. 이전의 메모리를 전부 무시하도록 (까먹음 게이트 값이 전부 0) 정하거나 새로운 입력값을 통째로 무시하도록 (입력 게이트 값이 전부 0) 할 수도 있지만, 보통은 이 양 극단보다는 중간의 좋은 지점을 찾도록 합니다.
- 메모리값 \\( c\_t \\)가 주어지면, 메모리와 출력 게이트의 곱으로 최종적으로 출력 hidden state \\( s\_t \\)가 계산됩니다. 모든 내부 메모리 값이 네트워크의 다른 유닛들에서 활용할 필요는 없을 수도 있기 때문에 출력 게이트를 통과시키는 것입니다.

![LSTM Gating Diagram](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/10/Screen-Shot-2015-10-23-at-10.00.55-AM.png)
LSTM Gating. Chung, Junyoung, et al. “Empirical evaluation of gated recurrent neural networks on sequence modeling.” (2014)

직관적으로 보면, 기본 RNN 모델은 LSTM의 특수 케이스로 생각할 수도 있습니다. 입력 게이트를 전부 1로 두고, 까먹음 게이트를 전부 0으로 (이전 메모리는 무조건 까먹는 것으로) 하고, 출력 게이트를 전부 1로 설정한다면 (메모리 값 전부를 보여줌) RNN 모델 기본형과 거의 같습니다. 출력값을 특정 범위 내로 압축시키는 tanh만 추가된 형태일 것입니다. LSTM의 이 게이팅 메커니즘이 모델에서 긴 시퀀스를 확실히 잘 기억하도록 (long-term 의존도를 잘 고려하도록) 하는 핵심 기법입니다. 게이트들의 파라미터를 잘 학습한다면, 네트워크에서 메모리 값들을 어떻게 기억하고 잊어버릴지를 학습하게 됩니다.

위에서 살펴본 기본 LSTM 구조 외에도 여러 변형 구조가 존재합니다. 자주 사용되는 변형 중 하나는, 게이트 값들이 이전 hidden state인 \\( s\_{t-1} \\)에만 의존하지 않고 이전 내부 메모리인 \\( c\_{t-1} \\)에도 의존하도록 peephole 연결을 만드는 것입니다. (게이트 값을 계산하는 수식에 추가적으로 항 하나를 더해줌으로써). 이외에도 더 많은 변형이 존재하는데, [LSTM: A Search Space Odyssey](http://arxiv.org/pdf/1503.04069.pdf)에서 여러가지 LSTM 구조의 장단점에 대해 실험적으로 폭넓게 평가되어 있습니다.


## GRU

GRU에 대한 기본적인 아이디어는 LSTM과 매우 비슷하고, 이는 수식에서도 확인할 수 있습니다.

$$
\begin{aligned}  
z &= \sigma(x\_tU^z + s\_{t-1} W^z) \\\\  
r &= \sigma(x\_t U^r +s\_{t-1} W^r) \\\\  
h &= tanh(x\_t U^h + (s\_{t-1} \circ r) W^h) \\\\  
s_t &= (1 - z) \circ h + z \circ s\_{t-1}  
\end{aligned}
$$

GRU는 리셋 게이트 r과 업데이트 게이트 z로, 총 두 가지 게이트가 있습니다. 게이트 이름에서 알 수 있듯이, 리셋 게이트는 새로운 입력을 이전 메모리와 어떻게 합칠지를 정해주고, 업데이트 게이트는 이전 메모리를 얼만큼 기억할지를 정해줍니다. 리셋 게이트 값을 전부 1로 정해주고 업데이트 게이트를 전부 0으로 정한다면, 기본 RNN 구조가 될 것입니다. 게이팅 메커니즘을 통해 긴 시퀀스를 잘 기억하도록 해준다는 점에서는 LSTM과 기본 아이디어가 같지만, 몇 가지 차이점이 있습니다.

- GRU는 게이트가 2개이고, LSTM은 3개입니다.
- GRU는 내부 메모리 값 ( \\( c\_t \\) )이 외부에서 보게되는 hidden state 값과 다르지 않습니다. LSTM에 있는 출력 게이트가 없기 때문입니다.
- 입력 게이트와 까먹음 게이트가 업데이트 게이트 z로 합쳐졌고, 리셋 게이트 r은 이전 hidden state 값에 바로 적용됩니다. 따라서, LSTM의 까먹음 게이트의 역할이 r과 z 둘 다에 나눠졌다고 생각할 수 있습니다.
- 출력값을 계산할 때 추가적인 비선형 함수를 적용하지 않습니다.

![GRU Gating Diagram](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/10/Screen-Shot-2015-10-23-at-10.36.51-AM.png)
caption: GRU Gating. Chung, Junyoung, et al. “Empirical evaluation of gated recurrent neural networks on sequence modeling.” (2014)


## GRU vs LSTM

여기까지 두 모델을 다 살펴보았는데, vanishing gradient 문제를 해결하기 위해서 어떤 모델을 사용하는 것이 좋을지 궁금할 것입니다. GRU는 상당히 최근 기술이고 (2014), 아직 그 장단점이 확실히 밝혀지지는 않았습니다. [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555)과 [An Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf) 두 논문에서의 실험적인 결과에 따르면, 확실한 승자는 없는 것으로 보입니다. 많은 문제들에서 두 모델 모두 좋은 성능을 보여주고 있고, 레이어 사이즈같은 파라미터 튜닝을 잘 하는 것이 모델을 고르는 것보다 더 중요할 것입니다. GRU는 파라미터 수가 적어서 (U와 W가 더 작다) 학습 시간이 더 짧게 걸리고 보다 적은 데이터로도 학습이 가능할 수 있겠지만, 반대로 말하면 충분한 수의 데이터가 있을 경우에는 LSTM의 우수한 모델링 파워가 더 좋은 결과를 보여줄 수도 있을 것입니다.


## 실제 구현

[파트 2](/blog/rnn-tutorial-2)에서 구현했던 것으로 돌아가서 RNN 대신에 GRU 유닛을 사용하도록 해 볼 것입니다. LSTM 대신 GRU를 선택한 것에 있어서 특별한 이유는 없는데, 구현 방법은 거의 동일하기 때문에 이 파트를 잘 이해한다면 GRU에서 LSTM으로 바꾸는 것은 스스로도 손쉽게 수식만 조금 바꾸면 가능할 것입니다.

코드는 이전의 Theano 기반 구현을 참고할 것입니다. GRU (또는 LSTM) 레이어는 결국 hidden state를 다른 방식으로 계산하는 것뿐이라는 점을 기억한다면, forward propagation 함수에서 hidden state를 계산하는 식만 바꿔주는 것으로 쉽게 구현할 수 있습니다.

```python
def forward_prop_step(x_t, s_t1_prev):
      # This is how we calculated the hidden state in a simple RNN. No longer!
      # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

      # Get the word vector
      x_e = E[:,x_t]

      # GRU Layer
      z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
      r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
      c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
      s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

      # Final output calculation
      # Theano's softmax returns a matrix with one row, we only need the row
      o_t = T.nnet.softmax(V.dot(s_t1) + c)[0]

      return [o_t, s_t1]
```
우리 구현에서는 bias로 b, c도 추가하였습니다. 수식에서는 보통 간단히 나타내기 위해 bias 항들을 생략하는 것이 보편적입니다 (bias trick 등을 통해 생략 가능). 또한, 이전의 파트 2에서의 코드에서 달라진 점은 파라미터 행렬 U와 W의 사이즈가 달라졌기 때문에 초기값 설정을 다시 제대로 해주어야 합니다. 여기서는 초기값 설정 코드는 생략했지만, Github에서 확인하실 수 있습니다. 위 코드에서 단어 임베딩 레이어로 E도 새로 추가했는데, 이는 아래에서 더 다루도록 하겠습니다.

코드를 보면 별로 복잡하지 않아 보이는데, 그럼 gradient는 어디서 계산될까요? 이전처럼 E, W, U, b와 c에 대한 gradient를 연쇄법칙을 활용해서 손으로 직접 유도할 수도 있지만, 많은 사람들은 Theano처럼 자동으로 미분을 해주는 라이브러리를 활용합니다. 만약 직접 미분값을 계산해야될 상황이 닥친다면, 여러 유닛에 대해 연쇄법칙을 이용해 자동으로 미분을 해주도록 하는 코드를 라이브러리 형태로 직접 구현하여 사용하는 것이 편할 것입니다. 여기서는 아래처럼 Theano가 대신 미분을 해주도록 활용합니다:

```python
# Gradients using Theano
dE = T.grad(cost, E)
dU = T.grad(cost, U)
dW = T.grad(cost, W)
db = T.grad(cost, b)
dV = T.grad(cost, V)
dc = T.grad(cost, c)
```

여기까지 구현의 대부분이 끝났습니다. 그러나 더 좋은 성능을 얻기 위해서는 현재 구현에서 추가로 시도해볼 몇 가지 트릭들이 있습니다.

### Rmsprop을 활용한 파라미터 업데이트

[파트 2](/blog/rnn-tutorial-2)에서 우리는 가장 간단한 형태의 Stochastic Gradient Descent (SGD)를 사용하여 파라미터 값들을 업데이트 했었습니다. 그러나 이것은 그다지 좋은 방법이 아닙니다. Learning rate을 낮게 잡아준다면 SGD는 좋은 해를 찾는 방향으로 업데이트 해주는 것이 보장되어 있지만, 실제로는 너무 오래 걸릴 것입니다. 따라서 몇 가지 개선된 알고리즘들이 존재하는데, [(Nesterov) Momentum 방법](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf), [AdaGrad](http://www.magicbroom.info/Papers/DuchiHaSi10.pdf), [AdaDelta](http://arxiv.org/abs/1212.5701), 그리고 [rmsprop](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) 등이 있습니다. [이 포스트](http://cs231n.github.io/neural-networks-3/#update)에서 다양한 알고리즘에 대한 설명을 찾아볼 수 있습니다. 이후 포스트에서 직접 각각의 방법의 구현 방법에 대해 작성하는 것도 계획중입니다. 여기서는 rmsprop을 구현해 보도록 하겠습니다. Rmsprop의 기본적인 아이디어는, 이전 gradient들의 합에 따라 ------공부 필요 ------- 파라미터별로 learning rate을 조정하는 것입니다. 직관적으로 이해해보면, 자주 등장하는 특징(feature)들은 작은 learning rate를 갖게 되고 (gradient들의 합이 작기 때문에), 드문드문 등장하는 특징들은 큰 learning rate를 갖게 됩니다.

구현은 상당히 간단합니다. 각 파라미터마다 캐시 변수를 두고, gradient descent가 진행될 때 아래와 같이 파라미터와 캐시를 업데이트하면 됩니다 (W에 대한 예시):

```python
cacheW = decay * cacheW + (1 - decay) * dW ** 2
W = W - learning_rate * dW / np.sqrt(cacheW + 1e-6)
```

'decay'는 보통 0.9나 0.95로 두고, 1e-6은 0으로 나눠지는 것을 방지하기 위해 더해졌습니다.

### 임베딩 레이어 추가하기

The embedding matrix is really just a lookup table – the ith column vector corresponds to the ith word in our vocabulary. By updating the matrix E we are learning word vectors ourselves, but they are very specific to our task (and data set) and not as general as those that you can download, which are trained on millions or billions of documents.

[Word2vec](https://code.google.com/p/word2vec/)이나 [GloVe](http://nlp.stanford.edu/projects/glove/)와 같은 단어 임베딩을 추가하는 것은 모델의 성능을 향상시키기 위해 자주 사용되는 방법입니다. 각 단어의 one-hot 벡터 표현법과 달리 word2vec이나 GloVe에서 학습된 낮은 차원의 벡터 표현은 그 단어의 의미 정보를 담게 됩니다. 즉, 비슷한 단어는 비슷한 벡터 값을 갖게 됩니다. 이것들을 사용하는 것은 pre-training (딥러닝에서 한번에 학습이 어렵기 때문에 단계별로 네트워크의 일부분을 미리 학습해 두는 작업)처럼 생각할 수 있습니다. 간단히 말하자면, 이 과정을 통해 네트워크가 언어에 대한 정보를 (미리 어느정도는 학습되어 있어서) 학습해야 될 부분이 줄어들게 되는 것입니다. Pre-train 된 벡터들은 학습할 데이터가 많지 않을 때 특히 유용한데, 네트워크가 사전에 보지 못한 단어들에 대해서도 일반화가 가능해지기 때문입니다. 여기서는 pre-train 된 단어 벡터를 사용하지는 않았지만, 임베딩 레이어 (코드의 행렬 E)를 추가함으로써 미리 학습된 단어 벡터를 넣는 것이 쉽게 해 두었습니다. 임베딩 행렬은

### 두 번째 GRU 레이어 추가하기

Adding a second layer to our network allows our model to capture higher-level interactions. You could add additional layers, but I didn’t try that for this experiment. You’ll likely see diminishing returns after 2-3 layers and unless you have a huge amount of data (which we don’t) more layers are unlikely to make a big difference and may lead to overfitting.
네트워크에 두 번째 레이어를 추가하는 것은 모델이 더 고차원적인 정보를 담을 수 있게 해줍니다. 여기서는

![2 Layer GRU/LSTM Unit](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/10/gru-lstm-2-layer.png)

Adding a second layer to our network is straightforward, we (again) only need to modify the forward propagation calculation and initialization function.

```python
# GRU Layer 1
z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

# GRU Layer 2
z_t2 = T.nnet.hard_sigmoid(U[3].dot(s_t1) + W[3].dot(s_t2_prev) + b[3])
r_t2 = T.nnet.hard_sigmoid(U[4].dot(s_t1) + W[4].dot(s_t2_prev) + b[4])
c_t2 = T.tanh(U[5].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + b[5])
s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev
```

[The full code for the GRU network is available here.](https://github.com/dennybritz/rnn-tutorial-gru-lstm/blob/master/gru_theano.py)

### A note on performance

I’ve gotten questions about this in the past, so I want to clarify that the code I showed here isn’t very efficient. It’s optimized for clarity and was primarily written for educational purposes. It’s probably good enough to play around with the model, but you should not use it in production or expect to train on a large dataset with it. There are many tricks to optimize RNN performance, but the perhaps most important one would be to batch together your updates. Instead of learning from one sentence at a time, you want to group sentences of the same length (or even pad all sentences to have the same length) and then perform large matrix multiplications and sum up gradients for the whole batch. That’s because such large matrix multiplications are efficiently handled by a GPU. By not doing this we can get little speed-up from using a GPU and training can be extremely slow.

So, if you want to train a large model I highly recommended using one of the existing Deep Learning libraries that are optimized for performance. A model that would take days/weeks to train with the above code will only take a few hours with these libraries. I personally like Keras, which is quite simple to use and comes with good examples for RNNs.


## 결과

To spare you the pain of training a model over many days I trained a model very similar to that in part 2. I used a vocabulary size of 8000, mapped words into 48-dimensional vectors, and used two 128-dimensional GRU layers. The iPython notebook contains code to load the model so you can play with it, modify it, and use it to generate text.

Here are a few good examples of the network output (capitalization added by me).

- I am a bot , and this action was performed automatically .
- I enforce myself ridiculously well enough to just youtube.
- I’ve got a good rhythm going !
- There is no problem here, but at least still wave !
- It depends on how plausible my judgement is .
- ( with the constitution which makes it impossible )

It is interesting to look at the semantic dependencies of these sentences over multiple time steps. For example, bot and automatically are clearly related, as are the opening and closing brackets. Our network was able to learn that, pretty cool!

That’s it for now. I hope you had fun and please leave questions/feedback in the comments!

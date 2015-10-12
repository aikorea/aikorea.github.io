---
layout: post
comments: true
title:  "RNN Tutorial Part 3 - BPTT와 Vanishing Gradient 문제"
date:   2015-10-10
mathjax: true
---

> [WildML](http://www.wildml.com/)의 세 번째 RNN 튜토리얼입니다. RNN 모델을 학습하는데 사용되는 핵심 알고리즘은 Backpropagation Through Time (BPTT)와, 기본 RNN 모델에서 발생하는 vanishing gradient 문제에 대해 조금 더 심도있게 다뤘습니다.
>
> 이전 번역 포스트들과 마찬가지로 [영문 버전](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)을 거의 그대로 옮겨왔고, 번역에 이상한 점을 발견하셨거나 질문이 있으시다면 댓글로 달아주세요!

[Recurrent Neural Network 튜토리얼](http://aikorea.org/blog/rnn-tutorial-1/)의 세 번째 파트입니다.

[이전 파트](http://aikorea.org/blog/rnn-tutorial-2/)에서는 RNN을 아예 처음부터 구현해 보았지만, BPTT 알고리즘이 어떻게 gradient를 계산하는지에 대해 깊숙히 들어가지는 않았었다. 이번 파트에서는 BPTT가 무엇인지, 기존의 backpropagation 알고리즘과의 차이점이 어떤게 있는지 살펴볼 것이다. 그리고 자연어처리(와 여러 다른) 분야에서 현재 가장 인기있는 LSTM과 GRU 구조를 필요하게 한 *vanishing gradient 문제* 에 대해 이해해볼 것이다. Vanishing gradient 문제는 [1991년에 Sepp Hochreiter에 의해 발견](http://people.idsia.ch/~juergen/fundamentaldeeplearningproblem.html)되었는데, 최근에 깊은(deep) 구조들이 많이 사용되면서 최근에도 주목받고 있다.

이번 파트를 제대로 이해하기 위해서는 편미분과 기본 backpropagation 알고리즘의 동작에 대한 이해가 필요한데, 이 부분에 대해서는 [여기](http://cs231n.github.io/optimization-2/)와 [여기](http://colah.github.io/posts/2015-08-Backprop/)와 [여기](http://neuralnetworksanddeeplearning.com/chap2.html)를 순서대로 읽어본다면 큰 도움이 될 것이다.

## Backpropagation Through Time (BPTT)

RNN의 기본 계산 수식을 다시 적어보자. 문자명이 \\( o \\)에서 \\( \hat{y\_t} \\)으로 살짝 바뀌었는데, 참고하는 이전 문헌들과 맞추기 위해서이다.

$$
\begin{align}
s\_t & = \tanh ( U x\_t + W s\_{t-1} ) \\\\
\hat{y\_t} & = softmax ( V s\_t ) \\\\
\end{align}
$$

Loss (에러)도 이전에 cross entropy로 정의했었고, 그 식은 아래와 같다.

$$
\begin{align}
E(y\_t, \hat{y\_t}) & = -y\_t \log{\hat{y\_t}} \\\\
E(y, \hat{y}) & = -\sum_{t}{E\_t (y\_t, \hat{y\_t})} \\\\
& = -\sum\_t -y\_t \log \hat{y\_t} \\\\
\end{align}
$$

여기서 \\( y\_t \\)는 시간 스텝 t에서 실제 단어이고, \\( \hat{y\_t} \\)는 우리의 예측값이다. 보통 전체 시퀀스(문장)를 하나의 학습 데이터(샘플)로 생각하고, 총 에러는 매 시간 스텝(단어)마다의 에러의 총 합으로 취한다.

![rnn-unfolded-img](http://www.wildml.com/wp-content/uploads/2015/10/rnn-bptt1.png)

우리의 원래 목표는 파라미터 U, V, W에 대한 에러의 gradient를 계산해서 Stochastic Gradient Descent (SGD)를 이용해 좋은 파라미터 값들을 찾는 것임을 기억하자. 에러들을 더하듯이, 매 시간 스텝의 gradient도 하나의 학습 데이터에 대해 모두 더해준다: \\( \frac{\partial E}{\partial W} = \sum_{t}{\frac{\partial E\_t}{\partial W}} \\).

이 gradient들을 계산하기 위해선 미분의 chain rule을 사용한다. 에러에서부터 거꾸로 된 방향으로 계산하는 것이 결국 [backpropagation 알고리즘](http://colah.github.io/posts/2015-08-Backprop/)이 된다. 본 튜토리얼의 나머지 부분에서는 예시로 \\( E\_3 \\)을 기준으로 설명할 것이다.

$$
\begin{align}
\frac{\partial E\_3}{\partial V} & = \frac{\partial E\_3}{\partial \hat{y\_3}} \frac{\partial \hat{y\_3}}{\partial V} \\\\
& = \frac{\partial E\_3}{\partial \hat{y\_3}} \frac{\partial \hat{y\_3}}{\partial z\_3} \frac{\partial z\_3}{\partial V} \\\\
& = (\hat{y\_3} - y\_3) \otimes s\_3 \\\\
\end{align}
$$

위 식에서, \\( z\_3 = Vs\_3 \\)이고, \\( \otimes \\)는 두 벡터의 외적이다. 위의 수식 전개는 몇 가지 스텝을 건너뛴 것이기 때문에, 바로 이해가 안 된다면 직접 미분 계산을 해보면 좋은 연습이 될 것이다. 핵심 포인트는 \\( \frac{\partial E\_3}{\partial V} \\)가 현재 시간 스텝의 \\( \hat{y\_3}, y\_3, s\_3 \\)에만 의존한다는 점이다. 이 세 값을 갖고 있다면 V에 대한 gradient를 계산하는 것은 단순한 행렬곱이 된다.

그러나, \\( \frac{\partial E\_3}{\partial W} \\)에 대해서는 (U에 대해서도) 상황이 조금 다르다. 이를 살펴보기 위해 위에서처럼 chain rule을 전개해 보았다.

\\[ \frac{\partial E\_3}{\partial W} = \frac{\partial E\_3}{\partial \hat{y\_3}} \frac{\partial \hat{y\_3}}{\partial s\_3} \frac{\partial s\_3}{\partial W} \\]

여기서 \\( s\_t = \tanh ( U x\_t + W s\_{t-1} ) \\) 는 \\( s\_2 \\)에 의존하고, \\( s\_2 \\)는 W와 \\( s\_1 \\)에 의존해서 chain rule이 계속 이어진다. 따라서, W에 대한 미분을 하기 위해서는 \\( s\_2 \\)를 단순히 상수로 취급하면 안된다. 다시 chain rule을 적용한다면 아래 식을 얻을 수 있다.

\\[ \frac{\partial E\_3}{\partial W} = \sum_{k=0}^{3}{\frac{\partial E\_3}{\partial \hat{y\_3}} \frac{\partial \hat{y\_3}}{\partial s\_3} \frac{\partial s\_3}{\partial s\_k} \frac{\partial s\_k}{\partial W}} \\]

각 시간 스텝이 gradient에 기여하는 것을 전부 더해준다. 즉, W는 우리가 현재 처리중인 출력 부분까지의 모든 시간 스텝에서 사용되기 때문에, \\( t=3 \\)부터 \\( t=0 \\)까지 gradient들을 전부 backpropage(역전파, 거꾸로 계산해주는 과정) 해 주어야 한다.

![rnn-unfolded-gradient](http://www.wildml.com/wp-content/uploads/2015/10/rnn-bptt-with-gradients.png)

이 과정은 [deep Feedforward Neural Network](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)(깊은 피드포워드 신경망 구조, 피드백 연결이 없는 네트워크)에서 사용하는 원래의 backpropagation 알고리즘과 똑같은 것을 알 수 있다. 중요한 차이점은 매 시간 스텝마다 W에 대한 gradient를 더해준다는 점이다. 기존의 신경망 구조에서는 layer별로 파라미터를 공유하지 않기 때문에 계산 결과들을 서로 더해줄 필요가 없다. 하지만 개인적인 생각으로 BPTT는 결국 시간 스텝으로 펼쳐낸 RNN에서의 backpropagation을 거창하게 부르는 것이라고 생각한다. Backpropagation에서처럼 이전 layer로 전해주는 델타 벡터를 정의할 수 있다: \\( \delta\_{2}^{(3)} = \frac{\partial E\_3}{\partial z\_2} = \frac{\partial E\_3}{\partial s\_3} \frac{\partial s\_3}{\partial s\_2} \frac{\partial s\_2}{\partial z\_2} \\), 여기서 \\( z\_2 = U x\_2 + W s\_1 \\)(2번째 시간 스텝의 activation에서 nonlinearity를 거치기 이전 상태)이다. 이전 시간 스텝에도 같은 수식이 적용된다. (역자 주: 이전 스텝의 델타 벡터는 \\( \delta\_{1}^{(3)} = \frac{\partial E\_3}{\partial z\_1} = \frac{\partial E\_3}{\partial z\_2} \frac{\partial z\_2}{\partial s\_1} \frac{\partial s\_1}{\partial z\_1} = \delta\_{2}^{(3)} \frac{\partial z\_2}{\partial s\_1} \frac{\partial s\_1}{\partial z\_1} \\)와 같이 계산할 수 있다. 델타 벡터 값을 알고 있으면 파라미터에 대한 gradient를 계산하는게 편리한데, 시간 스텝 i일 때 \\( \frac{\partial E\_3}{\partial U} = \delta\_{i}^{(3)} x\_i^T, \frac{\partial E\_3}{\partial W} = \delta\_{i}^{(3)} s\_{i-1}^T \\)처럼 벡터 외적 한번으로 얻을 수 있다 - 물론, RNN이므로 계산된 값들은 더해주어야 한다.)

```python
def bptt(self, x, y):
    T = len(y)
    # Perform forward propagation
    o, s = self.forward_propagation(x)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.
    # For each output backwards...
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)
        # Initial delta calculation: dL/dz
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            # Add to gradients at each previous step
            dLdW += np.outer(delta_t, s[bptt_step-1])
            dLdU[:,x[bptt_step]] += delta_t
            # Update delta for next step dL/dz at t-1
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]
```

위 코드를 보면 왜 RNN의 기본 형태를 학습하는 것이 어려운지 확인할 수 있다. 입력 시퀀스들(문장들)은 20 단어도 넘을 정도로 상당히 길 수 있기 때문에 매우 깊은 layer들에 backpropagate해 주어야 한다. 실제 상황에서는 보통 backpropagation through time을 할 시간 스텝을 적당한 숫자로 정해준다.


## Vanishing Gradient 문제

튜토리얼의 [앞선 포스트](http://aikorea.org/blog/rnn-tutorial-1/)에서 RNN은 긴 시퀀스를 처리하는데 (long-range dependency를 처리하는데) 한계가 있다고 했었다. 즉, 주요 단어들 사이에 여러 시간 스텝이 지났다면 잘 기억하지 못한다. 이는 보통 문장의 의미를 파악하는데 있어서 가까이 있지 않은 단어들이 밀접한 관련이 있을 수도 있기 때문에 문제가 된다. 예로, "The man who wore a wig on his head went inside."라는 문장을 보면, 이 문장은 "man"이 "inside"로 가는 것에 대한 문장이지 "wig(가발)"에 대한 것이 아니다. 그러나 기본 RNN 모델은 남자보다 가발에 대한 정보를 더 잘 기억할텐데, 왜 그런지 이해하기 위해 위에서 계산한 gradient 식을 더 자세히 살펴보자.

$$ \frac{\partial E\_3}{\partial W} = \sum_{k=0}^{3}{\frac{\partial E\_3}{\partial \hat{y\_3}} \frac{\partial \hat{y\_3}}{\partial s\_3} \frac{\partial s\_3}{\partial s\_k} \frac{\partial s\_k}{\partial W}} $$

주목할 점은 \\( \frac{\partial s\_3}{\partial s\_k} \\)도 chain rule을 내포하고 있다는 점이다. 즉, \\( \frac{\partial s\_3}{\partial s\_1} = \frac{\partial s\_3}{\partial s\_2} \frac{\partial s\_2}{\partial s\_1} \\)이다. 또 하나는, 벡터를 벡터로 미분하고 있기 때문에 결과는 행렬([Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)라 부른다)이 나오게 된다. 위의 gradient를 다시 써보면,

$$ \frac{\partial E\_3}{\partial W} = \sum_{k=0}^{3}{\frac{\partial E\_3}{\partial \hat{y\_3}} \frac{\partial \hat{y\_3}}{\partial s\_3} \left( \prod\_{j=k+1}^{3}{\frac{\partial s\_j}{\partial s\_{j-1}}} \right) \frac{\partial s\_k}{\partial W}} $$

위 자코비안 행렬의 2-norm (절대값으로 생각하면 된다) 값의 최대값이 1이라는 것이 증명되었다 (여기서 증명하진 않겠지만 [이 논문](http://www.jmlr.org/proceedings/papers/v28/pascanu13.pdf)에서 자세하게 다루고 있다). 현재 사용하고 있는 activation 함수인 tanh(또는 sigmoid)는 모든 값을 -1부터 1까지로 매핑시켜주고, 미분값은 최대 1로 (sigmoid의 경우 1/4) 정해지기 때문이라고 생각할 수 있다.

<div class="imgcap">
<img src="http://nn.readthedocs.org/en/rtd/image/tanh.png">
<div class="thecap" style="text-align:center">tanh와 그 미분값 그래프</div>
</div>

tanh 함수와 sigmoid 함수는 양쪽 끝에서 미분값이 0으로 수렴하는 것을 볼 수 있다. 이 현상이 발생할 때, 그 뉴런이 포화되었다고 말하는데, 이런 뉴런들은 gradient가 거의 0이기 때문에 곱해지는 이전 layer의 gradient들도 0으로 수렴하게 만든다. 따라서, 행렬에 작은 값들이 들어있고 여러 (t-k번) 행렬곱이 이루어지면 gradient는 지수 함수로 감소하고, 시간 스텝 몇 번만 지나도 사라져 버린다 (vanish!). 시퀀스에서 여러 시간 스텝이 떨어진 곳에서는 gradient가 전달되지 못하고, 먼 과거의 상태(state)는 현재 스텝의 학습에 아무 도움이 되지 못하게 된다. 즉, long-range dependency를 제대로 배우지 못한다. Vanishing gradient 문제는 RNN에서만 나타나는 것이 아니다. Deep Feedforward Neural Network에서도 마찬가지로 발생하지만, RNN은 보통 시간 스텝 횟수만큼 매우 깊은 구조이기 때문에 이 문제가 훨씬 더 잘 나타난다.

Gradient 계산을 보면, 자코비안 행렬 안의 값들이 크다면 activation 함수와 네트워크 파라미터 값에 따라 gradient가 사라지는게 아니라 지수 함수로 증가하는 경우도 충분히 상상해볼 수 있다. 이 문제 역시 *exploding gradient 문제* 로 잘 알려져 있다. Vanishing gradient 문제가 더 많은 관심을 받은 이유는 두 가지인데, 하나는 exploding gradient 문제는 쉽게 알아차릴 수 있다는 점이다. Gradient 값들이 NaN (not a number)이 될 것이고 프로그램이 죽을 것이기 때문이다. 두 번째는, gradient 값이 너무 크다면 미리 정해준 적당한 값으로 잘라버리는 방법 ([이 논문](http://www.jmlr.org/proceedings/papers/v28/pascanu13.pdf)에서 다뤄졌다)이 매우 쉽고 효율적으로 이 문제를 해결하기 때문이다. Vanishing gradient 문제는 언제 발생하는지 바로 확인하기가 힘들고 간단한 해결법이 없기 때문에 더 큰 문제였다.

다행히도, 이 문제를 어느 정도 해결할 수 있는 몇 가지 방법이 있다. W 행렬을 적당히 좋은 값으로 잘 초기화 해준다면 vanishing gradient의 영향을 줄일 수 있고, regularization을 잘 정해줘도 비슷한 효과를 볼 수 있다. 더 보편적으로 사용되는 방법은 tanh나 sigmoid activation 함수 말고 [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))를 사용하는 것이다. ReLU는 미분값의 최대치가 1로 정해져있지 않기 때문에 gradient 값이 없어져버리는 일이 크게 줄어든다. 이보다 더 인기있는 해결책은 Long Short-Term Memory (LSTM)이나 Gated Recurrent Unit (GRU) 구조를 사용하는 방법이다. [LSTM은 1997년에 처음 제안](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)되었고, 현재 자연어처리 분야에서 가장 널리 사용되는 모델 중 하나이다. GRU는 [2014년에 처음 나왔고](http://arxiv.org/pdf/1406.1078v3.pdf), LSTM을 간략화한 버전이다. 두 RNN의 변형 구조 모두 vanishing gradient 문제 해결을 위해 디자인되었고, 효과적으로 긴 시퀀스를 처리할 수 있다는 것이 보여졌다. 이 구조들에 대해서는 다음 파트에서 다룰 것이다.

**질문이나 피드백은 댓글로 달아주세요!**

---
<p align="right">
<b>번역: 최명섭</b>
</p>

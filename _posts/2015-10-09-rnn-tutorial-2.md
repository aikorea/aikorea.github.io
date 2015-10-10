---
layout: post
comments: true
title:  "RNN Tutorial Part 2 - Python, NumPy와 Theano로 RNN 구현하기"
date:   2015-10-09
mathjax: true
---

[WildML](http://www.wildml.com/)의 두 번째 RNN 튜토리얼입니다. 파이썬(프로그래밍 언어)으로 NumPy(매트랩처럼 행렬 을 다룰 때 편한 파이썬 기본 패키지)와 Theano(파이썬 기반 딥러닝 라이브러리)를 활용하여 실제로 RNN 모델을 처음부터 구현해보는 내용으로 코드도 전부 올라와 있어서 도움이 많이 될 것 같습니다!

[원문(영어)으로 된 버전](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)을 거의 그대로 옮겨왔습니다. 번역에 이상한 점을 발견하셨거나 질문이 있으시다면 댓글로 달아주세요.

---

두 번째 RNN 튜토리얼입니다. Part 1은 [여기](http://aikorea.org/blog/rnn-tutorial-1/)로

[따라할 수 있는 코드는 Github에 있습니다.](https://github.com/dennybritz/rnn-tutorial-rnnlm)

## 언어 모델링

이 파트에서의 목표는 Recurrent Neural Network (RNN)을 이용하여 [언어 모델](https://en.wikipedia.org/wiki/Language_model)을 만드는 것이다. 즉, *m* 개의 단어로 이루어져 있는 문장이 있다고 하면, 언어 모델은 이 문장이 (특정 데이터셋에서) 나타날 확률을 다음과 같이 예측할 수 있게 해준다:

$$ P(w\_1,...,w\_m) = \prod_{i=1}^{m}{P(w\_i|w\_1,...,w\_{i-1})} $$

말로 풀어쓰자면, 문장이 나타날 확률은 이전 단어를 알고있을 때 각 단어가 나타날 확률의 곱이 된다. 따라서 “He went to buy some chocolate”라는 문장의 확률은 "He went to buy some"이 주어졌을 때 "chocolate"의 확률 곱하기 "He went to buy"가 주어졌을 때 "some"의 확률 곱하기 ... 문장의 시작에서 아무것도 안 주어졌을 때 "He"의 확률까지의 곱이 된다.

그렇다면 언어 모델은 왜 필요할까? 문서에서 문장을 볼 수 있을 확률을 왜 찾고 싶은 것인가?

첫째로, 이러한 모델은 점수를 매기는 메커니즘으로 활용될 수 있다. 예를 들어, 자동 기계 번역 시스템은 보통 하나의 입력 문장에 대해 여러 개의 후보 답안 문장을 생성한다. 여기서 언어 모델로 가장 확률이 높은 문장을 고를 수 있을 것이다. 직관적으로 보면, 가장 확률이 높은 문장은 문법적으로도 더 맞을 확률이 높다. 음성 인식 시스템에서도 비슷한 방식으로 점수를 매기는데 활용된다.

그런데 언어 모델 문제를 풀다보면 상당히 재미있는 부산물이 나타난다. 문장에서 이전 위치에 나타나는 단어들을 알 때 다음 단어가 나타날 확률을 얻을 수 있기 때문에, 이를 기반으로 새로운 텍스트를 생성해낼 수도 있는 것이다. 즉, *생성 모델*  (generative model)이 나타난다. 현재 갖고 있는 단어들의 시퀀스를 주고 결과로 얻은 단어들의 확률 분포에서 다음 단어를 샘플링하고, 문장이 완성될 때까지 계속 이 과정을 반복할 수 있다. Andrej Karpathy가 [블로그 포스트](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)에 언어 모델이 어떤 일들을 할 수 있는지에 대해 훌륭하게 정리해 주었다. Karpathy의 모델은 단어 기준이 아니라 글자(character) 단위로 학습되었고, 셰익스피어부터 리눅스 소스 코드까지 전부 다 생성해낼 수 있다.

위의 수식을 보면 각 단어들의 확률은 이전에 나왔던 **모든** 단어들에 의존하고 있다. 하지만 실제 구현에서는 많은 모델들이 계산량, 메모리 문제 등으로 인해 long-term dependency를 효과적으로 다루지 못해서 긴 시퀀스는 처리하는 것이 힘들다. 이론적으로 RNN 모델은 임의의 길이의 시퀀스를 전부 기억할 수 있지만 실제로는 조금 더 복잡한데, 이에 대해서는 나중 포스트에서 다룰 예정이다.

## 학습 데이터 전처리 과정

언어 모델을 학습하기 위해선 학습할 텍스트가 필요하다. 다행히, 언어 모델 학습을 위해서는 특정 라벨들이 붙은 데이터가 필요 없고, 그냥 텍스트만 있으면 된다. 이를 위해 [구글 BigQuery에서 구할 수 있는 데이터셋](https://bigquery.cloud.google.com/table/fh-bigquery:reddit_comments.2015_08)으로 15,000개의 긴 reddit 댓글들을 다운받았다. (역자 주: reddit은 미국의 디씨같은 개념으로 생각하시면 됩니다. 다만, 게시판마다 사람들의 전문성이 어마어마해서 정말 유명한 교수들이나 연구자들이 여기저기 댓글을 달아주기도 하는 그런 곳이에요) 우리가 학습한 언어 모델은 아마 reddit 댓글러들의 말투와 비슷할 것이다. 하지만 그 전에 대부분의 기계 학습 프로젝트에서 필요하듯이, 텍스트 데이터를 쓰기 좋은 형태로 전처리를 해줘야 한다.

### 1. 텍스트의 *토큰* 화 (Tokenize Text)

텍스트 데이터에서 단어 단위로 예측을 하기 위해서는 댓글을 문장으로 *토큰화* 하고, 문장을 단어 단위로 쪼개야 한다. 단순히 공백(스페이스바)을 기준으로 자를 수도 있겠지만, 이는 문장 부호들을 제대로 처리하지 못하게 된다. 예시로, "He left!"라는 문장은 3개의 토큰 - "He", "left", "!" - 으로 이루어져야 한다. 여기서는 [NLTK](http://www.nltk.org/)의 `word_tokenize`와 `sent_tokenize` 방식을 사용하였다.

### 2. 빈도수가 낮은 단어들 없애기

우리의 데이터셋에 있는 대부분의 단어들은 한 번 내지 두 번 정도 등장한다. 이렇게 드문드문 나타나는 단어들은 없애는 것이 더 도움이 된다. 기억해야 할 단어의 종류가 너무 커지면 모델을 학습하는데 시간이 더 오래 걸리고 (왜 그런지는 나중에 얘기할 거에요), 빈도수가 낮은 단어들의 경우에는 어떤 상황에서 이런 단어들이 나타나는지에 대한 예시가 별로 없어서 잘 학습하기도 힘들기 때문이다. 사람이 배우는 것과도 비슷한데, 어떤 단어의 의미를 제대로 파악하려면 여러 상황에서 활용된 예시문을 봐야 할 것이다.

본 튜토리얼의 코드에서는 텍스트에 등장하는 빈도수 순으로 `vocabulary_size` 변수만큼으로 단어 수를 제한한다 (일단은 8000으로 정해줬는데, 값은 마음대로 바꾸시면 됩니다). 단어장에 없는 단어들은 전부 `UNKNOWN_TOKEN`으로 바꿔준다. 예시로, 단어장에 "nonlinearities"라는 단어가 없다면, "nonlineraties are important in neural networks"라는 문장은 "UNKNOWN\_TOKEN are important in neural networks"로 바뀔 것이다. `UNKNOWN_TOKEN`이라는 단어는 단어장에 추가되고, 다른 단어처럼 나타날 확률 예측도 하게 된다. 새로운 텍스트를 생성할 때는 `UNKNOWN_TOKEN`을 (예를 들면) 단어장에 없는 단어 중에서 아무거나 랜덤으로 뽑아서 대체할 수도 있고, 아니면 `UNKNOWN_TOKEN`이 나오기 전까지만 문장을 생성하는 방법도 있다.

### 3. 시작 토큰과 끝 토큰 붙이기

언어 모델은 어떤 단어들이 문장의 맨 처음 나타나고, 어떤 것들이 맨 뒤에 나타나는지도 학습하고자 한다. 이를 위해서 특별히 `SENTENCE_START` 토큰을 문장의 맨 앞에 이어붙이고, `SENTENCE_END` 토큰을 문장의 맨 뒤에 붙일 것이다. 모든 문장에 대해 이 과정을 처리해 주고나면, 문제는 다음과 같이 바뀐다: 첫 번째 토큰이 `SENTENCE_START`일 때, 다음 단어는 무엇일까? (실제 문장의 첫 단어)

### 4. 학습 데이터 행렬 구성하기

RNN 모델의 입력은 string이 아니라 벡터들이기 때문에 단어들과 인덱스들 사이의 매핑 - `index_to_word`와 `word_to_index` - 을 먼저 만든다. 예를 들어, "friendly"라는 단어는 2001번 위치에 있을 수 있다. 학습 데이터 *x* 는 `[0, 179, 341, 416]`과 같이 생겼을 것이고, 여기서 0은 `SENTENCE_START`을 뜻한다. 해당하는 라벨 *y* 는 `[179, 341, 416, 1]` 정도로 나타내질 것이다. 우리 목적은 다음 단어를 예측하는 것이기 때문에 *y* 는 단순히 *x* 벡터를 한 포지션 shift하고 마지막 위치에 `SENTENCE_END` 토큰을 넣어준 것이어야 한다. 즉, `179`번 단어의 올바른 예측값은 실제 다음 단어인 `341`이 되어야 한다.

```python
vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading CSV file..."
with open('data/reddit-comments-2015-08.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
 
# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())
 
# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
 
print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
 
# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
 
print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
```

```
Reading CSV file...
Parsed 79170 sentences.
Found 65751 unique words tokens.
Using vocabulary size 8000.
The least frequent word in our vocabulary is 'devoted' and appeared 10 times.
 
Example sentence: 'SENTENCE_START i joined a new league this year and they have different scoring rules than i'm used to. SENTENCE_END'
 
Example sentence after Pre-processing: '[u'SENTENCE_START', u'i', u'joined', u'a', u'new', u'league', u'this', u'year', u'and', u'they', u'have', u'different', u'scoring', u'rules', u'than', u'i', u"'m", u'used', u'to', u'.', u'SENTENCE_END']'
```

```python
# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
```

우리 텍스트 데이터의 실제 학습 데이터 예시는 다음과 같다.

```python
x:
SENTENCE_START what are n't you understanding about this ? !
[0, 51, 27, 16, 10, 856, 53, 25, 34, 69]
 
y:
what are n't you understanding about this ? ! SENTENCE_END
[51, 27, 16, 10, 856, 53, 25, 34, 69, 1]
```


## RNN 모델 만들기

RNN에 대한 기본 개념은 튜토리얼의 [첫 번째 파트](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)에 정리되어 있다.

![rnn-unfolded](http://www.wildml.com/wp-content/uploads/2015/09/rnn.jpg)

우리 언어 모델을 위한 RNN 구조가 어떻게 생겼는지 조금 더 구체적으로 살펴보자. 입력 *x* 는 단어 시퀀스 (위에서 프린트된 예시처럼)이고 각 \\(x\_t)\\ 는 단어 하나를 나타낸다. 여기에서 한 가지 사항이 더 있는데, 행렬곱 계산 때문에 실제로는 단어 인덱스(ex. 36)를 바로 입력으로 사용할 수는 없다. 대신, 각 단어를 `vocabulary_size`(단어장의 크기) 사이즈의 one-hot 벡터로 나타낸다. 예를 들어, 36번 인덱스의 단어는 36번째 원소만 1이고 나머지는 다 0인 벡터가 된다. 따라서 각 \(x\_t)\\는 벡터가 될 것이고, *x* 는 각 row가 한 단어를 나타내는 행렬이 될 것이다. 이 변환은 전처리 과정에서 하지 않고 신경망 코드에서 수행하도록 하자. 네트워크의 출력 *o* 도 비슷한 형태를 갖게 되는데, 각 \(o\_t)\\는 `vocabulary_size` 크기의 벡터가 될 것이고, 벡터의 각 원소들은 그 인덱스에 해당하는 단어가 다음에 나타날 확률 값을 의미할 것이다.

첫 번째 튜토리얼에서 살표보았던 RNN의 수식을 다시 보자.
$$ s\_t = \tanh ( U x\_t + W s\_{t-1} ) $$
$$ o\_t = softmax ( V s\_t ) $$

각 변수들의 차원을 써보면 이해하는데 도움이 많이 된다. 단어장의 크기를 *C = 8000* 으로 잡고, hidden layer의 사이즈를 *H = 100* 으로 정해 보자. Hidden layer의 사이즈는 네트워크의 "메모리"라고 생각할 수도 있다. H를 키운다면 더 복잡한 패턴을 학습할 수 있겠지만, 그만큼 계산량이 많아질 것이다. 정해진 변수들로 차원을 써보면 다음과 같다.

\\( x\_t \in \mathbb{R}^{8000} \\)

\\( o\_t \in \mathbb{R}^{8000} \\)

\\( s\_t \in \mathbb{R}^{100} \\)

\\( U \in \mathbb{R}^{100 \times 8000} \\)

\\( V \in \mathbb{R}^{8000 \times 100} \\)

\\( W \in \mathbb{R}^{100 \times 100} \\)

U, V, W는 데이터로부터 학습하고자 하는 네트워크의 파라미터들이다. 따라서 학습해야 하는 파라미터 수는 총 \\( 2HC+H^2 \\) 개이다. *C = 8000, H = 100* 인 경우에는 1,610,000 개가 된다. 위 정보를 통해 모델의 bottleneck도 판단할 수 있다.  \\( x\_t \\)가 one-hot 벡터이기 때문에, U와 곱하는 것은 결국 U의 column을 하나 선택하는 것과 마찬가지라 일일히 행렬곱을 계산할 필요가 없다. 따라서, 가장 큰 행렬곱은 \\( Vs_t \\)이 된다. 이것이 우리가 단어장의 크기를 가능한 한 줄여야 하는 이유가 된다.

사전 정보는 다 배웠으니, 실전 코딩으로 들어가 보자.

### 초기값 설정

먼저 RNN 클래스에서 파라미터들의 초기값을 정해주는 것으로 시작한다. 나중에 Theano 버젼도 구현할 것이기 때문에, 이 클래스를 `RNNNumpy`라고 부르자. 파라미터 U, V, W를 초기화하는 것이 약간 까다롭다. 단순히 전부 0으로 초기화한다면, 모든 layer에서 동일한 (대칭적인) 계산이 이뤄질 것이다. 따라서 랜덤으로 초기화해 주어야 하고, 올바른 초기화 방법은 학습 결과에 매우 큰 영향을 미치기 때문에 많이 연구가 진행되었다. 그 결과, 가장 좋은 초기화 방법은 activation 함수 (우리의 경우는 *tanh* )에 의존하고, 한 [연구 결과](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)에서는 파라미터 값들을 *n* 이 이전 layer에서부터 들어오는 연결 수라고 할 때 \\( -\frac{1}{sqrt{n}}, \frac{1}{sqrt{n}}] \\) 구간에서 랜덤으로 정하는 것이 좋다고 한다. 이정도면 너무 쓸데없이 복잡해 보이기도 하지만, 너무 걱정할 필요는 없다. 보통은 작은 랜덤 값들로 초기화 시켜주면 적당히 잘 동작한다.

```python
class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
```

위 코드에서, `word_dim`은 단어장의 크기이고, `hidden_dim`은 hidden layer의 사이즈이다 (마음대로 정해줄 수 있다). `bptt_truncate` 파라미터에 대해서는 나중에 설명이 있을 것이니 일단 신경쓰지 말고 넘어가도록 하자.

### Forward Propagation

다음으로, 위에서 정의한 수식대로 forward propagation (단어 확률 예측하는 것)을 구현해 보자.

```python
def forward_propagation(self, x):
    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s because need them later.
    # We add one additional element for the initial hidden, which we set to 0
    s = np.zeros((T + 1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
    # The outputs at each time step. Again, we save them for later.
    o = np.zeros((T, self.word_dim))
    # For each time step...
    for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
        o[t] = softmax(self.V.dot(s[t]))
    return [o, s]
 
RNNNumpy.forward_propagation = forward_propagation
```

계산된 출력값 외에 hidden state도 같이 리턴해 준다. 이 값들은 나중에 gradient를 계산할 때 사용되기 때문에, 중복되는 계산을 수행하지 않기 위해 미리 저장해 두기 위함이다. 각 \\( o\_t \\)는 각 단어의 확률 벡터인데, 모델을 평가하거나 할 때는 가장 확률값이 높은 단어가 무엇인지만 알면 되는 경우가 있다. 이를 찾는 함수를 `predict`라 하고, 아래 구현하였다.

```python
def predict(self, x):
    # Perform forward propagation and return index of the highest score
    o, s = self.forward_propagation(x)
    return np.argmax(o, axis=1)
 
RNNNumpy.predict = predict
```

여기까지 새로 구현한 방법들을 한 번 돌려보고 예시 출력값을 확인해 보자.

```python
np.random.seed(10)
model = RNNNumpy(vocabulary_size)
o, s = model.forward_propagation(X_train[10])
print o.shape
print o
```

```
(45, 8000)
[[ 0.00012408  0.0001244   0.00012603 ...,  0.00012515  0.00012488
   0.00012508]
 [ 0.00012536  0.00012582  0.00012436 ...,  0.00012482  0.00012456
   0.00012451]
 [ 0.00012387  0.0001252   0.00012474 ...,  0.00012559  0.00012588
   0.00012551]
 ..., 
 [ 0.00012414  0.00012455  0.0001252  ...,  0.00012487  0.00012494
   0.0001263 ]
 [ 0.0001252   0.00012393  0.00012509 ...,  0.00012407  0.00012578
   0.00012502]
 [ 0.00012472  0.0001253   0.00012487 ...,  0.00012463  0.00012536
   0.00012665]]
```

문장의 각 단어 (위의 경우 총 45개)에 대하여, 우리 모델이 8000 개의 단어에 대한 예측을 하여 다음 단어의 확률 분포를 얻었다. U, V, W를 랜덤한 값으로 초기화해 주었기 때문에 지금 얻는 결과들은 완전히 랜덤이다. 다음 코드로 가장 확률이 높은 다음 단어의 인덱스들을 뽑아낼 수 있다.

```python
predictions = model.predict(X_train[10])
print predictions.shape
print predictions
```
```
(45,)
[1284 5221 7653 7430 1013 3562 7366 4860 2212 6601 7299 4556 2481 238 2539
 21 6548 261 1780 2005 1810 5376 4146 477 7051 4832 4991 897 3485 21
 7291 2007 6006 760 4864 2182 6569 2800 2752 6821 4437 7021 7875 6912 3575]
```

### Loss 계산

네트워크를 학습하기 위해선, 현재 상태에서 모델이 내는 에러를 잴 수 있어야 한다. 이것을 loss 함수 *L* 으로 부르고, 우리의 목적은 학습 데이터에 대해서 loss 함수를 최소화하는 파라미터 U, V, W의 값을 찾는 것이다. 많이 사용하는 loss 함수로는 [cross-entropy loss](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression)가 있다. 학습 데이터가 N개가 있고 (텍스트 안의 단어들) C 개의 클래스 (단어장의 크기)가 있다면, 실제 라벨 *y* 에 대한 우리 네트워크의 예측값 *o* 의 loss는 다음과 같이 계산된다.
$$ L(y,o) = -\frac{1}{N} \sum\_{n \in N}{y\_n \log{o\_n}} $$


조금 복잡해 보이지만, 실제로 위 식이 하는 일은 모든 학습 데이터에 대해 예측값이 틀린 정도를 계산해서 loss에 더해주는 것이다. 실제 단어인 *y* 가 *o* 에서 멀수록 loss는 커질 것이다. 아래 `calculate_loss` 함수를 구현하였다.

```python
def calculate_total_loss(self, x, y):
    L = 0
    # For each sentence...
    for i in np.arange(len(y)):
        o, s = self.forward_propagation(x[i])
        # We only care about our prediction of the "correct" words
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        # Add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions))
    return L
 
def calculate_loss(self, x, y):
    # Divide the total loss by the number of training examples
    N = np.sum((len(y_i) for y_i in y))
    return self.calculate_total_loss(x,y)/N
 
RNNNumpy.calculate_total_loss = calculate_total_loss
RNNNumpy.calculate_loss = calculate_loss
```

코드에서 한발짝 물러나서, 랜덤 예측의 경우에는 loss가 어떻게 되어야 할 지 생각해보자. 베이스라인이 무엇인지 알 수 있고, 구현이 제대로 되었는지 확인해볼 수 있다. 단어장에 *C* 개의 단어가 있으므로, 각 단어는 (평균적으로) *1/C* 의 확률로 예측되어야 하고, 이 때의 loss는 \\( L = -\frac{1}{N} N \log{\frac{1}{C}} = \log{C} \\) 이 될 것이다.

```python
# Limit to 1000 examples to save time
print "Expected Loss for random predictions: %f" % np.log(vocabulary_size)
print "Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000])
```
```
Expected Loss for random predictions: 8.987197
Actual loss: 8.987440
```

거의 비슷한 것을 확인할 수 있다. 전체 데이터셋에 대해 loss를 계산하는 것은 상당히 계산량이 많은 작업이고, 데이터가 많은 경우에는 몇 시간씩 걸릴 수도 있다는 사실을 기억해 두자.


### SGD와 Backpropagation Through Time (BPTT)를 이용하여 RNN 학습하기

우리 목적은 학습 데이터에 대해서 loss를 최소화하는 파라미터 U, V, W를 찾는 것임을 기억하자. 가장 보편적으로 사용되는 방법은 SGD (Stochastic Gradient Descent)이다. 기본 아이디어는 간단한데, 모든 학습 데이터에 대해 반복적으로 학습하되, 매 iteration에서 에러를 줄이는 쪽으로 파라미터 값들을 조금씩 움직이는 것이다. 에러를 줄이는 방향은 loss의 gradient로 주어진다: \\( \frac{\partial L}{\partial U}, \frac{\partial L}{\partial V}, \frac{\partial L}{\partial W} \\). SGD에서는 *learning rate* 도 필요한데, 이는 매 iteration에서 얼만큼 큰 스텝만큼 파라미터 값 업데이트가 이루어질지 정해준다. SGD는 신경망 구조 외의 다른 많은 기계 학습 알고리즘들에서도 사용하는 최적화 기법이기 때문에, SGD를 잘 활용하는 법에 대한 연구 결과 및 트릭들도 많이 있다 (batch로 활용하는 법, learning rate 변화법 등). 아이디어는 간단하지만, 효율적으로 SGD를 구현하는 것은 복잡해질 수 있다. SGD에 대해 더 자세히 알고자 한다면 [여기](http://cs231n.github.io/optimization-1/)를 참고하기 바란다. 워낙에 유명한 알고리즘이고 웹 상의 여기저기에 좋은 튜토리얼들이 많기 때문에, 본 튜토리얼에서 SGD의 기본 개념에 대한 것을 자세히 다루지는 않을 것이다. 아래 구현에서는 최적화에 대한 사전 지식이 없더라도 충분히 이해할 수 있는 간단한 버전의 SGD를 만들 것이다.

그렇다면 앞에서 언급한 gradient는 어떻게 계산될까? [기존의 신경망 구조](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)에서는 backpropagation 알고리즘을 이용한다. RNN에서는 이를 살짝 변형시킨 버전인 *Backpropagation Through Time (BPTT)* 을 사용하는데, 그 이유는 각 파라미터들이 네트워크의 매 시간 스텝마다 공유되기 때문이다. 즉, 각 시간 스텝의 출력단에서의 gradient는 현재 시간 스텝에서의 계산에만 의존하는 것이 아니라 이전 시간 스텝에도 의존한다. 미적분에 대한 지식이 있다면 이 알고리즘은 결국 chain rule을 적용하는 것뿐이다. 본 튜토리얼의 다음 파트는 전부 BPTT에 대한 내용일 것이기 때문에, 여기서 자세한 유도 과정을 보이진 않겠다. Backpropagation에 대한 내용을 자세히 복습하고 싶다면 [여기]와 [이 포스트]를 확인하기 바란다. 일단 당장은 BPTT를 black box 함수처럼 생각하도록 하자. 입력으로는 학습 데이터 샘플 (*x, y*)을 받고, 출력으로 gradient들 - \\( \frac{\partial L}{\partial U}, \frac{\partial L}{\partial V}, \frac{\partial L}{\partial W} \\) - 을 내놓는다.


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
        # Initial delta calculation
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            dLdW += np.outer(delta_t, s[bptt_step-1])              
            dLdU[:,x[bptt_step]] += delta_t
            # Update delta for next step
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]
 
RNNNumpy.bptt = bptt
```

### Gradient 체크

Backpropagation을 구현할 때는 gradient값을 체크하는 코드도 같이 구현하는 것이 우리 구현이 맞는지 확인하는데 도움이 된다. 기본 아이디어는 파라미터에 대한 미분값은 그 지점의 기울기와 같다는 점인데, 이는 아래 미분의 정의식에서처럼 근사하여 계산할 수 있다.

$$ \frac{\partial L}{\partial \theta} \approx \lim_{h \to 0}{\frac{J(\theta + h) - J(\theta - h)}{2h}} $$

Backpropagation을 통해 우리가 얻은 gradient 값과 위 수식의 방법으로 얻은 값을 비교해보고, 거의 차이가 없다면 구현이 제대로 된 것을 확인할 수 있다. 이 계산은 *모든* 파라미터에 대해 이루어져야 하므로, gradient 체크는 매우 계산량이 많은 (위 예시에서 100만 개가 넘는 파라미터가 있었음을 기억하자) 과정이다. 따라서, 작은 단어장 크기에 대해 수행해서 구현이 제대로 되었는지 확인하는 용도로만 사용하면 될 것이다.

```python
def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
    # Calculate the gradients using backpropagation. We want to checker if these are correct.
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to check.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter = operator.attrgetter(pname)(self)
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            # Reset parameter to original value
            parameter[ix] = original_value
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print "Backpropagation gradient: %f" % backprop_gradient
                print "Relative Error: %f" % relative_error
                return 
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)
 
RNNNumpy.gradient_check = gradient_check
 
# To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
grad_check_vocab_size = 100
np.random.seed(10)
model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
model.gradient_check([0,1,2,3], [1,2,3,4])
```


### SGD 구현

이제 파라미터들에 대한 gradient를 계산할 수 있으므로 SGD 알고리즘을 구현해 보도록 하자. 두 단계로 나눠서 진행할텐데, 1. `sgd_step` 함수는 gradient를 계산하고 하나의 batch에 대해 파라미터 값을 업데이트한다. 2. 바깥쪽 루프에서 학습 데이터셋에 대한 iteration이 일어나고, learning rate을 조정할 것이다.

```python
# Performs one step of SGD.
def numpy_sdg_step(self, x, y, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)
    # Change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW
 
RNNNumpy.sgd_step = numpy_sdg_step
```

```python
# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
```

끝! 이제 네트워크를 학습하는데 시간이 어느 정도 걸릴지에 대한 대략적인 감을 잡아 보자.

```python
np.random.seed(10)
model = RNNNumpy(vocabulary_size)
%timeit model.sgd_step(X_train[10], y_train[10], 0.005)
```

좋지 않은 소식이다. 현재 내 노트북에서 SGD 한 스텝은 약 350 ms가 걸리는데, 현재 학습 데이터에는 80,000개의 데이터가 있으므로 한 epoch (데이터셋 전체에 대한 iteration)은 7~8시간이 걸릴 것이다. 여러 epoch을 돌리려면 몇 일, 몇 주가 걸릴지도 모른다. 게다가 현재 사용하고 있는 데이터셋은 많은 기업과 연구자들이 사용하고 있는 데이터에 비하면 매우 작은 데이터셋에 불과하다.

하지만 다행히도, 우리의 언어 모델을 빠르게 해줄 다양한 방법이 존재한다. 같은 구조를 사용하면서 코드가 더 빠르게 돌도록 수정하는 방법도 있고, 계산량이 더 적어지도록 네트워크 구조를 변형시킬 수도 있고, 둘 다 수행할 수도 있다. 많은 연구자들이 계산량이 적게 필요한 모델에 대해 연구해왔는데, 몇 가지 에시로는 hierarchical softmax, 큰 행렬곱을 피하기 위한 projection layer 등이 있다 ([이 논문](http://arxiv.org/pdf/1301.3781.pdf)와 [또다른 논문 하나](http://www.fit.vutbr.cz/research/groups/speech/publi/2011/mikolov_icassp2011_5528.pdf) 참고). 여기서는 현재의 간단한 구조를 활용하도록 하고, 대신 GPU를 이용하여 속도를 높여보도록 하자. 그 전에, 일단 매우 작은 데이터셋에 대해 SGD를 한 번 적용해서 실제로 loss가 줄어드는지 확인해보자.

```python
np.random.seed(10)
# Train on a small subset of the data to see what happens
model = RNNNumpy(vocabulary_size)
losses = train_with_sgd(model, X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)
```

```
2015-09-30 10:08:19: Loss after num_examples_seen=0 epoch=0: 8.987425
2015-09-30 10:08:35: Loss after num_examples_seen=100 epoch=1: 8.976270
2015-09-30 10:08:50: Loss after num_examples_seen=200 epoch=2: 8.960212
2015-09-30 10:09:06: Loss after num_examples_seen=300 epoch=3: 8.930430
2015-09-30 10:09:22: Loss after num_examples_seen=400 epoch=4: 8.862264
2015-09-30 10:09:38: Loss after num_examples_seen=500 epoch=5: 6.913570
2015-09-30 10:09:53: Loss after num_examples_seen=600 epoch=6: 6.302493
2015-09-30 10:10:07: Loss after num_examples_seen=700 epoch=7: 6.014995
2015-09-30 10:10:24: Loss after num_examples_seen=800 epoch=8: 5.833877
2015-09-30 10:10:39: Loss after num_examples_seen=900 epoch=9: 5.710718
```

원했던대로 loss가 줄어들도록 무언가 열심히 계산을 제대로 하고 있는 것을 확인할 수 있다.


## Theano와 GPU를 활용한 네트워크 학습

이전에 [Theano에 대한 튜토리얼](http://www.wildml.com/2015/09/speeding-up-your-neural-network-with-theano-and-the-gpu/)을 작성한 바 있어서 중복되는 디테일한 내용은 생략하도록 하겠다. 여기서는 `RNNTheano`라는 함수를 정의해서 이전에 numpy를 이용한 계산들을 해당하는 Theano 계산으로 치환하였다. 이 포스트의 다른 코드들과 마찬가지로, [코드는 Github에서](https://github.com/dennybritz/rnn-tutorial-rnnlm) 볼 수 있다.

```python
np.random.seed(10)
model = RNNTheano(vocabulary_size)
%timeit model.sgd_step(X_train[10], y_train[10], 0.005)
```

이번에는 내 맥북에서 SGD 한 스텝이 70 ms만에 계산되고 (GPU 없이), Amazon EC2 인스턴스의 [g2.2xlarge](https://aws.amazon.com/ko/ec2/instance-types/#g2)를 활용한다면 23 ms만에 가능하다. (역자 주: 아마존 웹서비스 중에서 GPU를 활용할 수 있는 건데, 다나와에서 30만원짜리 정도의 저렴한 GPU와 성능이 비슷한 것 같습니다. 아마존에서는 750 시간을 무료로 시험해 볼 수 있습니다.) 처음 구현했던 것에 비하면 15배가 빨라져서, 이제 몇 주씩 걸리던 작업을 몇 시간 내로 끝낼 수 있게 되었다. 훨씬 더 다양한 속도 최적화 기법들이 있지만, 일단은 이 정도까지만 하자.

이 튜토리얼을 읽고 실제로 학습을 시켜볼 모든 사람들이 몇 일씩 컴퓨터를 돌려놓는 것을 방지하기 위해 Theano 모델 하나를 pre-train해 놓았다. Hidden node의 개수는 50, 단어장의 크기는 8000으로 정하였고, 약 20 시간 동안 50 epoch을 돌려놓았다. 이 때까지도 loss는 계속 감소하고 있었고 더 오랜 시간동안 학습한다면 더 좋은 모델을 얻을 수 있었겠지만, 이 포스트를 얼른 작성하기 위해 이 정도에서 멈췄다. 직접 테스트 해보거나 더 오랜 시간동안 이어서 학습시켜보려면 역시 Github에 올라와있는 코드를 활용하기 바란다. 모델 파라미터들은 `data/trained-model-theano.npz`에서 찾을 수 있고, `load_model_parameters_theano` 방법으로 불러올 수 있다.

```python
from utils import load_model_parameters_theano, save_model_parameters_theano
 
model = RNNTheano(vocabulary_size, hidden_dim=50)
# losses = train_with_sgd(model, X_train, y_train, nepoch=50)
# save_model_parameters_theano('./data/trained-model-theano.npz', model)
load_model_parameters_theano('./data/trained-model-theano.npz', model)
```


## 텍스트 생성

이제 모델이 완성되었으니 이것을 이용해서 새로운 텍스트를 만들어낼 수 있다! 새로운 문장을 만들어내기 위한 helper 함수 몇 개를 더 구현해 보자.

```python
def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str
 
num_sentences = 10
senten_min_length = 7
 
for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print " ".join(sent)
```

모델이 생성한 것 중에서 몇 가지 엄선된 문장들은 다음과 같다. 대문자는 직접 바꿔주었다.

- Anyway, to the city scene you’re an idiot teenager.
- What ? ! ! ! ! ignore!
- Screw fitness, you’re saying: https
- Thanks for the advice to keep my thoughts around girls.
- Yep, please disappear with the terrible generation.

몇 가지 재밌는 점은, 모델이 문장 부호에 대한 문법은 성공적으로 학습했다는 점이다. 쉼표를 그럴듯한 자리에 넣어주었고, 마침표나 느낌표로 문장이 끝난다. 가끔씩은 인터넷 말투를 따라해서 느낌표를 여러 개 넣거나 :) (웃음 표시)를 만들어 내기도 한다.

그러나 대부분의 문장들은 단어들의 의미없는 나열에 불과하거나 문법적으로 말이 안 되는 경우가 많다 (위의 것들은 정말 *엄선된* 것이라..). 한 가지 이유는 네트워크를 충분히 오랜 시간동안 학습하지 않아서일 수도 있고, 학습 데이터가 부족했을 수도 있다. 일리있는 분석이지만, 정말 중요한 요인은 아마 아닐 것이다. 그 이유는, **RNN 기본 형태의 모델로는 긴 단어들의 시퀀스를 기억하지 못하기 때문에 (long-term dependency를 효과적으로 다루지 못하기 때문에) 의미있는 텍스트를 생성하는 것이 어렵다**. RNN이 처음 발명되었을때 많은 인기를 못 얻었던 것도 이 때문이다. 이론적으로는 매우 아름답지만 실제로 동작하지 않았고, 왜 그런지에 대해서 초창기에는 이해하지 못했었다.

다행히도 현재에는 [RNN 학습의 어려움에 대한 이해도](http://arxiv.org/abs/1211.5063)가 많이 높아졌다. 본 튜토리얼의 다음 파트에서는 Backpropagation Through Time (BPTT)에 대해 보다 더 자세히 살펴보고, *vanishing gradient 문제* 에 대해 공부해 볼 것이다. 이런 문제들로 인해 RNN 모델이 더 복잡해져서 현재 많은 자연어처리 문제에 대해 state-of-the-art 성능을 보이고 있는 LSTM 등이 제시되었다 (그리고 훨씬 더 그럴듯한 reddit 댓글을 만들어낼 수 있다!). **본 튜토리얼에서 배운 모든 내용은 LSTM과 다른 RNN 모델에도 그대로 적용되므로, 여기서 구현한 기본 RNN 모델의 성능이 안 좋다고 너무 실망하지는 않았으면 좋겠다.**

파트 2는 여기까지이다. **질문이나 피드백이 있다면 댓글로, 그리고 [Github에 올라와 있는 코드]도 꼭 확인해보기 바란다.**


---
<p align="right">
<b>번역: 최명섭</b>
</p>



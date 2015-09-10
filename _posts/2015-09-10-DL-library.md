---
layout: post
title: 프로그래밍 언어별 딥러닝 라이브러리 정리
---

AI Korea Open 그룹에서도 라이브러리에 관한 *[투표](https://www.facebook.com/groups/AIKoreaOpen/permalink/1029968110370632/)*가 있었고, 많은 분들이 관심있어할 만한 부분이라 생각해서 한 번 정리해 봤습니다!
![dl_library_vote](https://www.dropbox.com/s/svrlvvdgd1ykci9/DL_lib_vote.PNG)


## Python
요즘 뜨는 언어답게, 대부분의 라이브러리들이 빠른 속도로 업데이트되며 새로운 기능이 계속 추가되고 있다. 

1. [Theano](http://deeplearning.net/software/theano) - 수식 및 행렬 연산을 쉽게 만들어주는 파이썬 라이브러리. 딥러닝 알고리즘을 파이썬으로 쉽게 구현할 수 있도록 해주는데, Theano 기반 위에 얹어서 더 사용하기 쉽게 구현된 여러 라이브러리가 있다.
   * [Keras](http://keras.io/) - Theano 기반이지만 Torch처럼 모듈화가 잘 되어 있어서 사용하기 쉽고 최근에도 계속 업데이트되며 빠른 속도로 발전하고 있는 라이브러리.
   * [Pylearn2](http://deeplearning.net/software/pylearn2/) - Theano를 유지, 보수하고 있는 Montreal 대학의 Yoshua Bengio 그룹에서 개발한 Machine Learning 연구용 라이브러리
   * [Lasagne](https://github.com/Lasagne/Lasagne) - 가볍고 모듈화가 잘 되어 있어서 사용하기 편리함
   * [Blocks](https://github.com/mila-udem/blocks) - 위 라이브러리와 비슷하게 역시 Theano 기반으로 손쉽게 신경망 구조를 구현할 수 있도록 해주는 라이브러리 
2. [Chainer](http://chainer.org/) - 거의 모든 딥러닝 알고리즘을 직관적인 Python 코드로 구현할 수 있고, 자유도가 매우 높음. 대다수의 다른 라이브러리들과는 다르게 "Define-by-Run" 형태로 구현되어 있어서, forward 함수만 정의해주면 네트워크 구조가 자동으로 정해진다는 점이 특이하다.
3. [nolearn](https://github.com/dnouri/nolearn) - scikit-learn과 연동되며 기계학습에 유용한 여러 함수를 담고 있음.
4. [Gensim](http://radimrehurek.com/gensim/) - 큰 스케일의 텍스트 데이터를 효율적으로 다루는 것을 목표로 한 Python 기반 딥러닝 툴킷
5. [deepnet](https://github.com/nitishsrivastava/deepnet) - cudamat과 cuda-convnet 기반의 딥러닝 라이브러리
6. [CXXNET] (https://github.com/dmlc/cxxnet) - MShadow 라이브러리 기반으로 멀티 GPU까지 지원하며, Python 및 Matlab 인터페이스 제공
7. [DeepPy](https://github.com/andersbll/deeppy) - NumPy 기반의 라이브러리
8. [Neon](https://github.com/NervanaSystems/neon) - Nervana에서 사용하는 딥러닝 프레임워크

## Matlab
1. [MatConvNet](http://www.vlfeat.org/matconvnet/) - 컴퓨터비젼 분야에서 유명한 매트랩 라이브러리인 vlfeat 개발자인 Oxford의 코딩왕 Andrea Vedaldi 교수와 학생들이 관리하는 라이브러리. 
2. [ConvNet](https://github.com/sdemyanov/ConvNet)
3. [DeepLearnToolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox)

## C++
1. [Caffe](http://caffe.berkeleyvision.org/) - Berkeley 대학에서 관리하고 있고, 현재 가장 많은 사람들이(추정) 사용하고 있는 라이브러리. C++로 직접 사용할 수도 있지만 Python과 Matlab 인터페이스도 잘 구현되어 있다.
2. [DIGITS](https://developer.nvidia.com/digits) - NVIDIA에서 브라우저 기반 인터페이스로 쉽게 신경망 구조를 구현, 학습, 시각화할 수 있도록 개발한 시스템.
3. [cuda-convnet] (https://code.google.com/p/cuda-convnet/) - 딥러닝 슈퍼스타인 Alex Krizhevsky와 Geoff Hinton이 ImageNet 2012 챌린지를 우승할 때 사용한 라이브러리
4. [eblearn](http://sourceforge.net/projects/eblearn/) - 딥러닝 계의 또하나의 큰 축인 NYU의 Yann LeCun 그룹에서 ImageNet 2013 챌린지를 우승할 때 사용한 라이브러리
5. [SINGA](http://www.comp.nus.edu.sg/~dbsystem/singa/)

## Java
1. [ND4J](http://nd4j.org/)
2. [Deeplearning4j](http://deeplearning4j.org/)
3. [Encog](http://www.heatonresearch.com/encog)

## JavaScript
자바스크립트로의 딥러닝 구현은 Stanford의 Andrej Karpathy가 혼자서 개발했음에도 불구하고 높은 완성도를 보이며 널리 사용되고 있는 아래 두 라이브러리가 가장 유명하다.

1. [ConvnetJS](http://cs.stanford.edu/people/karpathy/convnetjs/)
2. [RecurrentJS](https://github.com/karpathy/recurrentjs)

## Lua
1. [Torch](http://torch.ch/) - 페이스북과 구글 딥마인드에서 사용하는 라이브러리. 양대 대기업에서 사용하고 있는 만큼 필요한 거의 모든 기능이 잘 구현되어 있고, 스크립트 언어인 Lua를 사용하기 때문에 쉽게 사용 가능하다.

## Julia
MIT에서 새로 개발한 언어로, 최근에 주목받기 시작하여 효율적인 딥 러닝 라이브러리도 여러 가지 구현되었다.

1. [Mocha.jl](https://github.com/pluskid/Mocha.jl)
2. [Strada.jl](https://github.com/pcmoritz/Strada.jl)
3. [KUnet.jl](https://github.com/denizyuret/KUnet.jl)

## Lisp
1. [Lush](http://lush.sourceforge.net/)

## Haskell
1. [DNNGraph](https://github.com/ajtulloch/dnngraph)

## .NET
1. [Accord.NET](http://accord-framework.net/)

## R
1. [darch](http://cran.um.ac.ir/web/packages/darch/index.html)
2. [deepnet](https://cran.r-project.org/web/packages/deepnet/index.html)

출처: http://www.teglor.com/b/deep-learning-libraries-language-cm569/ 블로그 포스트 자료에서 약간의 수정을 거쳤습니다.

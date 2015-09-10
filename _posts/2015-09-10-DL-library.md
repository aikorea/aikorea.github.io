---
layout: post
title: 프로그래밍 언어별 딥러닝 라이브러리 정리
---

AI Korea Open 그룹에서도 라이브러리에 관한 논의가 있었고, 많은 분들이 관심있어할 만한 부분이라 생각해서 한 번 정리해 봤습니다!

## Python
1. [Theano](http://deeplearning.net/software/theano)
수식 및 행렬 연산을 쉽게 만들어주는 파이썬 라이브러리.
딥러닝 알고리즘을 파이썬으로 쉽게 구현할 수 있도록 해주는데, Theano 기반 위에 얹어서 더 사용하기 쉽게 구현된 여러 라이브러리가 있다.
  * [Keras](http://keras.io/)
  * [Pylearn2](http://deeplearning.net/software/pylearn2/)
  * [Lasagne](https://github.com/Lasagne/Lasagne)
  * [Blocks](https://github.com/mila-udem/blocks)
2. [Chainer](http://chainer.org/)
거의 모든 딥러닝 알고리즘을 직관적인 Python 코드로 구현할 수 있고, 자유도가 매우 높음.
3. [nolearn](https://github.com/dnouri/nolearn)
4. [Gensim](http://radimrehurek.com/gensim/)
5. [deepnet](https://github.com/nitishsrivastava/deepnet)
cudamat과 cuda-convnet 기반의 딥러닝 라이브러리
6. [Hebel](https://github.com/hannes-brt/hebel)
PyCUDA를 활용하여 구현된 딥러닝/신경망 라이브러리
7. [CXXNET] (https://github.com/dmlc/cxxnet)
8. [DeepPy](https://github.com/andersbll/deeppy)
9. [Neon](https://github.com/NervanaSystems/neon)

## Matlab
1. [MatConvNet](http://www.vlfeat.org/matconvnet/)
2. [ConvNet](https://github.com/sdemyanov/ConvNet)
3. [DeepLearnToolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox)

## C++
1. [Caffe](http://caffe.berkeleyvision.org/)
2. [DIGITS](https://developer.nvidia.com/digits)
3. [cuda-convnet] (https://code.google.com/p/cuda-convnet/)
딥러닝 슈퍼스타인 Alex Krizhevsky와 Geoff Hinton이 ImageNet 2012 챌린지를 우승할 때 사용한 라이브러리
4. [eblearn](http://sourceforge.net/projects/eblearn/)
5. [SINGA](http://www.comp.nus.edu.sg/~dbsystem/singa/)

## Java
1. [ND4J](http://nd4j.org/)
2. [Deeplearning4j](http://deeplearning4j.org/)
3. [Encog](http://www.heatonresearch.com/encog)

## JavaScript
1. [ConvnetJS](http://cs.stanford.edu/people/karpathy/convnetjs/)
2. [RecurrentJS](https://github.com/karpathy/recurrentjs)

## Lua
1. [Torch](http://torch.ch/)
페이스북과 구글 딥마인드에서 사용하는 라이브러리.

## Julia
1. [Mocha.jl](https://github.com/pluskid/Mocha.jl)
2. [Strada.jl](https://github.com/pcmoritz/Strada.jl)
3. [KUnet.jl](https://github.com/denizyuret/KUnet.jl)

## Lisp
1. [Lush](http://lush.sourceforge.net/)

## Haskell
1. [DNNGraph](https://github.com/ajtulloch/dnngraph)

출처: http://www.teglor.com/b/deep-learning-libraries-language-c…/ 블로그 포스트 자료에서 약간의 수정을 거쳤습니다.

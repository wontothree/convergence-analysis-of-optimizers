# Convergence-analysis-of-optimizers

## Empirical evaluation

1. **최종 성능 (Final Performance):** 훈련이 완료된 후 모델의 최종 성능을 평가
2. **수렴 속도 (Convergence Speed):** 각 옵티마이저가 모델을 수렴시키는 데 걸리는 시간

Optimizer들의 수렴성을 분석하고, Convergence analysis of optimizers에 실험을 진행합니다.

Image classification Experiment

|Datasets|CNN Models|Optimizers|
|---|---|---|
|ImageNet|AlexNet|SGD|
||VggNet|Momentum|
||GoogleNet|Adagrad|
||ResNet|RMSProp|
|||Adam|

## Imagenet

Imagenet : 224x224

*스마트폰의 해상도 : 1080x1920

## CNN Architectures

||AlexNet|VGGNet|GoogleNet|ResNet|
|---|---|---|---|---|
|Convolution layer|5|13|21|152|
|Fully connected layer|3|3|1|3(전역 평균 풀링층)|
|규제 기법|드롭 아웃|||배치 정규화|

## 궁금한 점

옵티마이저의 일반화 능력을 어떻게 정량적으로 평가할 수 있을까?

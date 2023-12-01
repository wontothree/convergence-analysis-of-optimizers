# Convergence-analysis-of-optimizers

Optimizer들의 수렴성을 분석하고, Convergence analysis of optimizers에 실험을 진행합니다.

두 가지 측면에서 옵티마지어의 성능을 비교합니다.

1. **최종 성능 (Final Performance):** 훈련이 완료된 후 모델의 최종 성능을 평가
2. **수렴 속도 (Convergence Speed):** 각 옵티마이저가 모델을 수렴시키는 데 걸리는 시간

training loss와 valid loss의 차이가 뭐야

## Mnist

고등학생과 미국 인구조사국 직원들이 손으로 쓴 70,000개의 0부터 9까지의 숫자 이미지를 모은 데이터셋

- 각 이미지에는 어떤 숫자를 나타내는지 label되어 있다.
- 훈련 데이터 6만 개, 테스트 데이터가 1만개
- 이미지는 28 x 28 해상도의 흑백 사진이다.
- 각 픽셀은 0부터 255까지의 밝기를 가진다.

## Cifar-10

- CIFAR-10과 CIFAR-100은 8000만 개의 작은 이미지 데이터셋의 레이블이 지정된 하위 집합입니다.
- CIFAR-10 데이터셋은 10개의 클래스에서 각각 6000개의 이미지로 이루어진 총 60000개의 32x32 컬러 이미지로 구성되어 있습니다.
- 50000개는 훈련 이미지이고, 10000개는 테스트 이미지입니다.
- 이 데이터셋은 5개의 훈련 배치와 1개의 테스트 배치로 나뉘어져 있으며, 각각에는 10000개의 이미지가 포함되어 있습니다.
- 테스트 배치에는 각 클래스에서 무작위로 선택된 정확히 1000개의 이미지가 포함되어 있습니다.
- 훈련 배치에는 나머지 이미지가 무작위로 포함되어 있지만, 일부 훈련 배치에는 다른 클래스의 이미지보다 더 많은 이미지가 포함될 수 있습니다.
- 이들 훈련 배치를 통틀어 각 클래스당 정확히 5000개의 이미지가 포함되어 있습니다.

## ImageNet

## 실험 진행 방식

이미지 분류 문제

- 데이터 셋 : mnist, cifar10, cifar100, ImageNet, SVHN, Fashion MNIST
- 모델 : AlexNet, VGGNet, GoogleNet, ResNet, (+ Generative model)
- 옵티마이저 : SGD, Momentum, Adagrad, RMSProp, Adam
- 평가지표 : training loss, validation loss, 모델의 과적합, 학습 속도, 안정성, 다양성, 하이퍼 파라미터 튜닝

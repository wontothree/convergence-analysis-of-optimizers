# Convergence-analysis-of-optimizers

Optimizer들의 수렴성을 분석하고, Convergence analysis of optimizers에 실험을 진행합니다.

Image classification Experiment

|Datasets|CNN Models|Optimizers|
|---|---|---|
|Mnist|AlexNet|SGD|
|Cifar10|VggNet|Momentum|
|Cifar100|GoogleNet|Adagrad|
|ImageNet|ResNet|RMSProp|
|||Adam|

두 가지 측면에서 옵티마지어의 성능을 비교합니다.

1. **최종 성능 (Final Performance):** 훈련이 완료된 후 모델의 최종 성능을 평가
2. **수렴 속도 (Convergence Speed):** 각 옵티마이저가 모델을 수렴시키는 데 걸리는 시간
3. 모델의 과적합
4. 학습 속도
5. 안정성
6. 다양성
7. 하이퍼 파라미터 튜닝

## Benchmark datasets

||Mnist|Cifar10|Imagenet|
|---|---|---|---|
|Description|고등학생과 미국 인구조사국 직원들이 손으로 쓴 70,000개의 0부터 9까지의 숫자 이미지를 모은 데이터셋|||
|Resolution|28x28|32x32|224x224|
|Training set, test set|60,000개, 10,000개|50,000개, 10,000개||
|Class|10개|10개||

*스마트폰의 해상도 : 1080x1920

## CNN Models

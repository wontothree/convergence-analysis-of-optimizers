# Convergence analysis of optimizers

## Empirical evaluation

Training loss, Validation accuracy

Optimizer들의 수렴성을 분석하고, Convergence analysis of optimizers에 실험을 진행합니다.

Image classification Experiment

|Datasets|CNN Models|Optimizers|
|---|---|---|
|Mnist|LeNet-5|SGD|
|Fashion MNIST|AlexNet|SGDM|
|Cifar10|VGGNet|Adagrad|
|Cifar100|GoogleNet|RMSProp|
|Imagenet|ResNet|Adam|

## Imagenet

|Mnist|Cifar10|Cifar100|Imagenet|
|---|---|---|---|
|28x28|||224x224|

*스마트폰의 해상도 : 1080x1920

## CNN Architectures

||AlexNet|VGGNet|GoogleNet|ResNet|
|---|---|---|---|---|
|Convolution layers|5|13|21|152|
|Fully connected layers|3|3|1|3(전역 평균 풀링층)|
|Parameter|67M|138M|5M|7000M|
|규제 기법|Dropout|||Batch normalization|
|특징|||Inception module, No FC layer||

## 궁금한 점

옵티마이저의 일반화 능력을 어떻게 정량적으로 평가할 수 있을까?

모든 딥러닝 문제는 non convex problem이다.

따라서 딥러닝에서는 전역 최적화가 아닌 지역 최적화를 수행하게 되며, 이는 문제에 따라 다양한 해결 방법이 적용됩니다.

non convex problem에서 최적해를 얼마나 잘 찾아가느냐가 관건이다.



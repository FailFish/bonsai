# bonsai
21fall system security final project

## Baseline framework

-  Pytorch

## Target model
-   SqueezeNet
-   MobileNet

## Dataset
-   [Cifar10]()
    28x28, 60000 training, 10000 test
-   [Digit-Five](https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA/code_MSDA_digit)
    -   [MNIST(pytorch)]()
        28x28, 60000 training, 10000 test, greyscale
    -   [USPS(pytorch)]()
        16x16, total size: 9298, greyscale
    -   [SVHN(pytorch)]()
        format 1: random shape, format 2: 32x32, total size: 600000, colored
    -   [MNIST-M]()
        28x28, 59001 training, 90001 test, colored
    -   [Synthetic Digits](https://www.kaggle.com/prasunroy/synthetic-digits)
        random image shape, 12000 training, 2000 test, colored

## TODO
-   add validation
-   refine test

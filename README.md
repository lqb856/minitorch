## Introduction

**MiniTorch**

[MiniTorch](https://github.com/minitorch/) is a *diy teaching library* for machine learning engineers who wish to learn about the internal concepts underlying deep learning systems. It is a pure Python re-implementation of the [Torch](http://www.pytorch.org/) API designed to be simple, easy-to-read, tested, and incremental. The final library can run Torch code.

![img](https://minitorch.github.io/figs/Conv/networkcnn.png)

**MiniTorch++**

[MiniTorch++](https://github.com/lqb856/minitorch.git) extend MiniTorch by implement core functionalities in C++ backend and emitted it in Python. So, MiniTorch++ is more efficient while keeping simple and easy to use.

> Note: some functionality is under development....

Besides, MiniTorch++ support multiple device, such as CPU、GPU and Atlas accelerator.



## How to build?

Go to `core/` directory, and build with CMake:

```shell
# 1. create build directory
mkdir build
# configurate project
cmake .. -D-DASCEND_CANN_PACKAGE_PATH="Your CANN Package Path"
# build project
make -j
# try test
./tensor_test
```




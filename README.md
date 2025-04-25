**(Educational Project)** This is a basic Deep Learning framework designed for educational purposes, helping users understand the internal workings of a DL system. The code structure is inspired by [minitorch](https://github.com/minitorch/minitorch), while the implementation details are based on the concepts taught in [LLM System 2025 Spring](https://llmsystem.github.io/llmsystem2025spring/).

## Features
- [x] Automatic differentiation
- [ ] Basic tensor operations:
    - [x] Element-wise addition
    - [x] Element-wise multiplication
    - [x] Matrix multiplication
    - [ ] Broadcasting
    - [ ] Advanced indexing
    - [ ] Sparse tensor support
- [x] Neural network module support
- [x] Optimizers (e.g., SGD)
- [x] Loss functions (e.g., MSE, Cross-Entropy)
- [ ] Decoder-only Transformer Model
    - [ ] Softmax loss function
    - [ ] MultiHeadAttention
    - [ ] FeedForward
    - [ ] DecoderLM
    - [ ] Move softmax and LayerNorm backward to CUDA for optimization
- [ ] Distributed training support
    - [ ] Data Parallel
    - [ ] Pipeline Parallel

## Future features consideration
- [ ] GPU acceleration
- [ ] Advanced optimizers (e.g., Adam)
- [ ] Support for recurrent neural networks
- [ ] Model serialization and loading
- [ ] Visualization tools for training progress
- [ ] Integration with external datasets
- [ ] Pre-trained model support
- [ ] Custom layer creation

## Known issues
* [test_cuda_matmul_transpose](https://github.com/tuananhlfc/minitensor/issues/1)

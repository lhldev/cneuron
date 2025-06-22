# Neural-Network

## Prerequisites
Ensure that a BLAS (Basic Linear Algebra Subprograms) library is installed. This project relies on the C interface to BLAS (cblas.h), which is provided by most major BLAS distributions, including:
- Intel MKL
- OpenBLAS

## For optimal performance, compile the project using the following cmake command:
```
cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTING=OFF
```
## Benchmark - Highest average recorded
- Intel Core i5 9th Gen: ~150,000 Data/s
- Intel Core Ultra 5: ~250,000 Data/s

## This project utilizes the **MNIST dataset**
Information regarding its license (Creative Commons Attribution-ShareAlike 3.0) and attribution can be found in the [data/mnist/MNIST_Copyright.md](data/mnist/MNIST_Copyright.md) file

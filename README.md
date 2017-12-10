# Overview

This is our CUDA GPU implementation of prioritized DCI. The paper can be found at https://arxiv.org/abs/1703.00440.

### Authors

David Tseng, Michael Tong

### Installation Instructions
* clone this repository
* Install a CUDA driver with version > 8.0
* Set `cuda_samples_dir` to the directory in which the CUDA driver samples are located.
* run `make clean && make`
* Output files are found in the bin directory.

### Directory Layout
* `src`, all of the `*.c`, `.cu` files
* `bin`, all of the binaries to run
* `build`, all of the intermediate `*.o` files
* `include`, the header files

### Important Files
* `src/util.cu`, where we have a lot of our matrix multiplication and data generation functions
* `src/dci.cu`, main part of the code where we implement prioritized DCI
* `src/test/test_pipeline.c`, example of a full pipeline of loading data and running a k-NN query

### Example pipeline and tests
After running the make file, you can look at the examples to see how the code is run. To run an example use.

```
$ ./bin/<file_name>.out
```

* `test_pipeline.out`
    * Full example of running a k-nearest neighbors query.
* `matmul_test.out`
    * Tests some simple matrix multiplication operations. Note this calls the host versions of the functions not the device ones.
* `random_matrix_gen.out`
    * Tests the randomized number generation of matrices. This is used in generating random projection vectors for some of the performance benchmarks. 
* `random_normal_data_test.out`
    * Tests the generation of data along a uniform hypercube. This is used in generating random data points for some of the performance benchmarks.
* `test_gen_projection_vectors.out`
    * Tests the generation of data along a uniform hypercube. This is used in generating random data points for some of the performance benchmarks.
* `test_gen_projection_vectors.c`
    * Tests the generation of randomized unit vectors.


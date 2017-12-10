main_files = src/util.cu src/dci.cu
test_files = src/test/matmul_test.c src/test/random_matrix_gen.c src/test/random_normal_data_test.c src/test/test_gen_projection_vectors.c src/test/test_pipeline.c
cuda_samples_dir = /usr/local/cuda-9.0/samples
include_statements = -I include -I $(cuda_samples_dir)/common/inc/
gpu_arch = --gpu-architecture=sm_61
python_flags = -dc -Xcompiler -fPIC
src_files = build/util.o build/dci.o
cuda = -lcublas

all: $(main_files) $(test_files)
	mkdir build
	mkdir bin
	nvcc $(main_files) $(test_files) $(include_statements) $(gpu_arch) $(cuda) -x cu -dc
	mv *.o build
	nvcc $(gpu_arch) $(src_files) $(cuda)  build/matmul_test.o -o bin/matmul_test.out
	nvcc $(gpu_arch) $(src_files) $(cuda) build/random_matrix_gen.o -o bin/random_matrix_gen.out
	nvcc $(gpu_arch) $(src_files) $(cuda) build/random_normal_data_test.o -o bin/random_normal_data_test.out
	nvcc $(gpu_arch) $(src_files) $(cuda) build/test_gen_projection_vectors.o -o bin/test_gen_projection_vectors.out
	nvcc $(gpu_arch) $(src_files) $(cuda) build/test_pipeline.o -o bin/test_pipeline.out

clean:
	-rm -r build bin

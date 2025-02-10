ARCH?=native
NVFLAGS=-std=c++14 -arch=$(ARCH) -O3 --expt-relaxed-constexpr --default-stream legacy 

all: main.out

main.out: main.cu
	nvcc $(CXXFLAGS) $(NVFLAGS) $(INCLUDES) main.cu -o main.out

clean:
	$(RM) main.out

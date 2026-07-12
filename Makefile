# ============================================================
#  PCFG 口令猜测 — 多架构并行实验
#
#  目标:
#    make serial    — 串行基线版本
#    make simd      — SIMD 加速版本 (ARM NEON)
#    make openmp    — OpenMP 多线程版本
#    make mpi       — MPI 分布式版本
#    make gpu       — GPU CUDA 版本 (需要 nvcc)
#    make all       — 编译所有版本
#    make clean     — 清理编译产物
#
#  编译器:
#    CXX  = g++ (串行/SIMD/OpenMP/MPI)
#    NVCC = nvcc (GPU/CUDA)
# ============================================================

CXX      = g++
CXXFLAGS = -std=c++11 -O3 -Wall
MPICXX   = mpicxx
NVCC     = nvcc
NVFLAGS  = -O3 -arch=sm_60

# ---- 公共源文件 ----
COMMON_SRC = common/train.cpp common/md5_serial.cpp

# ---- 各版本目标 ----
all: serial simd openmp mpi

serial: serial/main_serial
simd:   simd/main_neon
openmp: openmp/main_omp
mpi:    mpi/main_mpi

# ---- Serial 版本 ----
serial/main_serial: serial/main_serial.cpp serial/guessing_serial.cpp $(COMMON_SRC)
	$(CXX) $(CXXFLAGS) -Icommon -o $@ $^

# ---- SIMD (ARM NEON) 版本 ----
# 需要 ARM 平台 + NEON 指令集支持
simd/main_neon: simd/main_neon.cpp simd/guessing_neon.cpp simd/md5_neon.cpp $(COMMON_SRC)
	$(CXX) $(CXXFLAGS) -march=armv8-a+simd -Icommon -o $@ $^

# ---- OpenMP 版本 ----
openmp/main_omp: openmp/main_omp.cpp openmp/guessing_omp.cpp $(COMMON_SRC)
	$(CXX) $(CXXFLAGS) -fopenmp -Icommon -o $@ $^

# ---- MPI 版本 ----
mpi/main_mpi: mpi/main_mpi.cpp mpi/guessing_mpi.cpp $(COMMON_SRC)
	$(MPICXX) $(CXXFLAGS) -Icommon -o $@ $^

# ---- GPU (CUDA) 版本 ----
gpu: gpu/main_gpu
gpu/main_gpu: gpu/main_gpu.cu gpu/guessing_gpu.cu gpu/guessing.cu gpu/train.cu gpu/md5.cu
	$(NVCC) $(NVFLAGS) -Igpu -o $@ $^

# ---- 清理 ----
clean:
	rm -f serial/main_serial simd/main_neon openmp/main_omp mpi/main_mpi gpu/main_gpu
	rm -f *.o *.out *.log

.PHONY: all serial simd openmp mpi gpu clean

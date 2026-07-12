# PCFG 口令猜测 — 综合并行化实验

基于**概率上下文无关文法 (PCFG)** 的口令猜测生成器，针对四种并行架构进行了综合实验。

## 选题

**进阶选题：口令猜测 (Password Guessing)**

## 目录结构

```
kouling/
├── common/              # 公共代码
│   ├── PCFG.h           # 基础数据结构 (segment, PT, model, PriorityQueue)
│   ├── train.cpp        # 模型训练（从口令字典学习 PCFG）
│   ├── md5.h            # MD5 宏定义 (F/G/H/I, FF/GG/HH/II)
│   └── md5_serial.cpp   # 串行 MD5 参考实现
├── serial/              # 串行基线版本
│   ├── main_serial.cpp
│   └── guessing_serial.cpp
├── simd/                # SIMD 加速版本 (ARM NEON)
│   ├── main_neon.cpp
│   ├── guessing_neon.cpp
│   └── md5_neon.cpp     # NEON 4路并行 MD5
├── openmp/              # OpenMP 多线程版本
│   ├── main_omp.cpp
│   └── guessing_omp.cpp
├── mpi/                 # MPI 分布式版本
│   ├── main_mpi.cpp
│   └── guessing_mpi.cpp
├── gpu/                 # GPU CUDA 版本
│   ├── main_gpu.cu
│   ├── guessing_gpu.cu  # CUDA kernel 实现
│   ├── guessing.cu
│   ├── train.cu
│   ├── md5.cu
│   ├── md5.h
│   └── PCFG.h
├── scripts/             # 运行脚本
│   ├── qsub.sh          # PBS 集群提交脚本
│   └── test.sh
├── Makefile
└── README.md
```

## 编译与运行

### 串行版本
```bash
make serial
./serial/main_serial [训练集路径]
```

### SIMD 版本 (ARM NEON)
```bash
make simd
./simd/main_neon [训练集路径]
```

### OpenMP 版本
```bash
export OMP_NUM_THREADS=4
make openmp
./openmp/main_omp [训练集路径]
```

### MPI 版本
```bash
make mpi
mpirun -np 4 ./mpi/main_mpi [训练集路径]
```

### GPU 版本
```bash
make gpu
./gpu/main_gpu [训练集路径]
```

## 算法概述

1. **训练阶段**：从口令字典中解析每条口令的字符类型序列（如 `password123` → `L8D3`），统计 PT 模板概率和 segment value 频率。

2. **猜测生成**：用优先队列按概率降序生成猜测。每次弹出概率最高的 PT，枚举其最后一个 segment 的所有 value 形成具体猜测。

3. **并行化热点**：`Generate()` 函数中枚举 segment values 的 for 循环是主要并行目标。

## 并行化策略对比

| 架构 | 并行粒度 | 加速目标 | 关键技术 |
|------|---------|---------|---------|
| Serial | — | 基线参考 | 串行 MD5 |
| SIMD  | 4 路数据并行 | MD5 哈希 | ARM NEON uint32x4_t |
| OpenMP | 线程级 | 猜测生成 + MD5 | #pragma omp parallel for |
| MPI   | 进程级 | 猜测生成 | MPI_Gatherv 汇总 |
| GPU   | 线程级 (>1000) | 猜测生成 | CUDA kernel |

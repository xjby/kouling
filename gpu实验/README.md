# GPU 实验 — PCFG 口令猜测 GPU 并行化

## 环境要求
- NVIDIA GPU + CUDA Toolkit 12.x
- 或 AMD GPU + ROCm HIP SDK

## 编译

### NVIDIA CUDA
```bash
nvcc -std=c++17 -O2 main_gpu.cpp guessing.cpp guessing_gpu.cu train.cpp md5.cpp -o main
```

### AMD HIP
```bash
hipcc -O2 main_gpu.cpp guessing.cpp guessing_gpu.hip train.cpp md5.cpp -o main
```

## 运行
```bash
./main train_data.txt 500000
```
- 参数1：训练文件路径
- 参数2：生成口令数量上限

## 文件说明
| 文件 | 说明 |
|:-----|:-----|
| main_gpu.cpp     | GPU 版主程序 |
| guessing.cpp     | PCFG 优先队列逻辑（GPU 版 PopNext） |
| guessing_gpu.cu  | CUDA kernel（口令生成） |
| guessing_gpu.hip | HIP kernel（AMD ROCm 兼容版） |
| PCFG.h           | PCFG 模型头文件 |
| train.cpp        | 模型训练 |
| md5.cpp / md5.h  | MD5 哈希实现 |
| Final_Report.tex | 实验报告 LaTeX 源文件 |

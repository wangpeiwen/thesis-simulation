# thesis-simulation

毕设论文模拟实验项目：基于 simdistserve 离散事件模拟器，对比 LLM 推理服务的五种调度方案。

## 对比方案

| 方案 | 说明 |
|------|------|
| Disagg-Static | 纯 PD 分离（DistServe），P:D=2:6，节点角色固定 |
| Coloc-Sarathi | 纯共置（Sarathi-Serve），chunked prefill，chunk=512 |
| CBS-NoMig | CBS 逐请求决策，无迁移 |
| CBS-NoRole | CBS + 双向迁移，无角色自适应 |
| CBS-Full | CBS + 迁移 + 角色自适应（完整系统） |

## 负载模式

| 负载 | 输入长度 | 输出长度 | 到达过程 |
|------|---------|---------|---------|
| uniform | [128, 2048] | [32, 512] | 泊松 |
| bursty | [128, 2048] | [32, 512] | 泊松 + 每30s出现5s的4x突发 |
| longctx | [1024, 8192] | [64, 256] | 泊松 |

## 安装

```bash
pip3 install -r requirements.txt
```

## 运行实验

### 完整实验

5 方案 × 3 负载 × 8 到达率 = 120 组：

```bash
python3 experiments/run_all.py \
    --workload all \
    --rates 2,4,6,8,10,12,14,16 \
    --sim-time-limit 120 \
    --output results/full.csv
```

### 单负载实验

```bash
python3 experiments/run_all.py \
    --workload uniform \
    --rates 2,4,6,8,10,12,14,16 \
    --output results/uniform.csv
```

### CBS 消融实验

```bash
python3 experiments/run_all.py \
    --variants cbs-nomig,cbs-norole,cbs-full \
    --rates 4,8,12 \
    --output results/cbs_ablation.csv
```

### CBS 参数调优

```bash
python3 experiments/run_all.py \
    --cbs-mu 1.0 --cbs-lambda 0.5 \
    --rates 8,12 \
    --output results/param_sweep.csv
```

## 生成图表

```bash
python3 experiments/plot_results.py \
    --input results/full.csv \
    --output-dir figures/ \
    --rate-for-bars 12.0
```

输出：
- `figures/goodput_*.pdf` — Goodput vs 到达率曲线
- `figures/slo_*.pdf` — SLO 达标率曲线
- `figures/latency_*.pdf` — P99 尾时延柱状图

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n-prefill` | 2 | Prefill 实例数 |
| `--n-decode` | 6 | Decode 实例数 |
| `--tp` | 1 | 张量并行度 |
| `--slo-ttft` | 2000 | TTFT SLO (ms) |
| `--slo-tpot` | 100 | TPOT SLO (ms) |
| `--sim-time-limit` | 600 | 模拟时长 (s)，调小可加速 |
| `--cbs-mu` | 0.5 | CBS 风险惩罚权重 |
| `--cbs-lambda` | 0.1 | CBS 外部性权重 |
| `--cbs-kappa` | 0.1 | CBS dispatch 争抢系数 |
| `--kv-latency` | 5.0 | KV Cache 传输延迟 (ms) |
| `--model` | facebook/opt-13b | 模型（需有对应 profile 数据） |

## 项目结构

```
thesis-simulation/
├── simdistserve/              # 模拟器（含 CBS 扩展）
│   ├── base/
│   │   ├── cbs_scheduler.py   # CBS 评分 + 迁移 + 角色自适应
│   │   ├── cbs_worker.py      # 干扰时间膨胀
│   │   ├── scheduler.py       # 基础调度器
│   │   ├── worker.py          # 基础 Worker
│   │   └── request.py         # 请求定义
│   ├── clusters/
│   │   ├── cbs.py             # CBS 混合集群
│   │   ├── disagg.py          # 分离集群
│   │   └── vllm.py            # 共置集群
│   └── estimators/
│       ├── interference_model.py  # 干扰系数模型
│       └── time_estimator.py      # Prefill/Decode 时间模型
├── experiments/
│   ├── run_all.py             # 主实验脚本
│   ├── plot_results.py        # 绘图脚本
│   └── configs/               # 负载配置
├── results/                   # 实验输出 CSV
└── figures/                   # 生成的图表
```

# Bagel 两机 Yuanrong Connector PR 测试记录与对比方案

## 1. 文档目的

本文档记录 Yuanrong TransferEngineConnector 的两机 Bagel 验证结果、1 GiB
跨节点传输 benchmark，以及 SharedMemoryConnector 基线测试方案，内容可
直接整理到 PR 的 `Test Plan` 和 `Test Result` 部分。

## 2. 测试对象

- 模型：`ByteDance-Seed/BAGEL-7B-MoT`
- 目标 connector：`YuanrongTransferEngineConnector`
- 对照 connector：`SharedMemoryConnector`
- 可选参考：`MooncakeTransferEngineConnector`
- vLLM-Omni commit：`346e5790f4e3bde3619a75a2ea26b2c89cde8a6b`

`SharedMemoryConnector` 只能作为同机 baseline。两台物理机之间不能在不
使用任何 connector 的情况下传递 stage 0 到 stage 1 的中间结果；因此
“没有 TEConnector”在本文中具体表示“不使用 RDMA TransferEngine，而使用
同机 SharedMemoryConnector”。

## 3. 已完成的两机 Bagel 结果

测试结果为 PASS：

- HTTP 请求：`3/3` 成功
- benchmark wall time：`60.603 s`
- image throughput：`0.0495 requests/s`
- connector receive：`3` 次
- 总 payload：`3.076 MiB`
- fast path / zero-copy：`3/3`
- connector receive 总耗时：`25.50 ms`
- effective end-to-end throughput：`120.61 MiB/s`
- read-path throughput：`199.71 MiB/s`

首次传输存在冷启动开销：

- 第一次：`16.5 ms`
- 第二次：`4.2 ms`
- 第三次：`4.8 ms`

这组数据用于证明两机 Bagel 和 Yuanrong connector 的功能路径可用，不应
作为稳定态高吞吐性能结论。

## 4. 已完成的 1 GiB 跨节点传输 benchmark

### 4.1 Yuanrong

- 成功：`30/30`
- 失败：`0/30`
- 总传输量：`30,720 MB`
- 传输时间：`3.44 s`
- producer throughput：`8,935.27 MB/s`
- connector initialize：`9,686.5 ms`
- connector put 平均耗时：`1.4 ms`
- consumer round trip：平均 `111.6 ms`，P50 `111.1 ms`，P95 `111.8 ms`
- 单次 payload：`1,073,741,824 bytes`
- `[YR GET]` read：约 `108.4–108.6 ms`
- `[YR GET]` total：约 `109.8–110.2 ms`
- 接收带宽：约 `9,293.3–9,322.1 MB/s`
- 路径：`fast_path, zero-copy`

注意：throughput 统计应明确不包含 connector 初始化时间；初始化耗时应
单独报告。

### 4.2 Mooncake 参考结果

- 成功：`20/20`
- 失败：`0/20`
- 总传输量：`20,480 MB`
- 传输时间：`2.01 s`
- consumer throughput：`10,186.13 MB/s`
- RDMA receive latency：约 `94.5–94.7 ms`
- 接收带宽：约 `10,814–10,858 MB/s`
- 路径：`fast_path, zero-copy`

两次 benchmark 的传输次数不同，且当前记录未包含完整的并发、硬件和
网络配置，因此只能作为独立的成功参考，不能直接宣称 Yuanrong 与
Mooncake 的严格性能优劣。

## 5. SharedMemory baseline

新增脚本参数：

```text
--connector shared_memory
```

该模式生成如下 stage connector：

```yaml
connectors:
  transfer_engine_connector:
    name: SharedMemoryConnector
```

运行时必须保证 stage 0 和 stage 1 在同一台物理机上，例如：

```bash
python3 tests/distributed/omni_connectors/run_bagel_yuanrong_two_node.py \
  --connector shared_memory \
  --model /path/to/BAGEL-7B-MoT \
  --stage0-host <same-host> \
  --stage1-host <same-host> \
  --stage0-ip <same-host-ip> \
  --stage1-ip <same-host-ip> \
  --run-dir /tmp/vllm-omni-bagel-shm
```

该模式的结果应主要比较：

- Bagel 请求是否成功；
- wall time 和请求延迟；
- SharedMemory 的 stage 间传递开销；
- 与 Yuanrong 两机结果相比增加的网络传输成本。

SharedMemory 模式不会生成 Yuanrong `[YR GET]` 记录，也不应填写 RDMA
throughput。脚本生成的 summary 会将这些字段标记为 `not applicable`。

## 6. 建议的 PR 测试矩阵

| 场景 | 机器 | connector | 请求/传输数 | 目的 |
| --- | --- | --- | ---: | --- |
| Bagel smoke | 两机 | Yuanrong | 3 请求 | 验证端到端功能 |
| Bagel baseline | 同机 | SharedMemory | 3 请求 | 排除 Bagel 自身问题并获得无 RDMA 基线 |
| 大 payload | 两机 | Yuanrong | 30 × 1 GiB | 验证大数据量、内存池复用和稳定性 |
| 参考实现 | 两机 | Mooncake | 20 × 1 GiB | 提供已有 TransferEngine 的参考结果 |

## 7. PR 中需要补充的环境数据

- 两台机器的 GPU 型号和数量；
- RDMA HCA 名称、链路类型和理论带宽；
- Python、CUDA、PyTorch、vLLM 和 Yuanrong package 版本；
- benchmark 是否包含 warmup；
- producer/consumer 的并发配置；
- 完整 benchmark 命令；
- `run-metadata.json` 中的实际 source commit。

## 8. 当前已知限制

Bagel 运行的 Prometheus 快照曾出现以下问题：

- `transfer_size_bytes_sum` 为 `0`，但 connector 日志中有非零 payload；
- `transfer_rx_s` 没有样本；
- `transfer_in_flight_s` 没有样本。

因此 PR 中的 payload 和接收带宽应以 connector benchmark summary 以及
`[YR GET]` 日志为准，不能使用错误的 Prometheus payload sum 作为数据源。

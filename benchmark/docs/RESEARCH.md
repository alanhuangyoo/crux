# Terminal-Bench 竞争格局调研

> 调研日期：2026-08-25。所有数字来自当日官方榜单与一手仓库，非二手报道。

## 1. Benchmark 版本与提交状态

Terminal-Bench 现在是一个**持续演进的 benchmark 系列**，harness 已从旧的 `tb` CLI
迁移到 [Harbor](https://github.com/harbor-framework/harbor)。

| 版本 | 状态 | 社区提交 | 备注 |
|---|---|---|---|
| `terminal-bench@latest` (TB3 / Frontier) | 现役主战场 | ✅ 开放 | 榜单托管在 Harbor Hub |
| terminal-bench 2.0 | Live | ✅ 开放 | 142 条目 |
| terminal-bench 2.1 | Live | ❌ **关闭** | 仅维护者可提交 |
| terminal-bench 1.0 | Legacy | — | 80 tasks |
| TB-Science 1.0 | Coming soon | — | 科学计算专项 |
| TB Challenges | Live | ✅ | 单任务长跑，交仓库+日志 |

TB 2.1 关闭的原文（`leaderboard/SUBMIT.md`）：

> Community submissions are currently closed for Terminal-Bench 2.1.
> Only submissions run by the maintainers will be added to the leaderboard at this time.

## 2. 两个口径不同的榜单

跨榜单的分数**不可直接比较**——sandbox 与 harness 都不同。

| | 官方 tbench.ai | Artificial Analysis（第三方） |
|---|---|---|
| 条目 | 17 (2.1) / 142 (2.0) | 208 models |
| harness | 官方 Modal | Terminus 2 + e2b sandbox |
| 榜首 | Claude Code + Fable 5 **83.8%** | GPT-5.6 Sol **89.5%** |

官方 TB 2.1 榜（截至 2026-08-25）：

| # | Agent | Model | Score | Cost |
|---|---|---|---|---|
| 1 | Claude Code | Fable 5 | 83.8% ± 1.2% | $552.67 |
| 2 | Codex | GPT-5.5 | 83.1% ± 1.1% | $2,059.19 |
| 3 | Terminus 2 | Fable 5 | 80.4% ± 1.2% | $438.64 |
| 4 | Cursor CLI | Grok 4.5 | 79.3% ± 1.5% | $134.09 |
| 8 | mini-SWE-agent | Muse Spark 1.1 | 76.2% ± 1.2% | $198.05 |

## 3. 核心发现：DeepSeek 在官方榜上几乎空白

| 榜单 | 总条目 | DeepSeek 条目 |
|---|---|---|
| TB 2.0 | 142 | **仅 1 条**：Terminus 2 + DeepSeek-V3.2 = **39.6%**（rank 86, 2026-02-10） |
| TB 2.1 | 17 | **0 条** |

而第三方评测的现役 DeepSeek：**V4 Pro 0813 ≈ 87.9%**、**V4 Flash 82.7%**。

**结论**：官方榜记录的 DeepSeek 停留在半年前的 V3.2/39.6 分，实际实力已是 80+。
没有人把现役 DeepSeek 提交到官方榜。这是本项目的目标生态位。

## 4. 提交流程与硬性红线

```
harbor run --upload --public
        ↓
lb filter → lb metadata → lb open-prs
        ↓
CI 静态分析 → 自动提升为 bot PR → /judge 轨迹审计 → /apply → merge → 上榜
```

CI 会**自动拒绝**以下情况：

- 数据集版本非 pinned 版本
- `timeout_multiplier` 非 unset/1.0；任何 agent/verifier/setup timeout override
- 任何资源 override（cpus/memory/storage/gpus）
- 未覆盖全部 task，或任一 task 少于 **5 trials**
- 剔除 errored trial（errored 必须计 reward 0）
- **成功 trial 缺少 ATIF trajectory** —— 只上传自定义日志的 agent 在此直接 fail

`/judge` 阶段由 LLM 逐条审计成功轨迹：

- **Harness cheating 🔴** → 整个提交作废
- **Reward hacking 🟡** → 该 trial 计 0，可申诉
- Refusals ⚪ / No trajectory ⚫ → 信息性

历史处罚案例（[Leaderboard Integrity Update](https://www.tbench.ai/news/leaderboard-integrity-update)）：
OpenBlock OB-1（改 timeout + binary 内置加密答案）、QuantFlow Pilot（上传 task 的 test 目录）、
ForgeCode（联网检索答案）——三者均被移除。

**设计约束**：`SUPPORTS_ATIF = True` 是上榜的必要条件，必须在 scaffold 第一版就实现。

## 5. 技术方向

[Tmax: A simple recipe for terminal agents](https://arxiv.org/pdf/2606.23321) 的核心结论是
**架构创新不重要，工程细节才重要**。五个杠杆：

1. **Trajectory-based learning** —— 成功轨迹做 in-context demonstration
2. **Test-time compute 分配** —— interleaved reasoning、rollout
3. **Prompt 工程** —— 明确任务目标与输出格式
4. **Tool 抽象** —— 干净的文件/命令接口减少错误
5. **Context 管理** —— 决定放什么、丢什么

官方自己的数据佐证 scaffold 的价值：同一个 Gemini 2.5 Pro，换用 Terminus 2 scaffold
比 OpenHands 提升 **17%**。

## 6. 成本结构

榜上单次评测 $134–$2,059。DeepSeek 单价远低于 Fable 5 / GPT-5.5，
在同等预算下**可负担的迭代轮次高出一个数量级**——而 scaffold 调优恰恰是迭代密集型工作。
这是本项目相对大厂提交的结构性优势。

## 7. 参考资料

- [Terminal-Bench 官网](https://www.tbench.ai/) · [TB 2.1 榜](https://www.tbench.ai/leaderboard/terminal-bench/2.1) · [TB 2.0 榜](https://www.tbench.ai/leaderboard/terminal-bench/2.0)
- [harbor-framework/terminal-bench](https://github.com/harbor-framework/terminal-bench) · [harbor](https://github.com/harbor-framework/harbor)
- [官方 SUBMIT.md](https://github.com/harbor-framework/terminal-bench-2-1/blob/main/leaderboard/SUBMIT.md)
- [Leaderboard Integrity Update](https://www.tbench.ai/news/leaderboard-integrity-update)
- [Artificial Analysis TB v2.1](https://artificialanalysis.ai/evaluations/terminalbench-v2-1)
- [Tmax 论文](https://arxiv.org/pdf/2606.23321)

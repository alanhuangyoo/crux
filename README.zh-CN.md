<h1 align="center">Crux</h1>

<p align="center">
  <b>一个针对 Terminal-Bench 调优的编码 Agent，以及决定什么改动能进去的那套测量装置。</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Terminal--Bench_2.1-0.773_pass@1-2ea44f" alt="pass@1 0.773">
  <img src="https://img.shields.io/badge/tests-4%2C339_passing-2ea44f" alt="4339 tests">
  <img src="https://img.shields.io/badge/base-pi_(Earendil_Works)-blue" alt="forked from pi">
  <img src="https://img.shields.io/badge/model-self--hosted_Qwen3.8--27B-8a2be2" alt="self-hosted model">
</p>

<p align="center">
  <a href="README.md">English</a> · 简体中文
</p>

---

Crux 是 [pi](https://pi.dev) 的一个 fork，改动集中在 agent 循环和它的工具上，
并且把用来在 [Terminal-Bench](https://www.tbench.ai/) 2.1 上衡量这些改动的
harness 一起放在同一个仓库里。

放在一起是有意的：**下面每一条改动都挂着促成它的那个测量，每个数字旁边都记着
是哪一轮跑出来的**。`packages/` 是应用了改动的 pi 代码，`benchmark/` 是 harness。
CLI 同时以 `crux` 和 `pi` 安装。

## 结果

自部署 Qwen3.8-27B，Terminal-Bench 2.1 中 89 道不需要容器内 GPU 的任务，
8× agent 预算，pass@1 按整轮计算。

| 配置 | pass@1 |
|---|---:|
| **crux** — 262,144 上下文窗口 | **0.773** &nbsp;<sub>（第二次跑 0.793）</sub> |
| crux — 同一份代码，32,768 窗口 | 0.678 |
| crux — 修复压缩与挂死之前 | 0.591 |
| Claude Code，同一模型 | 0.730 |
| pi 原版 | 0.539 |

Claude Code 那一行是早一轮在 32,768 窗口下测的，之后没有重跑，所以它和第一行的
对比同时混入了脚手架差异和部署参数差异。

与 32K 那一轮在两边都评分的 86 道上做配对：分歧 16 道中 **12 比 4**，
符号检验 z = +2.00。

获胜配置跑了两次，分别是 0.773 和 0.793，在 86 道里有 13 道结果不同（7 比 6）
——**这就是这个 benchmark 在 89 道规模下的噪声底**，也是这里每一个结论都用配对
比较、而不是比总分的原因。

<details>
<summary><b>这些提升从哪来</b></summary>

<br>

大致一半来自把坏掉的东西修好，另一半来自两个**被当成机器属性、其实是人定的部署参数**：

| | |
|---|---|
| 0.591 → 0.678 | 压缩的请求装不下它自己，以及 trial 有 93% 的墙上时钟是挂死的 |
| 0.678 → 0.773 | 端点以 `--context-length 32768` 服务一个声明 262,144 的模型；8 路并发让引擎每次只批处理一两个请求 |

后者顺带把一轮全量从 6 小时压到 2 小时。

</details>

## 改了什么

每条改动都挂着促成它的测量，没有一条是凭空推理出来的。

### 上下文与压缩

| 改动 | 促成它的测量 |
|---|---|
| 摘要请求必须装得进它所在的窗口 | 压缩把历史当输入、摘要当输出，两者出自同一个窗口，而代码从不检查两者之和：**2,370 次压缩里 2,115 次失败**，每次只追加一条错误、什么都没裁掉 |
| 被拒绝就把请求砍半重试 | 字符/token 是文本的属性——散文 3.99，压缩输出 1.95——没有哪个常数装得下所有情况；一次拒绝本身就是信息 |
| 写不完的摘要也要留下 | 在 200K 窗口上丢弃它是对的，在 32K 上等于禁用压缩，而那恰恰是常态 |
| keep 预算被末尾的工具结果吃光时仍然要裁 | 切点永远不是工具结果，所以一条巨大的末尾结果会让搜索无处可切，压缩于是保留了全部 |

### 存活性

| 改动 | 促成它的测量 |
|---|---|
| 连上之后不再发数据的流要判失败 | SDK 的超时管"拿到响应"，不管"保持响应"：**25 道超时的 trial，在被杀之前中位沉默了 121 分钟里的 112 分钟** |
| 没写超时的命令也有超时 | 取十分钟，因为这些任务要编译；否则一条 `grep -rl ... /` 会一直跑到 trial 结束 |
| 后台进程不能把已结束的命令一直挂住 | 退出后的宽限期每来一块数据就重置，于是一个脱离的子进程往继承的管道里写就能无限续命——有一道题**干了 48 秒之后占着容器挂了 8 小时** |

### 循环

| 改动 | 促成它的测量 |
|---|---|
| 什么都没回答的一轮不算 agent 结束 | `regex-chess` 把整个输出预算花在思考上、以 `length` 收尾，而这一轮被记录成"已完成且未采取任何行动" |
| 截断轮次的两段式恢复 | 先静默抬高上限重试；只有再次截断才告诉模型 |
| 循环能看见的截止时间 | 24 道失败里有 6 道是带着未完成的工作被切断的，而解出题的中位数只用掉了预算的一小部分 |
| 预算没花完时质疑"停下" | 失败的 trial 停下时中位只花了 24% 的预算，Claude Code 是 95% |
| 被截断的一轮不能买光所有预算提示 | 把截断当作"又挣到一条提示"，等于为反复发生的那种情况拆掉了唯一的刹车，一个反复截断的 run 会把 40 条全部用光 |
| 被杀死的 run 要续跑，而不是记零分 | cgroup 的 OOM kill 会带走进程组里的所有进程，于是一道耗尽内存的任务直接结束整轮 |

### 工具

| 改动 | 促成它的测量 |
|---|---|
| 单次响应要有默认上限 | Claude Code 的 p99 输出是 4,911 token，对着 8K 的上限——**重要的是这个差距，不是上限本身** |
| 失败的编辑要给出文件内容，而不是一条规则 | "必须精确匹配"让模型无路可走；528 次编辑里失败 43 次，而同期只读了 156 次 |

## 方法

不同 arm 之间按题配对做符号检验，不比总分，而且只比较**同时开跑**的
arm——有一次 p=0.041 的结论，在同一时间窗里重跑两个 arm 之后变成了 p=0.453。
按实测 15.5% 的翻转率，89 道题需要 11 个点的差距才能靠自己达到 p&lt;0.05，
所以 8 个点的效应必须靠复现来确立。

**调一个机制之前，先数它成功了几次。** 压缩的 reserve 和保留深度被调了好几天，
才有人问它的成功率是多少——答案是 14%。

`benchmark/docs/` 保留了所有被测量并否决掉的候选解释，而那是其中的大多数
——目前二十四个，其中好几个是已经写完了的设计。

## 目录结构

```
packages/          pi 的 fork —— agent 核心、模型层、TUI、编码 agent CLI
benchmark/         harness
  src/crux/        harbor agent、提示词分段、端点工具
  scripts/         preflight、配对比较、机制健康检查
  docs/            测量记录，包括被排除掉的那些
  tests/           464 个测试
```

## 跑这个 benchmark

```bash
cd benchmark && uv sync
harbor run --dataset terminal-bench/terminal-bench-2-1 \
  --agent crux.pi_agent:CruxPiAgent --model openai/<model>
```

`benchmark/scripts/preflight.sh` 会在开跑前检查端点、启动脚本和 harness 校验和。
评测主机的情况见 `benchmark/docs/ENVIRONMENT.md`。

## 上游

pi 由 [Earendil Works](https://pi.dev) 开发，是这个项目得以存在的基础；
见 [`AGENTS.md`](AGENTS.md) 和 [`CONTRIBUTING.md`](CONTRIBUTING.md)。
上游的 issue 和 PR 请提到
[earendil-works/pi](https://github.com/earendil-works/pi)，不要提在这里。

| 包 | 说明 |
|---------|-------------|
| **[@earendil-works/pi-coding-agent](packages/coding-agent)** | 交互式编码 agent CLI（安装为 `crux` 和 `pi`） |
| **[@earendil-works/pi-agent-core](packages/agent)** | 带工具调用与状态管理的 agent 运行时 |
| **[@earendil-works/pi-ai](packages/ai)** | 统一的多provider LLM API |
| **[@earendil-works/pi-tui](packages/tui)** | 终端 UI 组件 |
| **[@earendil-works/chord](packages/chord)** | 服务、RPC 与插件的应用组装运行时 |

许可证与上游一致，见 [`LICENSE`](LICENSE)。

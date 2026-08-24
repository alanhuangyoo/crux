# 执行环境

## 结论：评测跑在 dev 开发机（x86-64），不在本机 Mac

| | 本机 Mac | dev 开发机 |
|---|---|---|
| 架构 | arm64 ❌ | **x86_64** ✅ |
| 系统 | macOS | Ubuntu 24.04.4 LTS |
| CPU / 内存 | — | 16 核 / 29 GB |
| 磁盘可用 | — | 1.9 TB |
| Docker | arm64 | **amd64/linux** ✅ |
| 容器内 apt | 502 ❌ | **rc=0** ✅ |

评测一律在 dev 上跑。本机只写代码，用 rsync 同步过去。

## 本机 Mac 为什么不行

2026-08-25 的 oracle 冒烟（2 任务，`--env docker`）**两个都失败**。
oracle 跑的是参考解，本应满分——失败即环境问题。两个根因不同：

### 1. 代理劫持 DNS

```
E: Failed to fetch http://deb.debian.org/debian/dists/bookworm/InRelease
   502 Bad Gateway [IP: 198.18.0.212]
```

`198.18.0.0/15` 是 RFC 2544 保留段，本机代理（`127.0.0.1:7897`）的 fake-IP 模式
把 `deb.debian.org` 解析到了这里，Docker 构建拿不到 Debian 源。

### 2. 架构不匹配（本地无解）

```
ico_patched: ELF 64-bit LSB executable, x86-64
本机 docker : arm64/linux
```

任务分发 x86-64 原生二进制，本机 Docker 是 arm64，被测服务起不来：

```
AssertionError: service did not accept connection on port 50373: [Errno 111] Connection refused
17 failed, 2 passed
```

通过的两个测试恰好只做文件校验、不执行该二进制。Rosetta 模拟虽可跑 amd64 容器，
但慢且与官方环境不一致——分数不可比，调优会被架构噪声污染。

## dev 机器为什么可以

- **x86_64** —— 与官方榜单环境一致，分数可比
- **DNS 正常** —— `deb.debian.org` 解析到真实 IP（fastly），无劫持；容器内 `apt-get update` rc=0
- **16 核 / 29 GB / 1.9 TB** —— 足够跑 89 tasks × 5 trials 所需的并发与镜像存储

Terminal-Bench 官方用 Modal 跑 CI 和榜单实验。dev 机器满足同样的架构与网络前提，
本阶段无需引入 Modal；若后续并发成为瓶颈，再评估 `--env modal`。

## 工作流

```bash
# 本机写代码 → 同步到 dev → 在 dev 上跑
rsync -az --delete --exclude 'jobs/' --exclude '.venv' ./ dev:~/crux/
ssh dev 'cd ~/crux && ./scripts/smoke.sh'
```

dev 上已安装：`uv 0.12.5`、`harbor 0.22.0`。

## h20-43 的容器出网

H20 子网到 Fedora 的镜像管理服务完全不通，而 jump 主机完全通得上：

```
mirrors.fedoraproject.org/metalink 可达率
  h20-43   0/10      h20-44  0/6      h20-45  0/6
  jump    10/10      B300-1  6/6      B300-7  6/6
```

这不是我们能忽略的：任务 `retro-console-soc` 的镜像基于 `fedora:42`，构建时
`dnf install` 直接失败，整个 trial 记为 error —— 而 error 按 **reward 0** 计且
不允许剔除。在 12 任务的交叉验证里，dev 拿到 12/12 而 h20-43 只有 11/12，
差的就是这一个。

**解法**：只把 `.fedoraproject.org` 的流量经 jump 转发，其余保持直连——
让所有流量都走 jump 会把镜像拉取挤到单点上。

```
容器 → privoxy (172.17.0.1:8888) ─┬─ .fedoraproject.org → SOCKS(127.0.0.1:1080) → jump
                                  └─ 其它一切          → 直连
```

两个 systemd 单元，均 `Restart=always`：

| 单元 | 作用 |
|---|---|
| `crux-jump-socks` | `ssh -N -D 127.0.0.1:1080` 到 jump，提供 SOCKS 出口 |
| `crux-proxy` | privoxy，按域名分流；容器侧入口 `172.17.0.1:8888` |

Docker 通过两处配置使用它——缺一不可，前者管运行期容器，后者管 `docker build`：

- `/etc/docker/daemon.json` 的 `proxies`
- `/root/.docker/config.json` 的 `proxies`（BuildKit 从这里取构建期代理）

用了 privoxy 而非 tinyproxy：Ubuntu 24.04 的 tinyproxy 1.11.1 只支持
`upstream http`，无法把上游指向 SOCKS 隧道。

**验证**：修复后 `mirrors.fedoraproject.org` 容器内 6/6 可达，`dnf install` 通过，
apt 无回归，`retro-console-soc` 从 error 变为 reward 1.0。

## GPU 任务：当前被结构性阻断

74 个任务里有 4 个声明 `gpus=1`：`exam-pdf-eval`、`fp8-rmsnorm-gemm`、
`jax-speedrun-gpu`、`math-eval-grader`。Docker 未配置 nvidia runtime，这些任务在
环境校验阶段就抛 RuntimeError：

```
Task requires 1 GPU(s) but EnvironmentType.DOCKER environment does not
support GPU allocation.
```

errored trial 按 **reward 0** 计且不允许剔除，所以**当前分数上限是 70/74 = 94.6%**。

这不是疏忽，是权衡的结果。h20-43 的 8 张卡正被训练任务占满（h20-44/45 同样），
装上 nvidia-container-toolkit 并让评测任务申请 GPU，会直接和训练抢卡。
在训练让出卡之前，宁可让这 4 个任务记 0 分。

正式提交前必须解决这一项——4 分在榜单上不是小数目。可行路径是等训练结束后，
或换一台 GPU 空闲的节点，单独把这 4 个任务补跑。

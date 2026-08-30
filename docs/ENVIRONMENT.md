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

## Docker 存储驱动：必须用 btrfs，不能用 overlayfs

第一次全量跑（70 任务）**全部失败**，mean 0.000。67 个 trial 抛 RuntimeError，
只有 4 个真正执行到 agent。两类症状：

```
45 个: apt-get 构建失败 —— GPG error: At least one invalid signature was encountered
22 个: Failed to start tmux session
```

两者是同一个根因。tmux 不在任务镜像里，`TmuxSession` 会自己用 apt 装 ——
所以 apt 一坏，tmux 也装不上。

真凶是 **`/scratch` 是 btrfs，而 Docker 默认用 overlayfs 驱动**。
overlayfs 叠在 btrfs 上会破坏写入内容，apt 下载的 InRelease 签名文件因此校验失败。
时钟正常、无代理残留、磁盘和内存都充裕，全都排除掉之后才定位到这里。

```json
{ "data-root": "/scratch/docker", "storage-driver": "btrfs" }
```

改用原生 btrfs 驱动后：GPG 错误 0，`apt-get install tmux` 成功。

⚠️ **换驱动会丢弃已有镜像**（存储布局不同），首次运行需重新拉取和构建。

### 一段弯路

定位过程中我一度以为是自己加的 privoxy 代理导致的（它确实是可疑对象——
全局代理有可能破坏下载）。撤掉代理后 GPG 错误依旧，才继续往下查到文件系统。
代理最终也没保留：它只为 1 个 fedora 任务而设，风险面却覆盖全部任务，不划算。

## oracle 在 TB 2.1 上的复核：22/24，两个失败都是任务腐化

「分数低是不是环境有问题」只有一个直接的答法：跑 oracle。它执行任务自带的
参考解，不调模型，所以满分意味着容器、判分、数据都是好的，不满分就是环境
坏了。此前那次 oracle 全绿是在旧机器、旧数据集（74 任务）上做的，
**TB 2.1 加 h20-45 从没验过**。

25 个任务，**22 个满分，2 个失败**。逐个查完根因，两个都不归咎于本部署：

**`build-pov-ray`** — 参考解正确安装了 build-essential、gcc、make、wget（日志
里全部 Setting up 成功），然后去取源码：

```
https://www.povray.org/ftp/pub/povray/Old-Versions/Official-2.2/POVDOC.TAR.Z
HTTP request sent, awaiting response... 403 Forbidden
```

DNS 解析成功、TLS 握手成功、连接建立成功，**服务器主动拒绝**。任务依赖一个
1993 年的站点仍然愿意提供下载。

**`mcmc-sampling-stan`** — 参考解锁了 `StanHeaders 2.32.10`，却让传递依赖
`RcppParallel` 浮动：

```
RcppParallel (NA -> 6.2.1) [CRAN]
error: RcppParallel requires cmake (>= 3.5); cmake was not found
```

装到最新的 6.2.1，而新版要求 cmake，镜像的 Dockerfile 里没有，参考解也没装。
**任务写的时候能跑，上游发新版之后就编不过了。**

两者都是**外部世界随时间变化**造成的，今天任何人跑都会失败。它们对总分的影响
约 2 个百分点，解释不了 63.5% 与模型卡 73.0 之间的差距。

### 这次复核同时确认的

| 项 | 结果 |
|---|---|
| 数据集 | 官方包，sha256 锁定 `7d7bdc1c…` |
| 框架 | harbor 0.22.0 官方版 |
| 判分 | 任务自带 `test.sh` 加 pytest，agent 阶段不在容器里 |
| 每任务超时 | 无 override、无 multiplier，用数据集自带值 |
| apt 与网络 | 参考解装了几十个包全部成功 |
| **参考解** | **22/24 满分** |

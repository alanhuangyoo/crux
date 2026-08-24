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

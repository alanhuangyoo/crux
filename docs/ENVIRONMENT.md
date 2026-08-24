# 执行环境

## 结论：本机 Mac 不能用于评测，必须走 Modal

2026-08-25 的 oracle 冒烟测试（2 个任务，`--env docker`）**两个都失败**。
oracle 跑的是参考解，本应满分——失败即环境问题。两个失败根因不同：

### 1. 代理劫持 DNS → Docker build 失败

```
E: Failed to fetch http://deb.debian.org/debian/dists/bookworm/InRelease
   502 Bad Gateway [IP: 198.18.0.212]
```

`198.18.0.0/15` 是 RFC 2544 保留段，本机代理（`127.0.0.1:7897`）的 fake-IP 模式
把 `deb.debian.org` 解析到了这里，Docker 构建阶段拿不到 Debian 源。

**可修**：容器内已验证能访问 `host.docker.internal:7897`，
在 Docker Desktop → Settings → Resources → Proxies 里手动指向该地址即可。

### 2. 架构不匹配 → 本地无解

```
ico_patched: ELF 64-bit LSB executable, x86-64
本机 docker : arm64/linux
```

任务分发的是 **x86-64 原生二进制**，本机 Docker 是 **arm64**。被测服务根本起不来：

```
AssertionError: service did not accept connection on port 50373: [Errno 111] Connection refused
17 failed, 2 passed
```

两个纯文件校验的测试通过了，17 个需要真正执行该二进制的测试全部失败。

**不可修**：qemu 模拟慢且不可靠，且官方榜单跑在 x86-64 上——
本机分数与榜单分数不可比，调优会被架构噪声污染。

## 因此

Terminal-Bench 官方 README 明确说明：

> We develop our tasks using Modal in our CI/CD and leaderboard experiments.

Modal 同时解决三个问题：绕开本地代理、提供 x86-64、支持 89 tasks x 5 trials
所需的并发（官方示例用 `--n-concurrent 500`）。本机 Docker 至多用于跑通代码路径，
**不产生任何可信分数**。

## 待办

- [ ] 注册 Modal 账号，`modal token new` 写入 `~/.modal.toml`
- [ ] 用 `--env modal` 重跑 oracle 冒烟，确认全绿再进入 Phase 1

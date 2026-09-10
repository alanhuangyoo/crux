#!/usr/bin/env bash
# Provision an eval node. Idempotent — safe to re-run.
#
# Every setting here exists because its absence cost a full run:
#
#   storage-driver btrfs      overlayfs on btrfs corrupts writes; apt signature
#                             checks fail and 67 of 70 trials never reach the
#                             agent (docs/ENVIRONMENT.md)
#   default-address-pools     Docker's default carves a /12 into 16 networks,
#                             and every trial creates one. At 24 concurrent the
#                             pool is exhausted and trials die on
#                             "all predefined address pools have been fully
#                             subnetted". /24 gives 4096.
#   data-root on /scratch     root partitions on these nodes are shared with
#                             other people's work; image layers do not belong
#                             there.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "### 节点: $(hostname)  $(nproc) 核  $(free -g | awk '/Mem:/{print $2}')G 内存"
df -hT /scratch | tail -1

echo
echo "### 1. Docker"
FS=$(df -T /scratch | tail -1 | awk '{print $2}')
DRIVER="overlay2"
[ "$FS" = "btrfs" ] && DRIVER="btrfs"
echo "文件系统 $FS -> 存储驱动 $DRIVER"

mkdir -p /etc/docker /scratch/docker
cat > /etc/docker/daemon.json <<EOF
{
  "data-root": "/scratch/docker",
  "storage-driver": "${DRIVER}",
  "default-address-pools": [
    { "base": "172.17.0.0/12", "size": 24 }
  ],
  "log-driver": "json-file",
  "log-opts": { "max-size": "10m", "max-file": "3" }
}
EOF
python3 -c 'import json;json.load(open("/etc/docker/daemon.json"))'

if ! command -v docker >/dev/null; then
  apt-get update -qq >/dev/null
  apt-get install -y -qq ca-certificates curl >/dev/null
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $VERSION_CODENAME stable" > /etc/apt/sources.list.d/docker.list
  apt-get update -qq >/dev/null
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin >/dev/null
else
  systemctl restart docker
fi
sleep 6
docker info --format '驱动={{.Driver}}  data-root={{.DockerRootDir}}'

echo
echo "### 2. uv + harbor"
[ -x /root/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="/root/.local/bin:$PATH"
uv tool install harbor >/dev/null 2>&1 || true
harbor --version

echo
echo "### 3. 验收"
docker run --rm ubuntu:24.04 bash -c '
  n=$(apt-get update 2>&1 | grep -cE "GPG error|not signed")
  echo "  apt GPG 错误: $n"
  apt-get install -y -qq tmux >/dev/null 2>&1 && echo "  tmux 安装: OK" || echo "  tmux 安装: FAIL"
' 2>&1 | tail -2

for i in $(seq 1 30); do docker network create _nettest_$i >/dev/null 2>&1 || true; done
echo "  并发网络容量: $(docker network ls --format '{{.Name}}' | grep -c _nettest_)/30"
docker network ls --format '{{.Name}}' | grep _nettest_ | xargs -r docker network rm >/dev/null 2>&1

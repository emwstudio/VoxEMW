#!/usr/bin/env bash
# 生成局域网 HTTPS 证书（mkcert 本地 CA），供 iPhone/iPad 走 https 访问本机服务。
# 为什么需要：iOS 的 getUserMedia（麦克风）只在安全上下文可用，
# http://192.168.x.x 不算，必须 https + 设备信任的 CA。
#
# 用法：bash scripts/make_lan_tls.sh
# 幂等：已有证书则只打印 iPhone 侧指引。
set -euo pipefail
cd "$(dirname "$0")/.."

DIR="scripts/lan_tls"
LAN_IP="$(ipconfig getifaddr en0 || true)"
[ -n "$LAN_IP" ] || { echo "ERROR: 拿不到 en0 的局域网 IP（Wi-Fi 没开？）" >&2; exit 1; }

command -v mkcert > /dev/null 2>&1 || {
    echo "ERROR: 需要 mkcert（brew install mkcert）" >&2; exit 1; }

mkdir -p "$DIR"

# 本地 CA（不强行写系统信任库——那要 sudo；iPhone 侧手动装即可。
# Mac 自己的浏览器也要信任的话，跑一次 sudo mkcert -install）
if [ ! -f "$(mkcert -CAROOT)/rootCA.pem" ]; then
    mkcert -install 2>/dev/null || true   # CA 会生成，信任失败没关系
fi

if [ ! -f "$DIR/key.pem" ]; then
    mkcert -cert-file "$DIR/cert.pem" -key-file "$DIR/key.pem" \
        "$LAN_IP" localhost 127.0.0.1
    echo "==> 证书已生成：$DIR/cert.pem（覆盖 $LAN_IP / localhost）"
else
    echo "==> 证书已存在（$DIR），跳过生成"
fi

cp "$(mkcert -CAROOT)/rootCA.pem" "$DIR/rootCA-for-iphone.pem"

cat <<EOF

==> iPhone 侧一次性配置（让手机信任本机 CA）：
    1. 把 $DIR/rootCA-for-iphone.pem 用 AirDrop 发到 iPhone
    2. 手机：设置 → 通用 → VPN与设备管理 → 安装该描述文件
    3. 手机：设置 → 通用 → 关于本机 → 证书信任设置 → 打开 mkcert 根证书
    之后 https://$LAN_IP:9443 在 iPhone 上就是可信的。

==> Mac 侧（可选）：想让本机浏览器也信任 https，执行一次：
    sudo mkcert -install
EOF

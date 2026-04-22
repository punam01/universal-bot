#!/usr/bin/env bash
# universal-bot VM bootstrap — idempotent. Safe to re-run.
# Target: Ubuntu 22.04 on Oracle Cloud ARM (A1.Flex), 4 CPU / 24 GB RAM.
set -euo pipefail

log() { printf '\033[1;34m[bootstrap]\033[0m %s\n' "$*"; }

ARCH="$(dpkg --print-architecture)"
CODENAME="$(lsb_release -cs)"

log "updating apt"
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

log "installing base packages"
sudo apt-get install -y -qq \
  git curl jq unzip ca-certificates gnupg lsb-release \
  age iptables-persistent

# --- SOPS (Phase 4 encrypted secrets) ---
if ! command -v sops >/dev/null 2>&1; then
  log "installing sops"
  SOPS_VER="3.9.1"
  curl -fsSL "https://github.com/getsops/sops/releases/download/v${SOPS_VER}/sops-v${SOPS_VER}.linux.${ARCH}" \
    -o /tmp/sops
  sudo install -m 0755 /tmp/sops /usr/local/bin/sops
  rm -f /tmp/sops
fi

# --- Docker Engine (OSS, Apache-2.0) ---
if ! command -v docker >/dev/null 2>&1; then
  log "installing docker engine"
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${CODENAME} stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq \
    docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER"
  log "docker installed — log out and back in to use it without sudo"
fi

# --- Caddy (auto HTTPS, Apache-2.0) ---
if ! command -v caddy >/dev/null 2>&1; then
  log "installing caddy"
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq caddy
fi

# --- Ollama (local LLM runtime, MIT) ---
if ! command -v ollama >/dev/null 2>&1; then
  log "installing ollama"
  curl -fsSL https://ollama.com/install.sh | sh
fi

# --- Pull default small model ---
if ! ollama list 2>/dev/null | grep -q 'qwen2.5:7b-instruct'; then
  log "pulling qwen2.5:7b-instruct (~4.7 GB, one-time)"
  ollama pull qwen2.5:7b-instruct
fi

# --- Firewall: allow 80 / 443 ---
log "configuring firewall"
sudo iptables -C INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null || \
  sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -C INPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null || \
  sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save >/dev/null

# --- Summary ---
log "bootstrap complete"
echo
echo "  Versions:"
echo "    docker:  $(docker --version 2>/dev/null || echo 'not on PATH yet — log out and back in')"
echo "    caddy:   $(caddy version 2>/dev/null | head -1)"
echo "    ollama:  $(ollama --version 2>/dev/null)"
echo "    sops:    $(sops --version 2>/dev/null | head -1)"
echo
echo "  Next steps:"
echo "    1. Log out and back in (for docker group membership)."
echo "    2. Clone the repo, then copy infra/Caddyfile to /etc/caddy/Caddyfile"
echo "       and replace YOUR_DOMAIN with your DuckDNS subdomain."
echo "    3. sudo systemctl reload caddy"
echo

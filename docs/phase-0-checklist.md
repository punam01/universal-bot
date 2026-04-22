# Phase 0 Checklist

Provision the free infrastructure that will run universal-bot. Total cost: **$0**. Time: ~1 hour.

## Prerequisites

- A credit/debit card (Oracle requires one for identity verification — **they will not charge it** for Always Free resources)
- A GitHub account
- A Google account

---

## 1. Push this scaffold to GitHub

1. Create a new repo at https://github.com/new
   - Name: `universal-bot` (or whatever you prefer)
   - Visibility: **Public** (unlimited GitHub Actions minutes)
   - **Do not** initialize with a README / .gitignore / license — this scaffold already has them
2. From this folder, push:

   ```bash
   cd /c/Users/pkumavat/Documents/universal-bot
   git init -b main
   git add .
   git commit -s -m "chore: initial Phase 0 scaffold"
   git remote add origin https://github.com/punam01/universal-bot.git
   git push -u origin main
   ```

   (Replace `punam01`. The `-s` flag adds a DCO sign-off.)

---

## 2. Oracle Cloud Always Free VM — the cornerstone

This is where the entire stack will run.

1. Sign up: https://signup.cloud.oracle.com/
   - Pick the **home region closest to you** (Mumbai / Singapore / Frankfurt / Phoenix / etc.). You cannot change it later.
   - Card is for identity verification only.
2. In the OCI console: **Menu → Compute → Instances → Create Instance**
3. Settings:
   - **Image**: Canonical Ubuntu 22.04
   - **Shape**: `VM.Standard.A1.Flex` (ARM Ampere) — **4 OCPUs, 24 GB RAM** (max Always Free allotment)
   - **Networking**: keep defaults, assign a public IPv4
   - **SSH keys**: paste your public key. If you don't have one:

     ```bash
     ssh-keygen -t ed25519 -C "you@example.com"
     cat ~/.ssh/id_ed25519.pub
     ```

4. Create. Wait ~2 minutes for `Running` state.
5. Note the **public IP**.

### Open firewall ports

Oracle blocks everything by default.

**In the OCI console** → Networking → Virtual Cloud Networks → your VCN → Security Lists → Default Security List → add two ingress rules:

| Source | Protocol | Port |
|---|---|---|
| `0.0.0.0/0` | TCP | 80 |
| `0.0.0.0/0` | TCP | 443 |

**Inside the VM** (Oracle's Ubuntu images also have local iptables):

```bash
ssh ubuntu@YOUR_VM_IP
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo apt-get install -y iptables-persistent
sudo netfilter-persistent save
```

---

## 3. Bootstrap the VM

SSH in and run the bootstrap script from this repo (idempotent, safe to re-run):

```bash
ssh ubuntu@YOUR_VM_IP
curl -fsSL https://raw.githubusercontent.com/punam01/universal-bot/main/infra/vm-bootstrap.sh -o bootstrap.sh
bash bootstrap.sh
```

What it installs:
- Docker Engine + compose plugin (Apache-2.0)
- `git`, `curl`, `jq`, `age`, `sops` (for Phase 4 secret encryption)
- Caddy (auto-HTTPS, Apache-2.0)
- Ollama (local LLM, MIT) + pulls `qwen2.5:7b-instruct` (~4.7 GB)

After it finishes, **log out and back in** so your user picks up Docker group membership:

```bash
exit
ssh ubuntu@YOUR_VM_IP
docker ps   # should work without sudo
```

---

## 4. DuckDNS free subdomain

1. Go to https://www.duckdns.org/ → sign in with GitHub or Google
2. Create subdomain, e.g. `yourname-universal-bot` → it auto-points to your current IP
3. Copy the **token** shown at the top of the page
4. On the VM, set up auto-update (runs every 5 minutes):

   ```bash
   mkdir -p ~/duckdns && cd ~/duckdns
   cat > duck.sh <<'EOF'
   echo url="https://www.duckdns.org/update?domains=YOURNAME-universal-bot&token=YOUR_TOKEN&ip=" | curl -k -o ~/duckdns/duck.log -K -
   EOF
   chmod +x duck.sh
   ./duck.sh && cat duck.log   # should print "OK"
   (crontab -l 2>/dev/null; echo "*/5 * * * * ~/duckdns/duck.sh >/dev/null 2>&1") | crontab -
   ```

5. Verify from your laptop:

   ```bash
   ping yourname-universal-bot.duckdns.org
   ```

---

## 5. Caddy auto-HTTPS

Copy `infra/Caddyfile` from the repo to `/etc/caddy/Caddyfile` on the VM, replacing `YOUR_DOMAIN` with your DuckDNS subdomain:

```bash
# on the VM
sudo cp ~/universal-bot/infra/Caddyfile /etc/caddy/Caddyfile
sudo sed -i 's|YOUR_DOMAIN|yourname-universal-bot.duckdns.org|g' /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Caddy automatically fetches a Let's Encrypt certificate the first time it's hit.

Test from your laptop:

```bash
curl -I https://yourname-universal-bot.duckdns.org
# Expect HTTP/2 404 (no service behind Caddy yet — that's Phase 1). Cert should be valid.
```

---

## 6. Free API keys (for development)

### Google AI Studio — Gemini (free)
1. https://aistudio.google.com/app/apikey
2. Create API key → save it
3. Free limits: ~15 RPM, ~1500 req/day on Gemini 2.0 Flash

### Groq — free Llama 3.3 70B (fast)
1. https://console.groq.com/keys
2. Create API key → save it

### OpenRouter — optional, access to many `:free` models
1. https://openrouter.ai/keys
2. Create key → save it

**Do not commit keys to git.** Phase 4 adds SOPS-encrypted key storage. Until then, keep them in a local `.env` file (already in `.gitignore`) or your password manager.

---

## 7. Verification

From your laptop:

```bash
ping yourname-universal-bot.duckdns.org                        # resolves to VM IP
curl -I https://yourname-universal-bot.duckdns.org              # HTTPS cert valid
```

On the VM:

```bash
docker --version                                          # 24.x or newer
docker compose version                                    # v2.x
ollama --version
ollama run qwen2.5:7b-instruct "say hello"                # first run = slow (model load)
```

If all four work, **Phase 0 is done**. Ready for Phase 1.

---

## Cost tracker

| Line item | Cost |
|---|---|
| GitHub (public repo + Actions) | $0 |
| Oracle Cloud Always Free VM (4 CPU / 24 GB RAM) | $0 |
| DuckDNS subdomain | $0 |
| Caddy + Let's Encrypt certificate | $0 |
| Ollama + local models | $0 |
| Gemini / Groq / OpenRouter free tiers | $0 |
| **Total** | **$0** |

---

## Troubleshooting

**"My A1.Flex shape is out of capacity"**
Oracle's ARM capacity oscillates. Keep trying every few hours, or pick a less-popular region. Fallback: take the 2× AMD micro VMs (always available) and use a smaller local model like `qwen2.5:3b`.

**"Firewall rules don't seem to work"**
OCI needs both the Security List rules (cloud) **and** iptables rules (host). Check both. Test with `curl -v http://VM_IP` from another machine.

**"Let's Encrypt fails"**
Your DuckDNS subdomain must resolve to the VM's public IP *before* Caddy starts. Run `./duck.sh` manually once, confirm with `dig yourname-universal-bot.duckdns.org`, then restart Caddy.

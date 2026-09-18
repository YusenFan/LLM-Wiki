#!/usr/bin/env bash
# Run with sudo after extracting the source to /home/azureuser/LLM-Wiki.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip
cd /home/azureuser/LLM-Wiki
sudo -u azureuser python3 -m venv .venv
sudo -u azureuser .venv/bin/pip install -r requirements.txt
sudo -u azureuser .venv/bin/python build_test_hundred.py --prepare-only
if [[ ! -f /home/azureuser/.config/llm-wiki.env ]]; then
    install -d -m 700 -o azureuser -g azureuser /home/azureuser/.config
    install -m 600 -o azureuser -g azureuser /dev/null /home/azureuser/.config/llm-wiki.env
    cat > /home/azureuser/.config/llm-wiki.env <<'ENV'
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_PREMIUM_MODEL=gpt-4o
LLM_FAST_MODEL=gpt-4o-mini
ENV
fi
cat > /etc/systemd/system/llm-wiki-hotpotqa.service <<'UNIT'
[Unit]
Description=LLM-Wiki build from first 100 HotpotQA questions
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
User=azureuser
WorkingDirectory=/home/azureuser/LLM-Wiki
EnvironmentFile=/home/azureuser/.config/llm-wiki.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/azureuser/LLM-Wiki/.venv/bin/python /home/azureuser/LLM-Wiki/build_test_hundred.py
TimeoutStartSec=infinity
RemainAfterExit=yes
UMask=0077
StandardOutput=journal
StandardError=journal
UNIT
systemctl daemon-reload
echo 'Ready: add your key to ~/.config/llm-wiki.env, then sudo systemctl start --no-block llm-wiki-hotpotqa'

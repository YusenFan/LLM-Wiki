# Azure HotpotQA build

VM provisioned using Azure CLI on 2026-09-18:

- Subscription: Azure subscription 1 (`0d80a242-7df6-485c-85c5-a3bc3a9e77f5`)
- Resource group: `llm-wiki-hotpotqa`
- VM: `llm-wiki-vm`, Malaysia West, Ubuntu 24.04, `Standard_B2s_v2`
- Disk: 32 GB Standard SSD
- Public IP: `85.211.192.31`
- Project: `/home/azureuser/LLM-Wiki`

The source snapshot excludes local credentials, Git history, previous outputs,
and caches. The dataset contains exactly the first 100 entries from the local
HotpotQA distractor development dataset, producing 991 distinct context articles.
`build_test_hundred.py` passes all returned article paths to ingestion, then builds
summaries. It does not run QA predictions or evaluation.

## Add credentials and start

From your Mac:

```bash
ssh -i ~/.ssh/id_rsa azureuser@85.211.192.31
```

On the VM, edit the private environment file. Paste your key and, if using another
provider, change the endpoint and model names as appropriate:

```bash
nano ~/.config/llm-wiki.env
sudo systemctl start --no-block llm-wiki-hotpotqa
sudo journalctl -u llm-wiki-hotpotqa -f
```

The file is mode 600. Defaults match the project: OpenAI endpoint, `gpt-4o` for
synthesis and `gpt-4o-mini` for analysis. The systemd service loads this file and
uses the project's virtual environment. It continues after SSH disconnects.
Ctrl-C exits log watching without stopping the build.

## Results and status

```bash
sudo systemctl status llm-wiki-hotpotqa --no-pager
cat ~/LLM-Wiki/wiki_output/hotpotqa/first-100/build-result.json
```

While building, the oneshot service appears as `activating`. A successful exit
leaves it `active (exited)`; errors leave it `failed`. Check both article and
summary failures in the report. Wiki files are under
`~/LLM-Wiki/wiki_output/hotpotqa/first-100/wiki/`.

Restart after fixing a failure with `sudo systemctl restart --no-block
llm-wiki-hotpotqa`. This uses the existing build cache; inspect failure reports
rather than assuming cache skips mean all output is complete.

For manual runs instead of systemd (do not run concurrently):

```bash
cd ~/LLM-Wiki
source .venv/bin/activate
set -a
source ~/.config/llm-wiki.env
set +a
python build_test_hundred.py
```

## Download and stop billing for compute

From your Mac after the build finishes:

```bash
scp -i ~/.ssh/id_rsa -r azureuser@85.211.192.31:~/LLM-Wiki/wiki_output/hotpotqa/first-100 ./azure-hotpotqa-first-100
az vm deallocate -g llm-wiki-hotpotqa -n llm-wiki-vm
```

Disk and public IP charges can continue while deallocated. The VM is not set to
automatically shut down, to avoid interrupting the build. Start it again with
`az vm start -g llm-wiki-hotpotqa -n llm-wiki-vm`.

Azure CLI reference: https://learn.microsoft.com/en-us/cli/azure/vm

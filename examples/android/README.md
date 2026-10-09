# S25 on-demand Adreno serving

Device-local example used for the 2026-10-09 JackKnifeAI checkpoint. Requires
native Termux Python, a Vulkan-enabled `c/qwen36`, compiled shader, and the
converted Qwen3.6 container. It uses the existing Python OpenAI gateway;
Hexagon expert serving remains unsupported. This is not an Android background
service or boot receiver: Android may kill it, and the next agent launch starts
it again. A running server keeps resident weights in memory until stopped.

Copy `coli_serve.py` to `~/bin/coli_serve`, chmod 700. Default Termux home is
`/data/data/com.termux/files/home`; optional overrides: `COLI_SERVE_HOME`,
`COLI_REPO`, `COLI_MODEL`, `COLI_MODEL_ID`. Defaults match the S25 test layout.
The server listens only on `127.0.0.1:8080`, with four CPU threads, eight
expert cache slots per layer, 2048 MiB Vulkan dense budget, streamed embeddings,
8,192 maximum context tokens and 1,024 default output tokens. No API key is
required by this loopback service. Do not expose it to the network unprotected.

```sh
coli_serve start     # waits for the expected /v1/models entry; locked startup
coli_serve status
coli_serve stop
```

Before executing Pi or Hermes, each launcher calls:

```sh
/data/data/com.termux/files/home/bin/coli_serve ensure || exit $?
```

Then exec the actual runtime, avoiding recursion. On the tested Ubuntu PRoot:
Pi's actual command is `/opt/coli-node/bin/node
/opt/coli-node/lib/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js`;
Hermes's is `/root/.local/bin/hermes`. The native Termux wrappers run ensure
before entering PRoot; Ubuntu `/usr/local/bin` wrappers also run ensure for
direct Ubuntu launches. Existing launchers/settings are backed up before edits.

Configure both runtimes' OpenAI chat-completions provider `colibri` with:

- Base URL: `http://127.0.0.1:8080/v1`
- Model ID: `huihui-qwen3.6-35b-a3b-colibri-int4` (or the overridden ID)
- API-key placeholder if required: `colibri-local`
- Context: 8192; output budget: 1024; text input only
- Disable developer-role/reasoning-effort request extensions in Pi

Preserve the existing Venice provider and credentials; `/model` selects another
provider/model. Pi's per-model compaction settings use the key
`colibri/huihui-qwen3.6-35b-a3b-colibri-int4` with reserveTokens=2048 and
keepRecentTokens=2048. Hermes enables compression with a model threshold of
0.75. These are agent summaries, not streamed KV. Full-window compaction and
long tool-use sessions have not been validated. Large agent prompts are slow
on this streamed model; the measured Adreno decode rate is about 0.48 tokens/s.

Logs and PID: `~/.local/state/colibri/` in native Termux. Inspect `server.log`
on startup failure. A successful model listing proves load/readiness; use a
real chat request to verify generation. No private keys, SDK libraries, QNN
contexts or model artifacts are included here.

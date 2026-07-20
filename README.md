# Kouri

<p align="center">
  <img src="assets/LOGO.png" alt="Kouri Logo" width="120" />
</p>

<p align="center">
  <strong>Intelligence, Modularized.</strong><br />
  A privacy-first intelligence layer — multi-agent orchestration, on-device memory, and local inference via Ollama.<br />
  Less assistant. More cognitive operating system.<br />
  Built by Tadey.
</p>

<p align="center">
  <a href="https://kouri-rukkan.vercel.app/"><strong>Rukkan → Private Access</strong></a>
  ·
  <a href="https://github.com/TadeyRuk/Kouri-Public/wiki">Wiki</a>
</p>

| Private Access | Chat |
|:---:|:---:|
| ![Rukkan Private Access at kouri-rukkan.vercel.app](assets/screenshots/kouri-private-access.png) | ![Kouri chat workspace](assets/screenshots/kouri-chat.png) |

<p align="center"><em>Live Private Access surface: <a href="https://kouri-rukkan.vercel.app/">kouri-rukkan.vercel.app</a></em></p>

---

## What Kouri is

Kouri is a privacy-first intelligence layer designed to route complex tasks across specialized local models and behavioral logic. By merging multi-agent orchestration with persistent on-device memory, it evolves beyond a simple chatbot into a cognitive operating system — learning your patterns, securing your data, and integrating into daily workflows without exposing sensitive context to the cloud.

Rukkan is the private web workspace for Kouri: a personal AI environment, not a public SaaS product.

---

## Design philosophy

Kouri is built around five principles:

1. **Modular over monolithic**
2. **Local-first architecture**
3. **Behavior-driven intelligence**
4. **Task-specialized models**
5. **User sovereignty over data**

It is less "assistant" and more "cognitive operating system."

---

## Surfaces

### Web — Intelligent interaction. Seamless execution.

A React-based workspace for chat, workflows, and agent activity: real-time token streaming, artifact rendering, and a clean latency-free environment for rapid command execution.

### Mobile — Always within reach. Always private.

Access Kouri from your phone — ask questions, pull context, or kick off tasks without touching desktop. Same intelligence, grounded in your own memory layer. No cloud intermediaries for the core loop.

---

## Modules

The Kouri ecosystem is modular. Core surfaces ship today; some modules are expanding as the architecture grows.

| Module | Role |
| :--- | :--- |
| **Kouri Core** | Orchestration layer — intent routing, task decomposition, module hand-off |
| **KouriMemory** | Persistent on-device context — preferences, history, longitudinal companion memory |
| **KouriVault** | Folder-scoped semantic document search (RAG) grounded in your files |
| **Kouri Hub** | Local execution surface — web, TUI, and desktop-adjacent entry points |
| **KouriLink** | Cross-device continuity (expanding) |
| **KouriSense** | Usage insight / analytics concepts (experimental) |
| **KouriPulse** | Agent and process health concepts (experimental) |
| **KouriDevTools** | IDE / developer workflows (expanding) |
| **KouriCloud** | Optional cloud path — local by default; cloud only when needed |

---

## Privacy model

- **Inference:** local Ollama by default (`qwen3.5:4b`)
- **Memory & prompts:** stay on-device
- **Telemetry:** none by design
- **Exceptions:** optional satellites (e.g. Gmail OAuth) are cloud-adjacent and not part of the zero-cloud core

---

## Quick start

Prerequisites: Python 3.10+, Node.js 18+, and [Ollama](https://ollama.com) running locally.

```bash
ollama pull qwen3.5:4b
```

Clone this repository, then follow the [wiki](https://github.com/TadeyRuk/Kouri-Public/wiki) for run instructions (API on `:5000`, web UI on `:5173`).

For the live private workspace: [https://kouri-rukkan.vercel.app/](https://kouri-rukkan.vercel.app/)

---

## License

[AGPL-3.0](LICENSE)

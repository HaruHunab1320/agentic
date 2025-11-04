# Agentic

Agentic is an experimental multi-agent development environment that lets you
orchestrate Claude, Gemini, Aider, and now a preview Codex research agent from a
single CLI. The project focuses on fast iteration, rich observability, and
composable workflows that stitch together analysis, implementation, testing, and
long-running research loops.

## Highlights
- **CLI orchestrator** – `agentic` bootstraps shared memory, task routing, and
  intelligent coordination (`src/agentic/core`).
- **Rich documentation** – Phase plans, outlines, and architecture notes live in
  `docs/` while historical analyses are archived under `docs/archive/`.
- **Curated examples** – Legacy frontend/backend demos now reside in
  `examples/` so the Python package stays clean.
- **Codex research agent (preview)** – A new agent type scaffolds unattended
  research sessions and paves the way for automated experiments.

## Getting Started
1. Create a virtual environment and install the package in editable mode:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```
2. Run the CLI:
   ```bash
   agentic --help
   ```
3. Explore the documentation starting with `docs/ARCHITECTURE.md` and the phase
   briefs in `docs/Phase*.md`.

## Repository Layout
```
docs/                    → design notes, outlines, and archived reports
examples/                → sample projects (full-stack, vite, snippets, archives)
src/agentic/             → Python package with orchestrator, agents, and models
tests/                   → pytest suite covering agents, coordination, and QA
```

## Codex Research Agent (Preview)
- `AgentType.CODEX_RESEARCH` is registered with a placeholder
  `CodexResearchAgent` that maps out the control loop for future Codex
  integrations.
- The agent records preliminary notes, emits a `research_finding` discovery, and
  returns a summary output while running in preview mode (no external API calls
  yet).
- See `docs/outlines/RESEARCH_AGENT_PLAN.md` for the roadmap to full automation,
  including sandboxed experiment runners and outcome verification.

Spawn it manually once the orchestrator is configured:
```bash
agentic spawn research --model codex-latest
```
(API wiring is still pending; the command reports planned work.)

## Documentation Guide
- **Outlines & Roadmaps** – `docs/outlines/`
- **Phase Execution Plans** – `docs/Phase1.md` … `docs/Phase6.md`
- **Architecture & Specs** – `docs/ARCHITECTURE.md`, `docs/verification-loop.md`
- **Historical Analyses** – `docs/archive/analysis/`

## Contributing
The repo is intentionally exploratory. File issues or drop notes in
`docs/outlines/` if you identify new workflows or want to help implement the
Codex research pipeline.

# Codex Research Agent Plan

## Goal
Provide an agent that can run unattended research and code experiments for a
specified budget (time or iteration count) and synthesize discoveries that the
rest of the swarm can consume.

## Current State
- `AgentType.CODEX_RESEARCH` registers a placeholder `CodexResearchAgent`.
- The agent scaffolds a research loop, records preliminary notes, and emits a
  `research_finding` discovery so downstream automation can react.
- No external API calls or sandboxed executions are triggered yet; the loop
  intentionally exits after a single iteration.

## Next Steps
1. **Codex/API Integration**
   - Wire the agent to the selected large language model or toolchain.
   - Capture streamed tool outputs and attach artifacts (links, files).
2. **Experiment Runner**
   - Add a sandbox executor (Docker or firejail wrapper) that the agent can call.
   - Track experiment metadata (command, duration, result) inside shared memory.
3. **Budget Enforcement**
   - Replace the placeholder loop with cooperative scheduling that respects the
     time/iteration budget and supports pause/resume from orchestration.
4. **Outcome Evaluation**
   - Define target-outcome schemas (tests passing, metric thresholds, artifact
     presence) and surface them to the verification layer.
5. **CLI/Config UX**
   - Extend `agentic spawn research` to accept flags such as `--topic`,
     `--time-limit`, and `--target`.
   - Allow persistent defaults via `AgenticConfig`.
6. **Observability**
   - Stream progress into the unified swarm monitor and persist summaries for
     later review.

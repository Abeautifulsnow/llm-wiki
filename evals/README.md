# Evals

Evaluation fixtures and release gates land with §51 steps 18–19
(`eval fixtures`, `end-to-end tests`); thresholds and metric definitions are
specified in PRD §37. Deterministic CI evaluation runs against
`FakeLlmProvider`; real-LLM runs are explicitly triggered only.

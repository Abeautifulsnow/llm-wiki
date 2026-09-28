---
title: Tracing
description: Distributed trace collection.
---

# Tracing

## Propagation

Non-W3C headers are ignored and a new trace is started.

Traces use W3C trace context propagation headers.

## Retention

Trace data is retained for 30 days.

Export traces to your own backend if longer retention is required.

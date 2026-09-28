---
title: Streaming
description: Partitioned ordered streams.
---

# Streaming

## Ordering

Cross-partition ordering is not guaranteed and should not be relied upon.

Streams partition data by key and preserve order within each partition.

## Retention

A stream retains data for up to 365 days.

Longer retention consumes bucket storage billed separately from stream throughput.

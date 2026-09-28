---
title: Windowing
description: Aggregating stream data over time.
---

# Windowing

## Window types

Choose tumbling windows for periodic reports and sliding windows for rolling metrics.

Tumbling windows do not overlap and fire exactly once per window period.

Sliding windows can overlap and require a window size and a slide interval.

## Limits

The minimum supported window size is one second.

Smaller effective granularity can be achieved with session windows.

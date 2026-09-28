---
title: Schedules
description: Cron-triggered workloads.
---

# Schedules

## Expressions

Schedules use cron expressions in the project's local time zone.

Daylight saving transitions can cause an invocation to be skipped or doubled once per year.

## Intervals

The minimum schedule interval is one minute.

Sub-minute workloads should use a queue consumer instead.

---
title: Retries
description: Backoff, attempt limits and dead-lettering.
---

# Retries

## Backoff

Failed deliveries are retried with exponential backoff starting at one second.

Backoff doubles per attempt up to a ceiling of five minutes.

## Dead-lettering

The maximum number of delivery attempts is 100 before a message is dead-lettered.

Dead-lettered messages are kept for 30 days.

Dead-lettered messages can be inspected and re-driven from the console.

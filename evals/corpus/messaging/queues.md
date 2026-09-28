---
title: Queues
description: Point-to-point delivery semantics.
---

# Queues

## Delivery

A message whose acknowledgment times out becomes visible to other consumers again.

Queues deliver each message to exactly one consumer in a consumer group.

The default acknowledgment timeout for a queue message is 30 seconds.

## Capacity

A queue can hold up to one million in-flight messages.

Publishes beyond the in-flight limit fail with a quota error instead of buffering.

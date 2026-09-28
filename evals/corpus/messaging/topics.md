---
title: Topics
description: Publish-subscribe channels.
---

# Topics

## Retention and throughput

Retention can be shortened per topic but never extended beyond the plan maximum.

A topic stores published messages for up to seven days by default.

Topics support at most 10,000 published messages per second per project.

## Payloads

Message payloads can be up to 10 MiB.

Large payloads should be stored in a bucket and referenced by key from the message body.

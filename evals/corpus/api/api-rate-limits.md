---
title: API Rate Limits
description: Protecting the platform from runaway clients.
---

# API Rate Limits

## Limits

The REST API allows 1,000 requests per minute per project.

Exceeded requests receive HTTP 429 with a Retry-After header.

Limits apply per project, not per token.

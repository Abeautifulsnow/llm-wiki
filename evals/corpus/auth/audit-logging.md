---
title: Audit Logging
description: Administrative action records and retention.
---

# Audit Logging

## What is recorded

Read-only data plane operations are not part of the audit log.

Every administrative action is written to the audit log within ten seconds.

## Retention and export

Audit log entries are retained for 400 days on the Enterprise plan.

Audit logs can be exported to the customer's own bucket every hour.

Exports are newline-delimited JSON and include the acting identity and the target resource.

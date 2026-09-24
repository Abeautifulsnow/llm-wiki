---
title: Plugin Security
---

# Plugin Security

Plugins declare the permissions they need; the runtime grants them per host
policy. Undeclared capability access is denied and audited.

## Permission Model

There are three permission scopes:

- `read:data` — read documents inside the assigned workspace.
- `write:data` — create and update documents.
- `net:egress` — call configured external endpoints.

The [plugin runtime](architecture.md) enforces permissions at the message bus
boundary, so a plugin cannot bypass checks by talking to another plugin
directly.

## Audit

Every permission check is recorded in the audit log with the plugin id, the
requested scope, and the decision.

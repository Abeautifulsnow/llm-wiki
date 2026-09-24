---
title: Plugin Architecture
---

# Plugin Architecture

The plugin system is built around a small runtime core that loads plugin
packages and supervises their lifecycle.

## Plugin Runtime

Every plugin runs inside a sandboxed host process. The runtime provides a
message bus so plugins communicate without direct imports.

### Lifecycle

A plugin moves through the states `registered → resolved → active → retired`.
The runtime retries the `resolved → active` transition up to three times
before marking the plugin failed.

### Message Bus

The message bus delivers typed events. Delivery is at-least-once; handlers
must be idempotent.

## Packaging

A plugin package contains a `plugin.toml` manifest and compiled artifacts.
The manifest declares the plugin id, version, and requested permissions.

---
title: Getting Started
description: First steps with the DBX platform.
---

# Getting Started

This guide walks you through installing DBX and registering your first plugin.

## Installation

Download the CLI from the release page and add it to your `PATH`.

```bash
dbx --version
```

## Registering a Plugin

Place your plugin under `plugins/` and run:

```bash
dbx plugin register ./plugins/my-plugin
```

See the [plugin architecture](plugin/architecture.md) for how plugins are
loaded at runtime, and the [security model](plugin/security.md) for what
permissions a plugin can request.

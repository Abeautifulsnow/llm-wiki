---
title: Architecture
description: Control plane and data plane responsibilities.
---

# Architecture

## Control plane

The Nimbus control plane stores metadata in a replicated SQL cluster.

The control plane handles provisioning, quota accounting, and access policy decisions.

## Data plane

Data plane nodes are stateless and scale horizontally.

Control plane and data plane communicate over mutual TLS.

Because data plane nodes are stateless, node restarts do not affect in-flight durability.

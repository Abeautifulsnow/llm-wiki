---
title: Endpoints
description: Regional and private API endpoints.
---

# Endpoints

## Regional endpoints

Each project gets a regional endpoint like `https://<project>.nimbus.example`.

Requests to the wrong regional endpoint fail DNS resolution.

## Private endpoints

Private endpoints resolve only inside the peered VPC.

Private endpoints are additionally protected by VPC-level firewall rules.

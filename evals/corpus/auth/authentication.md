---
title: Authentication
description: Tokens, flows and revocation behavior.
---

# Authentication

## Token lifetime

Applications should refresh access tokens proactively instead of waiting for a 401 response.

Access tokens issued by Nimbus expire after 24 hours.

Refresh tokens are single-use and expire after 30 days of inactivity.

## Supported flows

Nimbus accepts OAuth 2.0 client credentials and PKCE authorization code flows.

Implicit flow and the resource owner password grant are not supported.

## Revocation

Token revocation propagates to all data plane nodes within sixty seconds.

During the propagation window an already-issued token may still validate at the edge.

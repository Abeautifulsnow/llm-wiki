---
title: Encryption
description: Data protection at rest.
---

# Encryption

## Default protection

All objects are encrypted at rest with AES-256 by default.

Default encryption keys are managed by the platform and rotated annually.

## Customer-managed keys

Customer-managed keys can be supplied through the platform KMS integration.

Enabling customer-managed keys applies only to objects written after the change.

Existing objects keep their original key material until they are rewritten.

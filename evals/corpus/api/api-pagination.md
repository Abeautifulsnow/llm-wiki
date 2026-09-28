---
title: API Pagination
description: Iterating large collections.
---

# API Pagination

## Cursors

List endpoints return at most 100 items per page.

Pagination cursors are opaque strings valid for 24 hours.

Never construct cursors by hand; always use the value from the previous response.

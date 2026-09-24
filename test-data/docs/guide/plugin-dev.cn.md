---
title: 插件开发指南
lang: zh-CN
---

# 插件开发指南

本指南介绍如何从零开始编写一个 DBX 插件，并在本地完成调试。

## 项目结构

一个插件项目至少包含 `plugin.toml` 清单文件与编译产物目录。
清单中声明插件 ID、版本以及所需的权限（参见安全模型中的三种权限范围）。

## 本地调试

使用开发服务器加载插件：

```bash
dbx plugin register ./plugins/my-plugin --watch
```

`--watch` 会在文件变化时自动重启插件宿主进程。

## 生命周期处理

插件会经历 `registered → resolved → active → retired` 四个状态。
运行时对 `resolved → active` 的转换最多重试三次，失败后插件将被标记为
failed。处理器必须保证幂等，因为消息总线提供的是至少一次投递。

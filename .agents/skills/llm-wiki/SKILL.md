---
name: llm-wiki
description: "用 llm-wiki 知识编译器把项目文档编译成带引证、可溯源的持久 Wiki，并从中检索知识。覆盖两件事：(1) 建库——init/scan/build/replan/lint/serve 命令与个人知识库的构建维护；(2) 取知识——优先用 HTTP /v1/context 或 search 拿到带引证的上下文后自行推理，除非交付物本身就是给人看的答案，否则不要调 /v1/query 或 ask（避免双重推理）。当仓库存在 .llm-wiki/ 工作区、或需要对文档做『带出处』的问答与上下文检索时使用。"
---

# llm-wiki

llm-wiki 是知识编译器：`docs/` 下的源文档保持不动，它把文档编译成一份**派生的、每句都可溯源**的 Markdown Wiki（每个 claim 带 source file + heading path + byte range + content digest，由编译器而非模型写入）。

它面向两类使用者：**人**（CLI）和 **AI agent**（HTTP API v1 / TS SDK）。本 skill 教 agent 两件事——**怎么建库**、**怎么取知识**。

## 0. 先判断该不该用

- **该用**：需要对项目文档做**带引证**的问答 / 需要一段 grounded 的上下文喂给自己的推理；仓库里有 `.llm-wiki/` 工作区；文档量大、跨多文件，靠 grep 拼不出答案。
- **不该用**：文档就几个小文件，直接读文件 / grep 更快；你只是要找一个字符串。

---

# 第一部分：建库（构建与维护知识库）

## 1.1 前置条件

- Rust 工具链；一个 OpenAI-compatible 的 LLM 端点（vLLM / Ollama / OpenRouter / 官方 API 都行）。
- LLM 密钥放**环境变量**（config 里只写变量名，不写密钥本身）：

```bash
export LLM_WIKI_API_KEY=sk-...      # 变量名可在 config.api_key_env 改
```

## 1.2 工作区布局

| 路径 | 内容 |
|---|---|
| `docs/`（可配） | **源文档**，唯一事实来源，llm-wiki 永不修改它 |
| `.llm-wiki/config.toml` | 配置 |
| `.llm-wiki/state.db` | SQLite 状态（registry / cache / graph / insights） |
| `wiki/generations/{build_id}/*.md` | 编译产物，纯 Markdown，任何 LLM/agent 都能直接读 |
| `wiki/current.json` | 指向当前生效 generation 的指针 |

`wiki_dir` 必须在 `source.root` **之外**（否则启动即报错）。

## 1.3 生命周期（核心命令）

```bash
llm-wiki init                  # 生成 .llm-wiki/config.toml（已存在则拒绝覆盖）
# —— 编辑 config：至少设 source.root 与 llm.model / llm.base_url ——
llm-wiki scan ./docs           # 扫源目录、建 manifest、落 Source Registry。零 LLM 调用
llm-wiki build                 # 全流程：scan → analyze → plan → compile → publish。首次构建花真 token
llm-wiki status                # 源数量 + 最近一次构建
llm-wiki search "单点登录"      # 只做全文检索，返回页面 section，不是答案
```

**建一个个人知识库的最小闭环**：把你的笔记（`.md` / `.markdown` / `.mdx`）放进源目录 → `scan` 确认扫到了什么（不花钱）→ `build` 编译 → 之后用 `search` / HTTP `/v1/context` 取知识。

## 1.4 成本与增量（关键行为）

- 首次 `build` 编译全部，花真 token。**重建只重编受影响页面**，未变页面原样搬运。
- **无改动的第二次 `build` 零 LLM 请求**（按 fingerprint 缓存）。
- `llm-wiki embed` 同样增量：已覆盖的 generation 零 embedding 请求。
- 模型质量重要：抽取要求严格 JSON + 精确 evidence range，小模型容易触发 rejected-claim 安全阀（超过 `max_rejected_claim_ratio` 即构建失败）。**思考型模型**（deepseek 系）会从 output budget 里烧推理 token，需调高 `[llm] max_output_tokens`（如 32768），否则分析在见到输出前就报 `NO_JSON`。

## 1.5 漂移与 REPLAN_REQUIRED

构建 fingerprint 覆盖 model + prompt + schema + parser + config 版本。任何一项漂移，增量构建会**停下并报 `REPLAN_REQUIRED`（退出码 7）**，而不是偷偷重写页面。既有页面出链拓扑发生结构性变化时同理（§19 拓扑护栏）。

处理方式——先审计再确认：

```bash
llm-wiki replan --dry-run      # 审计 plan diff + 成本估算，不写盘、不编译、不发布
llm-wiki replan                # 执行：page ID 稳定，能扛 merge/split；这条命令本身就是用户确认
```

## 1.6 维护与诊断

```bash
llm-wiki lint                  # 结构体检：引证完整性、死链、孤儿页、手改、重复。发现 Error 则退出码 11
llm-wiki lint --semantic       # 追加 LLM 语义审查（矛盾 / 过时 / 弱综述 / 覆盖缺口）。仅建议性，不改退出码，花模型调用
llm-wiki doctor                # 离线自检：config、源目录、state db、发布完整性、FTS5 可用性、密钥环境变量
llm-wiki doctor --live         # 额外对每个已配置端点发一次真实请求（唯一能区分『配了』和『活着』的办法）
```

`doctor` **只报告不修复**；每个 FAIL 都会打印能修它的命令。

## 1.7 HTTP 服务（agent 取知识的正规入口）

```bash
llm-wiki serve                 # 默认 127.0.0.1:8080；--host / --port 覆盖
```

- **本地模式（默认）**：`/v1` 只接受 loopback；绑非 loopback 直接拒绝启动。
- **远程模式**：`server.remote_enabled = true` 且 `server.auth_token_env` 指的变量非空，否则拒绝启动（fail closed）。请求需 `Authorization: Bearer <token>`。
- 一个进程只服务**一个**工作区；多工作区靠部署层隔离。

## 1.8 配置要点与密钥

配置文件 `.llm-wiki/config.toml`。优先级固定：**CLI args > 环境变量（仅密钥/部署）> 配置文件 > 默认值**。

需要知道的几个键：`source.root`（默认 `./docs`）、`source.include`（默认 md/markdown/mdx）、`llm.model`（**build 前必填**）、`llm.base_url`、`llm.api_key_env`（默认 `LLM_WIKI_API_KEY`）、`llm.max_output_tokens`、`search.source_mode`（默认 `wiki`）、`search.graph`、`search.rerank`（`none` / `cohere-compatible`）、`keep_generations`。

- 密钥**永远来自环境变量**，config 里只写变量名。chat / embedding / rerank 可以是**三个不同 provider**：`[embedding]` / `[rerank]` 里留空的字段继承 `[llm]`。
- 运行 `llm-wiki doctor` 看有效配置摘要（密钥全部脱敏）。

## 1.9 退出码（脚本 / CI 可分支）

| 码 | 含义 | 码 | 含义 |
|---|---|---|---|
| 0 | 成功 | 7 | 规划错误 / `REPLAN_REQUIRED` |
| 2 | 配置错误 / 非法 id | 8 | 编译 / 索引错误 |
| 3 | 源错误 / 路径冲突 | 9 | 预算超限 |
| 4 | 解析错误 | 10 | 需发布恢复 |
| 5 | 存储错误 | 11 | lint 发现错误 |
| 6 | LLM / schema / evidence 校验错误 | 12 | 已取消 |

---

# 第二部分：取知识（检索与消费）

## 2.1 一条硬规则：检索，不要让它替你回答

对 agent 而言，正确姿势是**拿到 grounded 上下文，自己推理**：

```text
Agent → llm-wiki context → Grounded Knowledge Context → Agent 自己的 LLM
```

**不要**默认走 `Agent → llm-wiki query → LLM answer → Agent LLM`，那会形成**双重 reasoning**（PRD §47）。

因此：优先 `/v1/context`（或 `search`）。`/v1/query`、`llm-wiki ask` 是**给人看的答案**服务，只有当交付物本身就是一段给人读的答案、且你不需要再加工时才用。

## 2.2 三个消费面

| 面 | 形态 | 何时用 |
|---|---|---|
| **HTTP API v1** | JSON，协议冻结 | **agent 首选**——结构化、有 `request_id`、无解析成本 |
| **CLI** | 人类可读文本 + 全局 `--json` | 人手操作；agent 用 `--json` 拿一个 JSON 对象（字段对齐 protocol v1） |
| **TS SDK** | `@llm-wiki/sdk`（零依赖 fetch，Node 18+） | TS agent 直接集成，实现的正是 protocol v1 |

CLI 的 `--json` 可放在子命令前后；**stdout 至多一个 JSON 对象**，错误只走 stderr、退出码不变（因此 `lint --json` 仍会在有 error 时退出 11）。需要已发布 wiki 的命令以 `{"published": false}` 表达"尚未构建"（仍退出 0）：

```bash
llm-wiki status --json | jq '.sources'
llm-wiki search "认证流程" --json | jq '.hits[].title'
llm-wiki doctor --json | jq '.checks[] | select(.status == "fail") | .remedy'
```

SDK 用法（协议主版本不匹配会直接拒绝，而不是猜测）：

```ts
import { LlmWikiClient } from "@llm-wiki/sdk";
const wiki = new LlmWikiClient({ baseUrl: "http://127.0.0.1:8080" });
const hits = await wiki.search("plugin permission", 5);        // 只检索
const bundle = await wiki.context("plugin permission", { maxChunks: 8 });
const prior = await wiki.insights(10);                         // 已有洞察层
```

## 2.3 检索语料三模式（`source_mode`）

检索可来自两个语料：**wiki**（当前生效 generation 的 section）与 **source**（原始文档切片）。

`POST /v1/search` 与 `/v1/context` 接受可选字段 `"source_mode": "source" | "wiki" | "fusion"`：

- **不传 → 用配置默认**（`search.source_mode`，默认 `"wiki"`）。非法值报 `400 invalid_request`。
- `"wiki"`：编译后的知识（传统行为）。
- `"source"`：只出原始文档切片。
- `"fusion"`：两边用 RRF 融合；**某切片逐字包含全部 query token 时保留保底前排位**（exact-match protection），融合顺序不会被服务端重排。

**怎么选**：要「已经被人/模型综合过的知识」→ `wiki`；要「原文精确措辞、条款、代码片段」→ `source`；不确定、要召回 → `fusion`。

**降级可见、绝不静默**：响应里 `served` / `degraded` 会逐侧报告 `served | not_published | index_not_built | no_matches | disabled`。看到 `no_matches` 就换 query 或换模式，不要当成「没有这个知识」。

## 2.4 关键端点

| 端点 | 用途 |
|---|---|
| `POST /v1/context` | **agent 集成主入口**：带预算的、多样性感知的检索包（`chunks` + `neighbors` + `estimated_tokens` + `dropped` + `truncated`） |
| `POST /v1/search` | 页面/section 命中，**从不给答案** |
| `GET /v1/pages` · `/v1/pages/{id}` | 当前 generation 的页面元数据 / 整页（带引证与链接），按 id 或 slug |
| `GET /v1/insights` | 已回写的洞察层（最新在前，游标分页） |
| `POST /v1/query` | 带引证校验的**答案**（给人看的）。`write_back: true` 持久化洞察 |
| `POST /v1/embed` | 给当前 generation 增量补 embedding（同步；全覆盖时零请求） |

`context` 的 `budget`：`{max_chunks, max_tokens, max_pages, max_per_source, graph_limit}`。`hybrid: true` 追加向量候选（需先 `embed`）。分页：`?limit`(1–100) + `?cursor`，响应带 `next_cursor`（末页为 null）与 `truncated`。

## 2.5 证据身份与引证（必须区分）

每个命中/chunk 带 `evidence_kind: "wiki" | "source"`：

- **wiki**：带 `page_id`、`slug`、`title`、`heading_path`、`snippet`、`rank`（search 命中还有 `citation_count`）。
- **source**：靠 `source_ref: {source_id, file_path, heading_path, ordinal, range_start, range_end}` + `file_path` / `title` / `heading_path` / `snippet` / `rank` 表达身份，**绝不带 `page_id` / `slug`**。

**因此：不要对 `evidence_kind: "source"` 的条目期待页面身份**（在 `/v1/context` 的 source chunk 上，`slug` 位置放的是**源相对文件路径**）。

引证是**验证过的**：每个被引 claim 都必须能在检索到的上下文里找到；答案里的 `<!-- llm-wiki:cite -->` 锚点由存储的 anchors 展开。无法验证的输出进 `rejected_claims` 审计表，**永不进你的 wiki**。引用时请带上 `file_path` + `heading_path`（必要时 `range_start/range_end`），这样你的回答也可溯源。

## 2.6 失败与降级速查

- `404 nothing_published`：该工作区从未成功 build。默认 wiki 模式会这样失败；`source` / `fusion` 则**可见降级**（200 + 空 hits + 逐侧 `served` 原因）。
- `replan_required`：见 §1.5，先 `replan --dry-run` 再 `replan`。
- `build_already_running` / `job_queue_full`：并发与队列上限，稍后重试。
- `llm_error` / `storage_error` / `internal_error`：看响应里的 `request_id` 排查。
- 先探 `GET /health`（无需鉴权）判断 server 是否活着；`GET /v1/status` 看源数量、最近构建、当前 generation。

---

## 权威参考

- `README.md` —— 命令表、完整 config 键与默认值、HTTP 端点表、退出码。
- `docs/agent-protocol.md` —— agent 协议 v1 契约（冻结，仅允许加法变更）；本 skill 的所有端点/字段以它为准。
- `docs/llm-wiki-prd.md` —— 完整规格（§19 增量/拓扑护栏、§34 退出码、§46–§47 agent 集成契约与此处「检索而非替答」的取舍依据）。

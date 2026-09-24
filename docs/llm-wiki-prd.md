# LLM-Wiki：面向 Agent 的可持续知识编译引擎 PRD

> 版本：V0.1 Engineering PRD  
> 状态：Implementation Ready  
> 首期形态：独立 CLI；HTTP Service 为 V0.4 交付物  
> 核心实现语言：Rust  
> 首期验证输入：具有多级目录结构的 Markdown `docs/` 目录  
> 长期方向：作为 Knowledge Provider / Context Infrastructure 接入 TypeScript Agent Harness

> 修订记录：R1 收敛版本范围并补齐发布/安全/Eval 契约；R2 补齐稳定身份、分层规划与 build/replan 边界；R3 完成跨章节一致性审查；R4 明确 Registry revision、路径规范和运行期观测契约；R5 补齐 MDX 支持、语言标注与平行语料策略、默认 exclude 一致性和全文直塞基线。

---

## 1. 背景

传统 RAG 通常将原始文档切分为 chunk、建立索引，并在查询时检索 Top-K chunk 交给 LLM。这种方式适合快速问答，但存在以下不足：

1. 原始文档的知识组织质量直接决定检索质量。
2. 跨文档概念、实体、依赖关系没有被显式整理。
3. 同一知识可能散落在多个文件中，查询阶段才临时融合。
4. 文档变化后，缺少稳定的知识级增量维护机制。
5. Agent 每次执行任务时需要重复完成“搜索 → 理解 → 组织上下文”。
6. 原始 chunk 难以直接成为长期、可审计、可维护的知识资产。

LLM-Wiki 的目标不是再实现一个“向量数据库 + RAG 服务”，而是构建一个 **Knowledge Compiler（知识编译器）**：将原始资料编译为结构化、可引用、可检索、可增量维护的持久 Wiki，并在此基础上提供知识检索和上下文构建能力。

核心模型：

```text
Raw Sources
    ↓
Parse / Normalize
    ↓
Knowledge Extraction
    ↓
Knowledge Model
    ↓
Wiki Planning
    ↓
Wiki Compilation
    ↓
Persistent Wiki
    ↓
Search / Graph / Context
    ↓
Human / Agent
```

---

## 2. 产品目标

### 2.1 V0.1 目标

首期产品需要证明以下核心假设：

> 给定一个真实、具有层级结构的 Markdown `docs/` 目录，系统能够稳定地将其编译为高质量、可追溯、可增量维护的 Wiki，并提供有效的查询能力。

具体目标：

- 支持递归读取 Markdown docs 目录。
- 保留源目录、文件、Heading 等 provenance 信息。
- 基于 LLM 提取概念、实体、关系、事实和摘要。
- 从多个源文档规划 Wiki 信息架构，而非简单“一文件一 Wiki”。
- 生成持久化 Markdown Wiki。
- Wiki 内容必须能够追溯到源文档。
- 支持 source hash、LLM 缓存和未变更输入的跳过执行。
- 支持 Wiki 之间的关系图。
- 支持本地 CLI。
- 提供 Eval 能力验证生成质量。

以下能力不属于 V0.1：对变更 source 的精确增量重编译和全文检索属于 V0.2；HTTP Service 属于 V0.4；vector retrieval 属于 V0.3。V0.1 的“第二次构建不重复调用 LLM”仅指输入、配置和版本指纹完全相同的缓存命中，不等同于增量编译。

### 2.2 长期目标

成熟后将 LLM-Wiki 作为通用 Knowledge Engine 接入自研 TypeScript Agent：

```text
TypeScript Agent Harness
        │
        ├── Memory
        ├── Tools
        ├── Skills
        └── KnowledgeProvider
                  │
                  ▼
             LLM-Wiki
```

Agent 不应依赖底层 SQLite、Vector Store 或 Graph Store，只依赖稳定的 Knowledge Provider API。

---

## 3. 非目标

V0.1 不实现：

- 多 Agent 编排。
- 通用工作流引擎。
- Neo4j。
- Milvus / Qdrant 等独立 Vector DB。
- Elasticsearch。
- Web 管理 UI。
- MCP Server。
- GitHub/Notion/飞书等 Connector。
- PDF/OCR/Office 文档解析。
- 多租户 SaaS。
- ACL/RBAC 企业权限体系。
- 自动修改原始文档。

这些能力不能污染核心知识编译模型的验证。

---

## 4. 核心设计原则

### 4.1 Source of Truth 不可修改

```text
sources/docs = Source of Truth
wiki         = Derived Knowledge
```

LLM-Wiki 永远不得自动修改 Source。

### 4.2 Wiki 是持久知识，不是临时回答

Wiki 页面生成后需要保存，并参与后续搜索、关系构建和增量更新。

### 4.3 Provenance First

任何由 LLM 生成的重要事实都应尽可能关联：

```text
source file
+ heading/section
+ source hash
```

### 4.4 Graph 是逻辑模型，不等于图数据库

V0.x 使用 SQLite 保存节点和边。

### 4.5 Vector Retrieval 不等于 Vector Database

V0.x 优先使用本地实现；规模证明有必要后再接入专用数据库。

### 4.6 Core 与 Runtime 解耦

`llm-wiki-core` 不依赖 HTTP、CLI、Axum 或具体 LLM Provider。

### 4.7 LLM 输出必须结构化

核心流水线禁止依赖自由文本解析；LLM 中间结果必须通过 JSON Schema/Serde 数据结构验证。

### 4.8 可重建

删除派生索引后，应能够从 Source + Wiki/State 重建。

---

## 5. 用户场景

### 5.1 初始化项目

```bash
llm-wiki init
```

生成：

```text
.llm-wiki/
├── config.toml
├── state.db
└── cache/

wiki/
├── current.json
└── generations/
```

### 5.2 构建 Wiki

```bash
llm-wiki build ./docs
```

输出示例：

```text
Scanning sources...
Documents: 87
Added: 87
Modified: 0
Unchanged: 0

Analyzing documents...
Planning wiki...
Compiling pages...

Wiki pages: 32
Citations: 286
Relations: 94

Build completed in 2m31s
```

### 5.3 增量构建（V0.2）

修改：

```text
docs/plugin/security.md
```

再次：

```bash
llm-wiki build ./docs
```

期望：

```text
Documents: 87
Added: 0
Modified: 1
Deleted: 0
Unchanged: 86

Affected wiki pages:
- plugin-system
- plugin-security

Recompiled: 2
Unchanged wiki pages: 30
```

### 5.4 查询（V0.5）

```bash
llm-wiki query "插件权限系统是如何工作的？"
```

返回答案和引用来源。

### 5.5 搜索（V0.2）

```bash
llm-wiki search "plugin permission"
```

返回 Wiki page/section，而非直接回答。

### 5.6 启动服务（V0.4）

```bash
llm-wiki serve --port 8080
```

---

## 6. 总体架构

```text
                         ┌─────────────────┐
                         │   Source Docs   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Source Scanner  │
                         └────────┬────────┘
                                  │
                         Manifest / Diff
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Markdown Parser │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Doc Analyzer    │
                         │      LLM        │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Knowledge Model │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Wiki Planner   │
                         │      LLM        │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Wiki Compiler   │
                         │      LLM        │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
               Markdown Wiki                SQLite
                    │                           │
                    │               ┌───────────┼───────────┐
                    │               ▼           ▼           ▼
                    │             State      Citation     Graph
                    │
                    ▼
             Retrieval Layer
          ┌─────────┼──────────┐
          ▼         ▼          ▼
        FTS       Vector      Graph
          └─────────┼──────────┘
                    ▼
              Context Builder
                    │
                    ▼
                 Query API
```

---

## 7. Rust Workspace

推荐：

```text
llm-wiki/
├── Cargo.toml
├── crates/
│   ├── llm-wiki-core/
│   ├── llm-wiki-source/
│   ├── llm-wiki-markdown/
│   ├── llm-wiki-llm/
│   ├── llm-wiki-storage/
│   ├── llm-wiki-search/
│   ├── llm-wiki-cli/
│   └── llm-wiki-server/
├── prompts/
│   ├── document-analysis.md
│   ├── section-analysis.md
│   ├── document-synthesis.md
│   ├── wiki-planning.md
│   ├── wiki-compilation.md
│   └── query-answer.md
├── migrations/
├── evals/
├── test-data/
│   └── docs/
└── docs/
```

### 7.1 llm-wiki-core

定义领域模型和核心接口：

- SourceDocument
- DocumentSection
- KnowledgeNode
- Entity
- Relation
- Claim
- Citation
- WikiPage
- WikiPlan
- ChangeSet
- BuildPlan
- SearchResult
- KnowledgeContext

禁止直接依赖：

- Axum
- Clap
- OpenAI SDK
- SQLite driver
- Qdrant/Milvus

### 7.2 llm-wiki-source

负责：

- 文件扫描。
- include/exclude。
- hash。
- source manifest。
- change detection。

### 7.3 llm-wiki-markdown

负责 Markdown AST 和结构化文档模型。

推荐 Rust Markdown parser 使用成熟 CommonMark/GFM AST 库；具体库在实现阶段根据 GFM、source-position 和维护状态选择。

### 7.4 llm-wiki-llm

负责：

- Provider abstraction。
- OpenAI-compatible provider。
- structured output。
- retry。
- timeout。
- concurrency limit。
- token usage。

### 7.5 llm-wiki-storage

V0.x：SQLite。

负责：

- state。
- manifest。
- knowledge model。
- citations。
- graph。
- build history。

### 7.6 llm-wiki-search

负责：

- FTS。
- rank fusion。
- 后续 vector search。
- graph expansion。
- context building。

### 7.7 llm-wiki-cli

Clap 等 CLI framework，仅做参数解析和调用 application service。

### 7.8 llm-wiki-server

Axum HTTP server，仅作为 transport adapter。

---

## 8. Source Scanner

### 8.1 支持范围

V0.1：

```text
*.md
*.markdown
*.mdx
```

`.mdx` 按 Markdown 解析，但必须先经过确定性的组件降级（见 §9）。

### 8.2 扫描结果

```rust
struct SourceFile {
    id: SourceId,
    relative_path: PathBuf,
    content_hash: String,
    size: u64,
    modified_at: Option<DateTime<Utc>>,
}
```

不得直接使用绝对路径。`SourceLocatorKey = hash(workspace + normalized relative path)` 用于定位当前文件；`SourceId` 是持久 Source Registry 分配的 opaque identity。V0.1 可让 path 变化表现为 delete/add；V0.2 在无歧义 rename match 成功时将新的 locator 绑定到原 SourceId，从而保持 provenance 与依赖连续性。

### 8.3 Hash

使用 SHA-256 或 BLAKE3。

内容 hash 是变更判断依据，mtime 只能用于快速预筛选。

### 8.3.1 Path Normalization

所有路径先以 source root 为边界 canonicalize，再以 `/` 作为存储分隔符、Unicode NFC 作为存储形式；解析 `.`，拒绝任何规范化后仍会逃离 source root 的 `..`。保留原始相对路径仅用于人类展示，`SourceLocatorKey` 必须使用规范化路径。

为满足 Windows/macOS 交付，Scanner 还必须用 Unicode case-folded 规范化路径检测碰撞：在同一 source root 中两个不同路径若映射到同一 portable key，必须以 `PathCollisionError` 拒绝 build，而不是在不同平台选择不同文件。不得仅依赖宿主文件系统的大小写行为。

### 8.4 Ignore

支持：

```toml
[source]
include = ["**/*.md", "**/*.mdx"]
exclude = [
  "node_modules/**",
  ".git/**",
  ".llm-wiki/**",
  "wiki/**"
]
```

`.llm-wiki/**` 和配置的 `wiki_dir/**` 是不可覆盖的硬性排除项：即使 include 或用户自定义 exclude 出现冲突，Scanner 也不得将它们作为 source ingest。它们包含 state、cache、publish journal 和派生 Markdown；重新摄入会产生自引用 citation 和代际漂移。CLI 必须在配置把 source root 与 `wiki_dir` 重叠时拒绝执行。

---

## 9. Markdown Parser

不要直接固定字符切 chunk。

输入：

```markdown
# Plugin System

Intro...

## Architecture

...

### Runtime

...
```

解析为：

```rust
struct ParsedDocument {
    source_id: SourceId,
    title: Option<String>,
    frontmatter: Map<String, Value>,
    sections: Vec<DocumentSection>,
    links: Vec<DocumentLink>,
}

struct DocumentSection {
    id: SectionId,
    heading: Option<String>,
    heading_level: u8,
    heading_path: Vec<String>,
    content: String,
    source_range: Option<SourceRange>,
}
```

必须保留 Heading Path：

```text
Plugin System > Architecture > Runtime
```

它将成为 citation 的重要定位信息。

frontmatter 仅作为受限元数据处理，不得覆盖系统字段（`id`、`generated`、`schema_version`、citation 等）。解析器必须保留 ATX/Setext heading、重复 heading、代码块、HTML、链接和 source range，并为无标题文档、非法 frontmatter、不可解析 Unicode 和二进制伪 Markdown 提供确定性诊断。

`.mdx` 在 Markdown 解析前必须经过确定性组件降级：内嵌 JSX 组件（如 `<Callout>`、`<Tabs>`）只保留其文本子节点和属性中的可读内容，frontmatter、代码块和常规 Markdown 语法不受影响。降级规则必须版本化并计入 parser version；无法识别的 JSX 构造不得静默丢弃，必须产生诊断并在 lint 中报告。降级只影响送入 analysis 的文本，不改写 Source。

语言标注在解析阶段确定，优先级为：frontmatter 显式 `lang`/`language` 字段 > 文件名语言后缀（如 `.cn.md`、`.zh-CN.mdx`）> 内容 script 启发式。结果写入文档与 Wiki frontmatter 的 `language` 字段，作为 §20 TextAnalyzer 选择 analyzer 的依据；无法判定时标记 `und`、使用默认 analyzer 并产生诊断。

---

## 10. 大文档处理

如果单文档超过模型安全上下文阈值，不允许简单截断。

采用 hierarchical analysis：

```text
Document
   ↓
Sections
   ↓
Section Analysis
   ↓
Section Knowledge
   ↓
Document Synthesis
```

每个 section 保留 SourceSection ID，最终 synthesis 只能引用 section analysis 已有的 source references。

配置：

```toml
[analysis]
max_input_tokens = 32000
section_target_tokens = 6000
max_concurrency = 4
```

具体 token 数由模型配置覆盖。

单个 section 同样不得被截断。当其超过 `section_target_tokens` 时，Parser 必须按 block 边界拆成 `SectionSegment`；segment 保留同一 `SectionId`、连续 source range 和稳定 segment ordinal。任何 citation 仍指向原始 section 的精确 range，而不是只指向截断后的文本。

---

## 11. Document Analysis

LLM 不直接生成最终 Wiki。

第一阶段生成结构化 `DocumentAnalysis`：

```rust
struct DocumentAnalysis {
    summary: String,
    entities: Vec<EntityCandidate>,
    concepts: Vec<ConceptCandidate>,
    claims: Vec<ClaimCandidate>,
    relations: Vec<RelationCandidate>,
    topics: Vec<String>,
}
```

### 11.1 Claim

```rust
struct ClaimCandidate {
    text: String,
    source_section_id: SectionId,
    evidence_ranges: Vec<SourceRange>,
    evidence_digest: String,
    confidence: Option<f32>,
}
```

禁止 LLM 创建不存在的 Source ID。

`ClaimCandidate` 是可审计的原子事实，而不是一段笼统摘要。每个 claim 必须至少提供一个位于 `source_section_id` 内的精确 source range；应用代码必须校验 range 边界、对应文本的 digest 及 source hash。无法给出有效 evidence 的输出不得入库或写入 Wiki。`confidence` 只用于排序或人工复核优先级，不能代替 evidence。

evidence/referential 校验失败的处置固定为：

1. 将 validator 的机器可读原因提供给模型，且仅 repair 一次；
2. 再失败时创建 `RejectedClaim` 记录（原始候选、source、reason、build ID），不得生成 knowledge/citation；
3. 在 build summary、observability 和 `lint` 中报告 rejected 数量与原因。若某个分析单元的 rejected claim 比例超过配置阈值（默认 10%），则该分析单元失败，build 不得发布。

这不是静默丢弃：被拒绝候选可审计、可重跑，也不会以无来源事实污染 Wiki。

输出 Schema 验证失败时：

1. 尝试一次 repair。
2. 再失败则任务失败。
3. 不允许静默丢弃。

---

## 12. Knowledge Model

V0.x 不追求复杂 ontology。

核心：

```text
Entity
Concept
Claim
Relation
Topic
Citation
```

### 12.1 Entity

```rust
struct Entity {
    id: KnowledgeNodeId,
    canonical_name: String,
    entity_type: String,
    aliases: Vec<String>,
    description: Option<String>,
}
```

### 12.1.1 Stable Knowledge Registry

所有 `Entity`、`Concept`、`Claim` 和可被页面引用的知识节点都必须先在持久化 Registry 中分配 opaque `KnowledgeNodeId`。推荐格式为 `kn_` 前缀的 ULID；实现可以选择其他不可从名称或内容派生、可排序且全局唯一的格式，但不得在每次 build 根据文本重新计算 ID。

Registry 至少保存：

```text
id, node_kind, canonical_key, status,
merged_into, created_build_id, retired_build_id
```

`status` 为 `active`、`merged`、`rejected` 或 `retired`。deterministic normalization 先用 canonical key 查找候选，LLM 仅提出 merge suggestion；应用代码确认合并后保留主节点 ID，把次节点标为 `merged` 并设置 `merged_into`，不得删除、重新分配或改变既有引用的语义。页面、relation、citation 和 cache 只保存 Registry ID，名称与 alias 可变化但不能替代身份。

`RegistryRevision` 是每个 workspace 单调递增的 `u64`。任何已提交的节点创建、合并、状态迁移、`merged_into` 变化或影响 resolution 的 canonical key/alias 变化，都必须在同一数据库事务中使 revision 恰好递增一次；回滚事务不得推进 revision。每个 build、plan、cache record 保存其读取的 revision，只有 revision 相同的结果才能复用。

该 Registry 是 V0.1 document analysis/knowledge persistence 的前置条件，也是 V0.2 增量依赖图和 plan diff 的稳定锚点。

### 12.2 Relation

```rust
struct Relation {
    source: KnowledgeNodeId,
    relation_type: String,
    target: KnowledgeNodeId,
    citations: Vec<CitationId>,
}
```

### 12.3 Citation

```rust
struct Citation {
    id: CitationId,
    source_id: SourceId,
    section_id: Option<SectionId>,
    source_range: Option<SourceRange>,
    source_hash: String,
    evidence_digest: Option<String>,
    heading_path: Vec<String>,
}
```

---

## 13. Entity Resolution

不同文档可能出现：

```text
Plugin Manager
plugin manager
PluginManager
插件管理器
```

V0.1 使用两阶段：

1. deterministic normalization；
2. LLM-assisted merge suggestion。

LLM 只能提出 merge candidate，核心系统执行并保存 resolution 结果。

不要在 V0.1 追求自动完美实体消歧。

中英平行语料（同一主题两种语言的成对文件）在 V0.1 建议按语言拆分 workspace 分别构建：两阶段 ER 的 deterministic normalization 只在语言内部有效，跨语言实体归并不做承诺；混合 ingest 会产生重复实体与重复页面。允许单一 workspace ingest 多语言文件（语言标注见 §9），此时平行文件预期各自成页；把平行主题归并为单一实体/页面及跨语言检索属于后续版本能力，启用前必须在 Eval 中验证。

---

## 14. Wiki Planning

这是系统与普通 RAG 最大区别之一。

错误实现：

```text
source A → wiki A
source B → wiki B
source C → wiki C
```

正确目标不是将全部知识节点一次性塞入一个 LLM context，而是构造可扩展的全局计划：

```text
Sources
   ↓
Knowledge Model
   ↓
Deterministic Clustering
   ↓
Cluster Summary
   ↓
Local Plans
   ↓
Plan Merge / Global Reconciliation
   ↓
Global Wiki Plan
```

例如：

```text
docs/plugin/architecture.md
docs/plugin/security.md
docs/plugin/development.md
```

可规划为：

```text
wiki/concepts/plugin-system.md
wiki/architecture/plugin-runtime.md
wiki/security/plugin-security.md
wiki/guides/plugin-development.md
```

规划必须分层执行：先按 entity/concept links、source 目录和主题对 Knowledge Nodes 聚类；每个 cluster 在独立 token budget 内生成可引用的 summary 和 local plan；最后仅以排序后的 cluster summary/local-plan hash 进行全局 reconciliation。全局阶段不得直接消费全部 source 或全部 claim 正文。

每层都有独立 `max_plan_input_tokens`、稳定排序和缓存 key：cluster summary key 由已排序的 `KnowledgeNodeId + content_hash` 集合组成；local plan key 由 cluster summary hash、planner version 和配置组成；reconciliation key 由已排序的 local plan hash、Registry revision 和配置组成。超过单层预算时继续细分 cluster，而不是截断或隐式放大上下文窗口。

V0.1 至少用 30–100 文档 Eval fixture 验证该路径；若配置禁用分层规划，CLI 必须在预估输入超过 `max_plan_input_tokens` 时失败并提示启用，而不能执行不完整的 global plan。

### 14.1 WikiPlan

```rust
struct WikiPlan {
    pages: Vec<WikiPagePlan>,
}

struct WikiPagePlan {
    id: WikiPageId,
    slug: String,
    title: String,
    category: String,
    purpose: String,
    knowledge_refs: Vec<KnowledgeNodeId>,
    source_refs: Vec<SourceId>,
    related_pages: Vec<WikiPageId>,
}
```

---

## 15. Wiki Compilation

Compiler 输入：

- WikiPagePlan。
- Selected Knowledge Nodes。
- Source Citations。
- Related Pages。

输出 Markdown。

示例：

```markdown
---
id: plugin-system
title: Plugin System
category: concepts
sources:
  - docs/plugin/architecture.md
  - docs/plugin/security.md
---

# Plugin System

...

## Architecture

...

## Security

...

## Related

- [[Plugin Runtime]]
- [[Plugin Security]]
```

### 15.1 Grounding Rule

Prompt 必须明确：

- 只能根据传入 Knowledge + Sources 写事实。
- 不确定内容不得补全。
- 不允许根据模型自身知识扩展项目事实。
- 每个重要 section 至少拥有一个 source reference。

这里的“重要”不能成为不可验证的例外：任何包含可证伪项目事实的句子都必须由一个或多个 atomic claim 支撑，并在 Markdown 中关联其 citation。纯导航、明确标注为推测的说明和由标题/元数据确定的内容可不逐句引用。lint 必须能报告无 citation 的事实段落及其对应 claim。

### 15.2 生成产物所有权

`wiki_dir/generations/{build_id}/` 是默认由编译器完全管理的派生产物，`wiki_dir/current.json` 是唯一公开入口。V0.1 不支持直接编辑生成页面：下次 build 可以替换其全部内容，CLI 和页面 frontmatter 必须明确标记 `generated: true`、`schema_version`、`language` 及生成的 build ID。

需要人工维护的内容必须位于 source docs，或通过未来版本的显式 overlay 目录提供；overlay 必须独立保存、带来源和优先级，并在生成时合并。编译器不得静默保留、猜测合并或丢弃 `wiki_dir/generations/` 内的手工改动；发现当前 generation 的内容 hash 不匹配时，build 应失败并提示用户迁移、删除或将内容移入 overlay。

WikiLink 使用 `[[Page Title]]` 或 `[[Page Title|显示文本]]` 语法，链接目标解析到 `WikiPageId` 后持久化；显示 title/slug 的修改不得打断链接。frontmatter 的 `schema_version` 允许 parser/lint 做兼容性检查，`language` 表示页面正文语言并供搜索 analyzer 选择语言策略。

---

## 16. Citation 模型

建议 Wiki Markdown 中保存机器可解析引用，而不是只写自然语言。

例如：

```text
<!-- llm-wiki:cite source="docs/plugin/security.md" section="Permission Model" -->
```

展示层可以渲染成：

```text
[docs/plugin/security.md > Permission Model]
```

数据库同时保存 citation mapping。

Markdown 文件是人类可读结果，SQLite 是机器状态；两者不要互相替代。

---

## 17. Wiki Graph

V0.x 不使用 Neo4j。

节点：

- WikiPage
- Entity
- Concept

边：

- links_to
- depends_on
- part_of
- uses
- implements
- related_to
- defined_in

SQLite：

```sql
CREATE TABLE graph_nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    label TEXT NOT NULL
);

CREATE TABLE graph_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY(source_id) REFERENCES graph_nodes(id),
    FOREIGN KEY(target_id) REFERENCES graph_nodes(id)
);
```

为 source_id / target_id / relation_type 建索引。

---

## 18. State Database

建议核心表：

```text
sources
source_sections
document_analyses
knowledge_nodes
claims
citations
relations
wiki_pages
wiki_page_sources
wiki_page_knowledge
wiki_links
graph_nodes
graph_edges
builds
build_tasks
```

### 18.1 Build

```text
build_id
started_at
finished_at
status
source_snapshot_hash
model
prompt_version
```

必须保存：

- model identifier。
- prompt version。
- compiler version。
- parser/normalizer version。
- schema version。
- 生效配置的 canonical hash。
- 每个 source 的 content hash，以及可选的不可变 source snapshot。

否则未来无法复现“为什么 Wiki 变了”。

每次 build 还必须产生 `BuildFingerprint`，至少由 source snapshot hash、模型与 provider 参数、prompt/schema/parser/compiler 版本、实体归并规则和生效配置 hash 构成。缓存命中、是否可复现、是否需要 re-plan 都以此指纹判断，而不是只比较 source 文件 hash。

默认仅承诺“使用当前仍可读取的 source 可重建”。启用 `history.store_source_snapshots = true` 后，系统保存受访问控制的 source snapshot，才承诺可重建历史 build。snapshot 的保留期、加密及删除策略必须可配置；不能在未声明的情况下持久保存可能包含敏感内容的源文件。

---

## 19. 增量编译

这是 V0.2 的核心能力。

### 19.1 Source Diff

```text
previous manifest
       +
current manifest
       ↓
ChangeSet
```

```rust
struct ChangeSet {
    added: Vec<SourceId>,
    modified: Vec<SourceId>,
    deleted: Vec<SourceId>,
    unchanged: Vec<SourceId>,
}
```

### 19.2 Dependency Tracking

必须保存：

```text
Source → Knowledge Node
Knowledge Node → Wiki Page
Wiki Page → Wiki Page
```

修改：

```text
plugin/security.md
```

得到：

```text
plugin/security.md
       ↓
permission-model
       ↓
plugin-security
plugin-system
```

只重新分析和编译 affected set。

该规则只适用于 WikiPlan 未改变且实体 resolution 结果未改变的情况。以下任一条件成立时，`build` 必须停止在当前 generation 之前，将 workspace 标记为 `REPLAN_REQUIRED`，而不是自动执行 global re-plan：

- source 的新增、删除或重命名改变主题覆盖范围；
- 抽取出的 entity/concept/relation 集合或 merge result 改变，且新/变更节点不能确定性映射到既有页面的 `knowledge_refs`、类别和页面边界；
- planner prompt、schema、规则或 BuildFingerprint 中影响规划的字段改变；
- 页面失去全部知识，或新知识无法映射到既有页面；
- Registry revision、层级聚类策略或 planner BuildFingerprint 变化。

用户必须显式运行 `llm-wiki replan` 才能进行结构性重构。`llm-wiki replan --dry-run` 不调用 Compiler 或发布 generation，必须输出触发原因、预计影响页面、plan merge/split/retire、预计 LLM 调用数和成本上界；不带 `--dry-run` 的命令在用户确认后生成新 plan。这样将高成本、高影响的知识架构变化与日常维护 build 分离，避免一次 source 编辑悄然重写大量页面。

global re-plan 必须把新 plan 与当前 plan 做稳定 ID diff：保留语义不变页面的 `WikiPageId`，记录 merge/split/retire 关系并重编译所有受影响页面。无法可靠分类的变更必须保守地进入 `REPLAN_REQUIRED`；正确性优先于最小重编译集合。

判定顺序固定为：先以 Registry ID、既有 `knowledge_refs`、页面 category 和局部 relation 规则尝试确定性映射；映射成功且不改变 cluster 成员、页面边界、related page topology 或当前 plan 的语义时，按局部 UPDATE 处理。否则标记 `REPLAN_REQUIRED`。每次判定必须记录 mapping outcome、触发原因、候选页面数、预计重编译页面数和成本；Observability 按原因聚合该分布，供后续版本决定是否引入有界局部 plan 更新。

### 19.3 Deleted Source

删除源文件时必须：

1. 标记 Source removed。
2. 删除/失效仅由该 source 支撑的 knowledge。
3. 找出 affected wiki pages。
4. 重新编译。
5. 如果页面失去全部有效知识，则删除或标记 obsolete。

不能留下 ghost knowledge。

---

## 20. Search V0.2

优先实现全文检索。

索引对象：

- Wiki title。
- headings。
- page body。
- aliases。

首版使用 SQLite FTS5（若目标构建环境的 SQLite feature 可用）；否则抽象 FullTextIndex 并提供等价本地实现。不得把默认 Latin tokenizer 当作中文检索策略：索引层必须通过 `TextAnalyzer` 抽象同时支持 Unicode Latin word token 和 CJK unigram/bigram token，查询与索引使用同一 normalization（NFKC、大小写、全半角和常见标点）。

Eval 必须包含 2–4 字中文词、中文短句、英文术语、中文/英文混合术语和 alias 查询；每类都验证 Top-K。若运行时 SQLite 不支持所选 tokenizer，`doctor` 必须报出降级模式，系统使用等价的预分词字段或拒绝启用 `full_text`，不得静默返回不可用的中文搜索。

API：

```rust
trait FullTextSearch {
    async fn search(&self, query: &str, limit: usize)
        -> Result<Vec<SearchHit>>;
}
```

---

## 21. Vector Retrieval V0.3

需要向量能力，但 V0.x 不要求独立 Vector DB。

Embedding 单元优先：

```text
Wiki Section
```

而不是原始固定长度 chunk。

记录：

```text
embedding_id
wiki_page_id
section_id
model
model_version
dimension
content_hash
vector
```

只有 `content_hash` 变化才重新 embedding。

抽象：

```rust
#[async_trait]
trait EmbeddingProvider: Send + Sync {
    fn model_id(&self) -> &str;
    fn dimension(&self) -> usize;
    async fn embed(&self, inputs: Vec<EmbeddingInput>)
        -> Result<Vec<Embedding>>;
}
```

`EmbeddingProvider` 与 `VectorStore` 独立：前者负责模型调用、batch、timeout、usage 和 provider version，后者只负责持久化与近邻检索。每个 vector record 必须保存 provider/model ID、dimension、normalized content hash 和 embedding schema version；其中任一项变化都使该 section 的 embedding cache 失效。

```rust
trait VectorStore {
    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<()>;
    async fn delete(&self, ids: Vec<VectorId>) -> Result<()>;
    async fn search(&self, vector: &[f32], limit: usize)
        -> Result<Vec<VectorHit>>;
}
```

首版 LocalVectorStore；未来可提供：

```text
QdrantVectorStore
MilvusVectorStore
```

Core 不感知具体实现。

---

## 22. Hybrid Retrieval V0.5

目标：

```text
Query
  │
  ├── Full Text
  ├── Vector
  └── Graph Expansion
        ↓
     Fusion
        ↓
     Rerank
        ↓
 Context Builder
```

首版 fusion 建议 Reciprocal Rank Fusion，避免过早设计复杂人工权重。

Graph Expansion 默认限制：

```text
depth <= 1
max expanded nodes <= 10
```

防止上下文爆炸。

---

## 23. Context Builder

这是未来接入 Agent 时最重要的 API 之一。

输入：

```rust
struct ContextRequest {
    query: String,
    max_tokens: usize,
    max_pages: usize,
    include_sources: bool,
}
```

输出：

```rust
struct KnowledgeContext {
    query: String,
    items: Vec<ContextItem>,
    citations: Vec<Citation>,
    estimated_tokens: usize,
}
```

职责：

1. 检索候选。
2. 去重。
3. graph expansion。
4. rerank。
5. token budget packing。
6. 输出 provenance。

它不能直接依赖某个 Agent Harness。

---

## 24. Query Engine

`search` 与 `query` 必须区分。

### search（V0.2）

只返回知识：

```text
query → SearchResult[]
```

### query（V0.5）

```text
query
 ↓
Context Builder
 ↓
LLM
 ↓
Answer + citations
```

未来 Agent 通常优先调用 `search/context`，而不是 `query`，避免 Wiki Engine 与 Agent 双重 reasoning。

在 search/query API 交付之前，`wiki_dir` 中的编译产物本身就是可被任何外部 LLM 直接消费的 grounded 上下文；这不是本产品的问答能力承诺——产品化检索与问答分别自 V0.2（search）与 V0.5（query）交付，也正因如此 §37.4 要求把全文直塞基线纳入对比。

---

## 25. LLM Provider

定义抽象：

```rust
#[async_trait]
trait LlmProvider: Send + Sync {
    async fn generate(&self, request: LlmRequest)
        -> Result<LlmResponse>;
}
```

V0.1 实现：

```text
OpenAICompatibleProvider
```

配置：

```toml
[llm]
provider = "openai-compatible"
base_url = "http://localhost:8000/v1"
model = "..."
api_key_env = "LLM_WIKI_API_KEY"
timeout_seconds = 120
max_concurrency = 4
```

API Key 不写入 project config。

---

## 26. Prompt 管理

Prompt 不应硬编码在 Rust source 中。

```text
prompts/
├── document-analysis.md
├── section-analysis.md
├── document-synthesis.md
├── wiki-planning.md
├── wiki-compilation.md
└── query-answer.md
```

每个 prompt 有 version，例如：

```yaml
name: wiki-compilation
version: 1
```

Build 保存 prompt version。

Prompt 修改应能够触发明确的 rebuild 策略，而不是悄悄影响结果。

---

## 27. Concurrency

LLM 调用使用 Tokio async。

需要全局 Semaphore：

```text
max_concurrency = 4
```

同时支持：

- request timeout。
- exponential backoff。
- 429 retry。
- 5xx retry。
- cancellation。

不要无限 retry。

建议默认：

```text
max retries = 3
```

---

## 28. Cache

LLM cache key：

```text
hash(
  task_type
  + source/content hash
  + model
  + prompt_version
  + schema_version
  + parser_normalizer_version
  + effective_config_hash
)
```

相同输入不得重复消耗 LLM token。

缓存 key 必须使用对应任务的 `BuildFingerprint` 字段；prompt、模型参数、解析器、schema、实体归并策略或影响输出的配置任一变化都必须失效相关缓存。缓存记录应保存 response schema version 和 source snapshot hash，避免跨不兼容版本复用。

planning 类任务不能使用单个 source hash 作为 key：cluster summary、local plan 和 global reconciliation 分别使用 §14 定义的已排序输入集合 hash，且必须包含 Registry revision、planner prompt/schema、有效配置和 token budget。删除、合并或重排任一输入都必须使对应层及其下游缓存失效。

Cache 可以保存 SQLite 或 `.llm-wiki/cache`。

---

## 29. CLI 设计

### init

```bash
llm-wiki init
```

### scan

```bash
llm-wiki scan ./docs
```

只扫描和展示 diff，不调用 LLM。

### build

```bash
llm-wiki build ./docs
```

`build` 只执行缓存命中、source diff 与可证明安全的局部更新。发现 `REPLAN_REQUIRED` 时返回非零 `ReplanRequired` 错误、保留当前 generation，并展示触发原因；不得自动升级为全局规划。

### replan（V0.2）

```bash
llm-wiki replan --dry-run
llm-wiki replan
```

前者输出全局计划变更和成本估算；后者显式执行分层规划、plan diff 和受影响页面重编译。

### status

```bash
llm-wiki status
```

### search

```bash
llm-wiki search "plugin runtime"
```

### query

```bash
llm-wiki query "How does plugin security work?"
```

### doctor（V0.1）

```bash
llm-wiki doctor
```

检查 config、source root、hard excludes、`wiki_dir` 可写性、SQLite、LLM endpoint、模型配置和是否存在未恢复的 publish journal。

### lint

```bash
llm-wiki lint
```

检查：

- broken wikilinks。
- missing source。
- stale citation。
- orphan page。
- unsupported claim。
- duplicate page slug。

### serve（V0.4）

```bash
llm-wiki serve --host 127.0.0.1 --port 8080
```

---

## 30. HTTP API

### Health

```http
GET /health
```

### Status

```http
GET /v1/status
```

### Build

```http
POST /v1/build
```

Request：

```json
{
  "source_id": "default"
}
```

`source_id` 必须预先在项目配置中映射到受允许的 source root；服务端 API 不接受任意本地路径。CLI 可以接受路径参数，但仍必须 canonicalize 后限制在当前 workspace 或显式允许的 root 内。

长任务建议返回 job：

```json
{
  "job_id": "...",
  "status": "queued"
}
```

### Job

```http
GET /v1/jobs/{job_id}
```

### Search

```http
POST /v1/search
```

### Context

```http
POST /v1/context
```

这是未来 Agent integration 的关键接口。

### Query

```http
POST /v1/query
```

### Pages

```http
GET /v1/pages
GET /v1/pages/{id}
```

### API 契约与远程暴露

所有 JSON endpoint 必须定义请求/响应 schema、稳定错误码、`request_id` 和 `build_id`。列表接口必须采用 cursor pagination；Search/Context 必须返回实际使用的 generation、截断标志和 citation。写操作接受 idempotency key，相同 key 在有效期内必须返回同一 job，而不是重复启动 LLM 消耗。

Server 默认仅绑定 `127.0.0.1`/`::1`，且 remote mode 必须显式启用。remote mode 至少要求：认证 token（仅通过环境变量或 secret store 提供）、TLS 由部署层保证、配置化 allowed source roots、请求大小/并发/job 队列上限和按调用方的速率/成本预算。未认证或越界 source 的请求返回明确的 4xx，绝不能尝试扫描。

---

## 31. Build Job 状态机

```text
QUEUED
  ↓
SCANNING
  ↓
PARSING
  ↓
ANALYZING
  ↓
PLANNING
  ↓
COMPILING
  ↓
INDEXING
  ↓
COMPLETED
```

异常：

```text
FAILED
CANCELLED
INTERRUPTED
REPLAN_REQUIRED
```

状态必须持久化，Server 重启后不能把 RUNNING 任务永久留在假状态；启动时需要 recovery，将未完成任务标记 interrupted/failed 或根据未来策略恢复。

取消是协作式的：在 ANALYZING/PLANNING/COMPILING 阶段停止新 LLM 请求并清理未发布 generation；一旦进入 publish critical section，不再接受取消，必须完成或按 publish journal 恢复。状态还应包含 `INTERRUPTED`，并记录 `failure_code`、可安全重试标志和关联的 build fingerprint。

---

## 32. Config

`.llm-wiki/config.toml`：

```toml
[project]
name = "example"
wiki_dir = "./wiki" # canonical managed generations + current pointer

[source]
root = "./docs"
include = ["**/*.md", "**/*.mdx"]
exclude = ["wiki/**", ".llm-wiki/**", ".git/**", "node_modules/**"]

[llm]
provider = "openai-compatible"
base_url = "http://localhost:8000/v1"
model = "model-name"
api_key_env = "LLM_WIKI_API_KEY"
max_concurrency = 4
timeout_seconds = 120

[analysis]
max_input_tokens = 32000
section_target_tokens = 6000
max_plan_input_tokens = 32000
max_rejected_claim_ratio = 0.10

[search]
full_text = true
vector = false
graph = true

[build]
incremental = true

[server]
bind = "127.0.0.1"
remote_enabled = false
auth_token_env = "LLM_WIKI_SERVER_TOKEN"
```

配置优先级固定为 CLI 显式参数 > 环境变量（仅 secrets/部署参数）> 项目 config > 安全默认值。解析后必须输出不含 secret 的 effective config；任何会改变 BuildFingerprint 的配置都必须在 build metadata 中记录。

`wiki_dir` 是唯一的公开、受编译器管理的 Wiki 根目录：其 `generations/{build_id}/` 保存不可变 Markdown generation，`current.json` 指向唯一当前可见 generation。`.llm-wiki/` 只保存 state、cache 和非公开运行时元数据。`wiki_dir` 不是可由用户任意编辑的工作目录；其管理和 hand-edit 检测规则见 §15.2。`wiki_dir` 与 source root 重叠是无效配置。

---

## 33. Observability

至少记录：

```text
build duration
source count
changed source count
LLM request count
LLM retry count
input tokens
output tokens
cache hit rate
wiki page count
citation count
rejected claim count / reason
replan-required count / reason
replan mapping outcome / estimated cost
failed tasks
```

使用 `tracing` 体系输出 structured logs。

每次 build 分配 `build_id`，所有日志带 build_id。

---

## 34. Error Model

统一错误类别：

```text
SourceError
PathCollisionError
ParseError
StorageError
LlmError
SchemaValidationError
EvidenceValidationError
PlanningError
ReplanRequired
CompilationError
IndexError
BudgetExceeded
PublishRecoveryError
ConfigError
```

CLI 根据错误类型返回非 0 exit code。

`PathCollisionError` 必须列出冲突的相对路径；`EvidenceValidationError` 仅在 repair/rejected 比例超过阈值时使分析单元失败；`BudgetExceeded` 不得发布部分 generation；`ReplanRequired` 是可操作状态，必须给出触发原因并指向 `replan --dry-run`；`PublishRecoveryError` 必须保留最后一致 generation 并要求 recovery，不得猜测当前版本。

禁止：

```text
LLM 失败 → 生成空页面 → build success
```

关键阶段失败应使 Build 明确 FAILED。

---

## 35. Atomic Build

不能直接覆盖当前可用 Wiki 后再发现 build 失败。

每个 build 写入 `wiki_dir` 下不可变 generation，而不是直接覆盖当前可见 Wiki：

```text
{wiki_dir}/generations/{build_id}/  # 不可变 Markdown generation
{wiki_dir}/current.json             # 仅保存当前可见 build_id
{wiki_dir}/.publish-journal.json    # 极短暂的发布意图记录
```

流程：

```text
compile immutable generation
      ↓
validate
      ↓
写入 versioned metadata / indexes，并标记 build READY
      ↓
记录 publish intent（old_build_id, new_build_id）
      ↓
原子替换 current.json 指针
      ↓
单一数据库事务切换 active_build_id 并标记 COMPLETED
      ↓
清除 publish intent
```

`wiki_dir/current.json` 与 `active_build_id` 不一致时，运行时不得把新旧 generation 混合提供；在 build lock 下按 publish journal 恢复到明确的旧版本或完成已验证的新版本。启动恢复必须覆盖“进程在每一步崩溃”的情况，并记录恢复结果。

发布指针和 generation 必须位于同一文件系统。Windows 上不能假定目录替换原子；只允许原子替换小型指针文件，并对短暂的 sharing violation 有有界重试。失败时始终保留上一版可读的 generation，后台清理只能删除不再被 current pointer 或保留策略引用的 generation。

---

## 36. Wiki Lint

`lint` 是产品核心能力，不是附加脚本。

检查项：

### Citation Integrity

所有 citation source 必须存在且 hash 状态可识别。

### Broken Link

`[[Plugin Runtime]]` 必须有目标。

### Orphan Page

没有入口和关系的页面给 warning。

### Unsupported Section

重要 Wiki section 没有 citation 给 warning/error。

### Duplicate Concept

通过 normalized title/alias 检测疑似重复。

### Stale Page

依赖 source 已改变但页面尚未成功重新编译。

---

## 37. Eval Framework

验证目录：

```text
evals/
├── dataset.yaml
├── questions.yaml
└── expected/
```

### 37.1 测试数据要求

首个 benchmark 使用真实风格 docs：

- 30–100 Markdown/MDX，其中至少含少量 `.mdx` 文件与一组中英平行文档。
- 至少 4 层目录中的部分文件。
- 存在跨文件知识。
- 存在同义术语。
- 存在引用和相对链接。
- 存在 README/overview/guide/reference 等不同类型。

### 37.2 核心指标

#### Source Coverage

重要 source knowledge 是否进入 Wiki。

#### Citation Correctness

Wiki claim 是否被引用 source 支撑。

#### Cross-document Synthesis

能否把多个 source 的同一主题融合成统一页面。

#### Hallucination Rate

Wiki 中无法从 source 支撑的事实比例。

#### Link Quality

Wiki page 之间关系是否合理。

#### Incremental Correctness

修改 source 后是否只更新必要知识，且无 stale content。

#### Retrieval Quality

测试问题的 relevant page 是否进入 Top-K。

### 37.3 V0.1 可执行门槛

Eval fixture 必须具有人工标注的 source facts、允许的 citation ranges、预期跨文档页面和至少 20 个检索问题；fixture 至少包含 30 个 Markdown/MDX 文档。CI 中使用 `FakeLlmProvider` 跑确定性评测，真实 LLM 评测只作为显式触发的补充。

V0.1 的 release gate：

- Source Coverage：人工标注的重要事实覆盖率 ≥ 90%。
- Citation Correctness：抽样及 fixture 全量关键 claim 的有效 citation 比例 ≥ 95%，且不存在无效 range/hash。
- Hallucination Rate：可证伪的无来源 claim 比例 ≤ 5%。
- Cross-document Synthesis：至少一个预期页面必须合并三个及以上 source，并保留各自 citation。
- Rebuild Determinism：相同 BuildFingerprint 的第二次 build 不新增 LLM 请求，generation 的结构化清单（页面 ID、citation mapping、links）完全一致。

每项指标必须在 `evals/` 中定义分母、标注格式、评分脚本和失败报告；阈值变化视为产品契约变更，需版本化记录。

### 37.4 基线对比

产品成功标准中“明显优于 Raw Chunk RAG”必须有可复现实验，而非主观描述。Eval 必须提供一个固定的 baseline：同一 source snapshot、同一问题集、相同模型、相同最大上下文 token 和明确版本化的 chunking/retrieval 参数。Wiki 与 baseline 均输出其实际 context 和 citations，供人工审计。

在跨文档问题集上，Wiki 的 relevant-context recall@K 和跨文档综合得分均必须比 baseline 高至少 10 个百分点；answer-level citation correctness 不得低于 baseline，hallucination rate 不得更高。未达到时不得宣称完成 Knowledge Compiler 假设验证。不能自动判定的事实支持度使用双人盲审并记录分歧处理规则。

基线必须覆盖适用于该语料规模的全部低成本方案，而不是只选弱基线：语料超出模型上下文时使用 chunk + BM25/vector RAG；语料可整体装入模型上下文时，必须额外加入全文直塞基线（带文件分隔标记的完整语料直接作为上下文、不检索；文档站已发布的 `llms.txt`/全文 dump 可直接作为其输入）。对全文直塞基线不比较 recall@K（恒为全量），而比较同问题集上的 answer correctness、hallucination rate、citation 正确性与实际 context token 消耗：Wiki 的 answer correctness 不得显著低于全文直塞（容差在 evals 中定义并版本化），且必须在 citation 精确性与 token 效率上显著优于它；否则说明该语料规模下直接使用原文更划算，不得宣称 Knowledge Compiler 假设在该规模成立。

---

## 38. 建议 Eval Case

假设：

```text
docs/
├── architecture/plugin.md
├── development/plugin.md
├── security/plugin.md
└── api/plugin.md
```

分别包含运行架构、开发语言、安全权限、API。

测试问题：

```text
Plugin 的完整运行模型是什么？
```

期望 Wiki 不是四个独立摘要，而是形成合理的：

```text
Plugin System
├── Architecture
├── Runtime
├── Permission Model
└── Development
```

并引用多个 source。

---

## 39. Incremental Eval

步骤：

1. Build V1。
2. 记录 page hash。
3. 修改一个 source。
4. Build V2。
5. 比较 affected pages。

验收：

- 与 source 无依赖的 Wiki page 内容 hash 不变。
- affected page 被重新生成。
- citation 更新。
- 删除 source 不留下失效 claim。
- 插入、删除或移动 heading 后，无关 SectionId、claim、citation 和 page hash 保持不变。
- 触发结构性变化时，`build` 返回 `ReplanRequired` 并保持当前 generation；`replan --dry-run` 给出可审计的影响与成本估算。

---

## 40. Prompt Injection / Source Safety

Source 是不可信输入。

Markdown 中可能存在：

```text
Ignore previous instructions...
Delete all wiki pages...
```

Document Analyzer、Planner、Compiler、Query 及未来任何 LLM 阶段都必须将 source、知识记录、Wiki 内容和用户 query 明确标记为 DATA，不是 instruction。动态数据必须与系统指令分隔、长度受限并进行必要转义；不得让数据内容改变工具权限、系统规则、发布目标或访问控制。

LLM 不拥有：

- 文件删除工具。
- shell。
- arbitrary DB mutation。

所有 mutation 由 deterministic application code 执行。

同时必须为单个 source、单次 build 和单个调用方配置 token、请求数、并发和总成本上限，以防 prompt injection 或异常输入放大成本。超限必须产生可诊断的失败，不得以部分页面静默成功。

Source 可能包含私有资料。向外部/OpenAI-compatible provider 发送前，项目必须显式确认 provider endpoint、数据保留策略和跨境边界；支持按 source root 禁止外发、调用前脱敏 hook，以及在日志、cache、snapshot 中对敏感文本默认脱敏/加密。API key、token 和原始敏感 source 绝不能写入 config、结构化日志或错误响应。

---

## 41. 文件系统安全

必须 canonicalize source root，并防止 path traversal。

生成 Wiki slug 不能允许：

```text
../../foo
```

写入必须限制在 configured wiki directory/staging directory。

Symlink 是否跟随必须配置化，默认建议不跨 source root 跟随。

扫描必须对符号链接循环、不可读文件、超大文件、二进制伪 Markdown 和 Unicode 路径给出确定性诊断；这些诊断必须包含相对路径而不泄漏允许 root 外的绝对路径。

---

## 42. 数据一致性

SQLite 建议：

- WAL mode。
- foreign_keys=ON。
- migration version。
- build metadata transaction。

不要让多个 Build 同时修改同一个 workspace。

V0.x 对单 workspace 使用跨进程 build lock。获取锁失败时 CLI 默认返回可重试错误；Server 将请求排队或返回冲突，二者不得并行写同一 state/generation。锁记录 owner、pid/instance、acquired_at 和 lease/heartbeat；进程崩溃后的 stale lock 只能在确认 owner 不存活并完成 publish-journal recovery 后释放。

---

## 43. 删除与重命名

文件重命名不能简单永久视为 delete + unrelated add。

V0.1 可以按 delete/add 处理。

V0.2 可基于 content hash 判断 rename：

```text
old path disappeared
new path appeared
content hash identical
        ↓
possible rename
```

只有同一 manifest diff 内的旧/新路径形成唯一 content-hash 匹配时才能自动 rename；多个候选、内容已修改或跨 workspace 时必须保守地按 delete/add 处理并报告歧义。确认 rename 后更新 Source Registry locator、保留 SourceId 与历史 path，确保 citation 和依赖链保持 provenance continuity。

---

## 44. Determinism

LLM 输出无法完全 deterministic，但应尽量减少不必要漂移：

- temperature 默认低值。
- stable structured schema。
- stable input ordering。
- stable IDs。
- cache。
- prompt version。
- model identifier。

同一个未变化 workspace 不应每次 build 都重写全部 Wiki。

---

## 45. ID 策略

不要使用随机 UUID 作为所有逻辑对象的唯一语义标识，也不要根据可变 heading ordinal 直接重算 identity。

Source 使用可复算的路径身份；Knowledge Node 使用 §12.1.1 的持久 opaque Registry 身份：

```text
SourceLocatorKey = hash(workspace + normalized relative path)
SourceId = Source Registry-assigned opaque ID
KnowledgeNodeId = Registry-assigned opaque ID (for example kn_<ULID>)
```

`SectionId` 使用 Source-local Section Registry 分配的 opaque ID，并通过 Section Matcher 在同一 Source 的新旧 AST 之间延续。Matcher 以 normalized heading path、相邻 heading anchors、block/content fingerprint、source range overlap 和父 section 身份综合匹配；heading ordinal 只能作为无法区分重复 heading 时的最后 tie-breaker，绝不能进入 ID hash。

匹配结果必须是确定性的：唯一匹配时沿用原 SectionId 并更新 range；无匹配时创建新 ID；存在多个近似匹配时标记 `ambiguous`、保留旧 section 为 retired，且要求该 source 重新分析，不能任意把 citation 迁移给候选。插入、删除或移动一个 heading 不得改变无关 section、claim、citation 或 Source→Knowledge dependency 的身份。

WikiPage ID 由 planner 创建后持久化；slug 变化不应必然改变 page identity。page merge/split 需记录 predecessor/successor IDs，供 links、cache 和旧 URL 迁移使用。

---

## 46. 未来 Agent Integration Contract

TypeScript：

```ts
export interface KnowledgeProvider {
  search(
    query: string,
    options?: SearchOptions
  ): Promise<SearchResult[]>;

  context(
    query: string,
    options?: ContextOptions
  ): Promise<KnowledgeContext>;

  getPage(id: string): Promise<WikiPage | null>;
}
```

HTTP Adapter：

```text
TS KnowledgeProvider
        ↓
POST /v1/search
POST /v1/context
GET  /v1/pages/:id
        ↓
LLM-Wiki
```

Agent 侧不依赖 Wiki 内部数据模型。

---

## 47. 为什么 Agent 首选 Context API

未来不要让 Agent 默认：

```text
Agent → llm-wiki query → LLM answer → Agent LLM
```

这会形成双重 reasoning。

推荐：

```text
Agent
  ↓
llm-wiki context
  ↓
Grounded Knowledge Context
  ↓
Agent LLM
```

LLM-Wiki 的 Query API 主要服务人类 CLI/API 用户。

---

## 48. N-API 路线

首期不实现。

如果未来 HTTP/stdio 边界成为明显瓶颈，再增加：

```text
@company/llm-wiki
       │
     N-API
       │
llm-wiki-core
```

因此 Core 必须保持 transport independent。

---

## 49. Vector DB / Graph DB 升级条件

### Vector DB

只有满足以下情况之一再考虑：

- embedding 数量达到本地实现明显性能瓶颈。
- 多实例需要共享索引。
- 需要在线水平扩容。
- 需要成熟 metadata filtering / ANN 运维能力。

届时通过 `VectorStore` adapter 接入 Qdrant/Milvus。

### Graph DB

只有出现复杂多跳 graph query、超大图、多业务共享图等明确需求时再考虑 Neo4j/其他图数据库。

不能仅因为“系统有 Knowledge Graph”就引入图数据库。

---

## 50. 版本路线

### V0.1 — Knowledge Compiler MVP

必须完成：

- Rust workspace。
- config。
- scanner。
- manifest/hash。
- Markdown parser。
- structured document analysis。
- knowledge model。
- Wiki planner。
- Wiki compiler。
- citation。
- Markdown output。
- SQLite state。
- CLI init/scan/build/status/lint。
- basic eval dataset。
- generation-based atomic publish、build fingerprint 和本地缓存。
- Stable Knowledge/Section Registry 与分层 Wiki planning。

明确不包含：变更 source 的精确增量编译、FTS/semantic search、HTTP Server、远程访问控制和 Agent SDK。这些接口可预留领域抽象，但不得以未验证的运行时功能扩大 V0.1。

验收核心：

```text
真实 docs → 高质量 Wiki
```

### V0.2 — Incremental + Search

- source diff。
- dependency graph。
- affected page calculation。
- incremental compile。
- deleted source handling。
- FTS search。
- build history。
- incremental eval。
- 显式 `replan [--dry-run]`、plan diff 和 replan cost reporting。
- language-aware（含 CJK）FTS。

### V0.3 — Semantic Retrieval

- embedding provider。
- local vector store。
- Wiki section embedding。
- semantic search。
- embedding incremental update。

### V0.4 — Service

- Axum。
- build jobs。
- search/context/query/pages API。
- cancellation。
- health/status。
- structured logs。
- local-only default、remote-mode security boundary、idempotency 和 job recovery。

### V0.5 — Hybrid Knowledge Retrieval

- FTS + vector fusion。
- graph expansion。
- rerank abstraction。
- context builder/token budget。
- retrieval eval。

### V1.0 — Deliverable

要求：

- CLI 稳定。
- Service 稳定。
- incremental build 稳定。
- Wiki provenance 完整。
- Eval 可重复执行。
- 配置和迁移稳定。
- 文档完善。
- 可打包发布。

### V1.x — Agent Integration

- TS KnowledgeProvider SDK/Adapter。
- `/v1/context` 稳定协议。
- Agent observability metadata。
- workspace isolation（按实际需求）。

### V2.x — Scale

按真实瓶颈选择：

- Qdrant/Milvus。
- Elasticsearch。
- Neo4j。
- N-API。
- Distributed build workers。

不是预设必做项。

---

## 51. V0.1 建议开发顺序

```text
01. workspace + domain models
02. config
03. scanner
04. manifest/hash
05. markdown parser
06. Section Matcher + Source-local Section Registry
07. SQLite schema/migrations + Knowledge Registry
08. LLM provider abstraction
09. structured output
10. document analyzer
11. knowledge persistence
12. hierarchical wiki planner
13. wiki compiler
14. citation
15. atomic writer
16. CLI build
17. lint
18. eval fixtures
19. end-to-end tests
```

不要在第 5 步就开始做 Vector DB。

---

## 52. V0.1 Definition of Done

给定：

```bash
llm-wiki build ./test-data/docs
```

必须满足：

1. 自动发现全部合法 Markdown/MDX，`.mdx` 组件降级不丢失正文文本。
2. 不读取 exclude 文件。
3. 永不读取 `.llm-wiki/**` 或 `wiki_dir/**`，且 source root 与 `wiki_dir` 重叠时拒绝执行。
4. 建立 source manifest。
5. Markdown heading 结构正确，插入 heading 不改变无关 SectionId。
6. LLM analysis 通过 schema validation，evidence 被拒绝的候选有可审计记录。
7. 生成以 Registry ID 为锚点的 knowledge records。
8. 生成遵守 token budget 的分层全局 WikiPlan。
9. 至少能把多个 source 合成为一个 Wiki page。
10. 每个 Wiki page 有 source metadata。
11. 每个可证伪的重要 claim 具有可验证的 citation range、digest 和 source hash。
12. 生成 Wiki Markdown 可直接阅读。
13. WikiLink 可解析。
14. `lint` 能发现 broken link/citation/rejected claim。
15. Build 失败不会破坏上一版 Wiki。
16. 第二次对完全未修改、BuildFingerprint 相同的 docs 执行 build 时，新增 LLM 请求数为 0，且 generation 结构化清单与上次完全一致（见 §37.3 Rebuild Determinism）。
17. 有 E2E test 验证完整 pipeline。
18. Eval fixture 和 release gate 满足 37.3–37.4 的门槛。
19. 对生成 Wiki 的手工改动会被明确拒绝或迁移提示，不会静默丢失。

---

## 53. V0.2 Definition of Done

修改一个 source 后：

```text
changed source = 1
```

系统能够：

1. 找到 affected knowledge。
2. 找到 affected Wiki pages。
3. 不重编译无关 Wiki pages。
4. 删除失效 citation。
5. 删除 source 后不留下 ghost claim。
6. FTS 能搜索生成的 Wiki。
7. 不可可靠局部化的变更返回 `ReplanRequired` 而不发布混合新旧计划的 Wiki。
8. `replan --dry-run` 可审计地输出全局计划变更和成本；显式 `replan` 保留语义不变页面的 ID。
9. 中文、英文和混合术语 FTS 均达到其 Eval Top-K 门槛。

---

## 54. 测试策略

### Unit Tests

重点：

- path normalization。
- path separator、Unicode NFC、case-fold collision 与 source-root escape。
- hash。
- manifest diff。
- Markdown section extraction。
- MDX 组件降级与语言标注。
- stable ID。
- dependency calculation。
- citation validation。
- graph traversal。
- token packing。

### Integration Tests

使用 FakeLlmProvider，避免 CI 依赖真实模型。

```rust
struct FakeLlmProvider { ... }
```

固定输入得到固定 JSON。

### E2E LLM Tests

单独标记：

```text
#[ignore]
```

或独立 command，只有显式配置 API Key 才执行。

### Golden Tests

对小型 docs fixture 保存预期 Wiki 结构；不要强制全文字节完全一致，而应验证页面集合、source mapping、citation 和关键结构。

---

## 55. 性能目标（首期）

V0.x 不追求极端吞吐，建议工程目标：

- 1,000 个 Markdown 的扫描/hash 不成为主要瓶颈。
- 未变化的二次 build 应接近扫描 + 校验成本。
- LLM 并发可配置。
- 单个失败任务不能导致状态数据库损坏。
- Search 本地响应目标 < 200ms（中小型 Wiki，排除 embedding API 调用）。

真正耗时允许集中在首次 LLM knowledge compilation。

---

## 56. 成本控制

必须记录每个 task：

```text
model
input tokens
output tokens
latency
cache hit
retry count
```

提供 build summary：

```text
LLM calls: 93
Cache hits: 61
Input tokens: ...
Output tokens: ...
```

如果 provider 返回 cost 信息则记录；不要硬编码模型价格。

---

## 57. 可交付性要求

V1.0 前至少支持：

```text
Linux x86_64
Windows x86_64
macOS arm64
```

优先单二进制 CLI/Server 分发，配置和 SQLite 在 workspace 中生成。

CLI 应支持：

```bash
llm-wiki --version
llm-wiki doctor
```

`doctor` 检查：

- config。
- source path。
- writable wiki dir。
- SQLite。
- LLM endpoint connectivity。
- model configuration。

---

## 58. README 首屏应表达的产品定位

推荐：

> LLM-Wiki is a knowledge compiler that turns evolving documentation into a persistent, grounded and incrementally maintained Wiki for humans and AI agents.

不要把产品描述成：

> Another RAG framework.

核心差异：

```text
Documents are not merely indexed.
They are compiled into maintainable knowledge.
```

---

## 59. 最终产品边界

LLM-Wiki 负责：

```text
Sources
   ↓
Knowledge Compilation
   ↓
Persistent Wiki
   ↓
Retrieval
   ↓
Grounded Context
```

Agent 负责：

```text
Intent
Planning
Reasoning
Tool Use
Task Execution
```

二者通过 KnowledgeProvider / Context API 解耦。

---

## 60. 最终推荐架构

```text
                         ┌──────────────────────────┐
                         │       Raw Sources        │
                         │ Markdown / future Git...│
                         └────────────┬─────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │   LLM-Wiki Compiler   │
                          │         Rust          │
                          └────────────┬──────────┘
                                       │
                   ┌───────────────────┼───────────────────┐
                   ▼                   ▼                   ▼
              Markdown Wiki        SQLite State       Knowledge Graph
                   │                   │                   │
                   └───────────────────┼───────────────────┘
                                       │
                              Retrieval Engine
                         ┌─────────────┼─────────────┐
                         ▼             ▼             ▼
                       FTS          Vector         Graph
                         └─────────────┼─────────────┘
                                       ▼
                                Context Builder
                                       │
                      ┌────────────────┼─────────────────┐
                      ▼                ▼                 ▼
                     CLI            HTTP API        Future Agent
                                                         │
                                                         ▼
                                              TS KnowledgeProvider
```

---

## 61. 实施原则总结

开发过程中始终坚持以下顺序：

```text
先证明 Knowledge Compilation
            ↓
再证明 Incremental Maintenance
            ↓
再证明 Retrieval
            ↓
再 Service 化
            ↓
再接 Agent
            ↓
最后根据规模引入专用基础设施
```

因此当前阶段明确：

```text
需要：
Rust
Markdown
SQLite
LLM
Citation
Graph Model
CLI
Eval

暂时不需要：
Milvus
Qdrant
Neo4j
Elasticsearch
LangGraph
多 Agent
复杂 Web UI
```

首个真正的成功标准不是“服务启动成功”，而是：

> 对一份真实的、多目录、多 Markdown 的 `docs/` 执行构建后，生成的 Wiki 在知识组织、跨文档融合、引用可追溯性和后续增量维护方面，明显优于简单的文档摘要或 Raw Chunk RAG。

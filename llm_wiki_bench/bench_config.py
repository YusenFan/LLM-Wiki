"""Benchmark configuration for LLM-Wiki.

Manages dataset paths, LLM model selection, and wiki directory structure
for HotpotQA / MuSiQue / 2WikiMultiHopQA. All credentials are read from
environment variables — no hard-coded keys or internal endpoints.

Environment variables:
    OPENAI_API_KEY      bearer token
    OPENAI_BASE_URL     API base URL (default: https://api.openai.com/v1)
    LLM_PREMIUM_MODEL   strong model used for synthesis steps
    LLM_FAST_MODEL      fast model used for analysis steps
    LLM_MAX_TOKENS      max tokens per LLM call (default: 16384)
    LLM_TEMPERATURE     sampling temperature (default: 0.2)
"""

import os
import re
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCH_DIR = Path(__file__).parent
BASE_DIR = BENCH_DIR.parent          # release/
CONFIGS_DIR = BASE_DIR / "configs"
SCHEMA_FILE = CONFIGS_DIR / "wiki-schema.md"

# ---------------------------------------------------------------------------
# LLM model configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------

LLM_PREMIUM_MODEL = os.environ.get("LLM_PREMIUM_MODEL", "gpt-4o")
LLM_FAST_MODEL    = os.environ.get("LLM_FAST_MODEL",    "gpt-4o-mini")
LLM_QUERY_MODEL   = os.environ.get("LLM_QUERY_MODEL",   LLM_FAST_MODEL)
LLM_LINT_MODEL    = os.environ.get("LLM_LINT_MODEL",    LLM_FAST_MODEL)
LLM_STEP1_MODEL   = LLM_FAST_MODEL
LLM_STEP2_MODEL   = LLM_PREMIUM_MODEL
LLM_MODEL         = LLM_FAST_MODEL

LLM_MAX_TOKENS        = int(os.environ.get("LLM_MAX_TOKENS",   "16384"))
LLM_TEMPERATURE       = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
LLM_STEP1_TEMPERATURE = LLM_TEMPERATURE
LLM_STEP2_TEMPERATURE = LLM_TEMPERATURE

_API_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_API_BASE       = _API_BASE
LLM_PREMIUM_API_BASE = _API_BASE
LLM_FAST_API_BASE    = _API_BASE
LLM_STEP1_API_BASE   = _API_BASE
LLM_STEP2_API_BASE   = _API_BASE
LLM_QUERY_API_BASE   = _API_BASE
LLM_LINT_API_BASE    = _API_BASE

LLM_HEADERS: dict = {}


# 【兼容配置】从环境组装 Bearer 请求头；当前 llm_client 使用自己的 _headers，此接口供旧调用兼容。
def get_llm_headers() -> dict:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else "",
    }


# ---------------------------------------------------------------------------
# Ingestion parameters
# ---------------------------------------------------------------------------

INGEST_TOOL_BUDGET        = int(os.environ.get("INGEST_TOOL_BUDGET", "40"))
SUMMARY_BUILD_LIMIT       = int(os.environ.get("SUMMARY_BUILD_LIMIT", "20"))
INGEST_TOKEN_BUDGET       = int(os.environ.get("INGEST_TOKEN_BUDGET", "250000"))  # per-document abort threshold
INGEST_BATCH_SIZE         = 10
INGEST_MAX_CONTENT_LEN    = 15000
INGEST_MIN_CONTENT_LEN    = 50
INGEST_CONSOLIDATE_EVERY  = 9999   # periodic maintenance disabled in bench mode
INGEST_OVERVIEW_EVERY     = 9999
INGEST_PERIODIC_EVERY     = 9999
INGEST_CONTRADICTION_EVERY = 9999

# ---------------------------------------------------------------------------
# Wiki directory structure
# ---------------------------------------------------------------------------

# Fixed infrastructure directories present in every wiki.
FIXED_DIRS = {
    "sources":          {"description": "Source pages — paragraph digests and original archives"},
    "sources/digests":  {"description": "Digest pages — structured summaries of source paragraphs"},
    "sources/articles": {"description": "Article archive — original source paragraph texts"},
}

# Default page types used when LLM auto-init fails.
DEFAULT_PAGE_TYPES = {
    "entities":  {"description": "Entity pages — people, organizations, places, works, objects",  "auto_created": True},
    "events":    {"description": "Event pages — historical events, incidents, ceremonies",          "auto_created": True},
    "concepts":  {"description": "Concept pages — theories, methods, genres, abstract ideas",      "auto_created": True},
    "relations": {"description": "Relation pages — comparisons and links between entities",         "auto_created": True},
}

# Compatibility stubs (not used in benchmark mode).
USER_MAP: dict       = {}
ALIAS_MAP: dict      = {}
USER_PAGE_TYPES: dict = {}

# Link-graph retrieval weights (used by query layer if enabled).
QUERY_TOP_K          = 5
WEIGHT_DIRECT_LINK   = 3.0
WEIGHT_SOURCE_OVERLAP = 4.0
WEIGHT_ADAMIC_ADAR   = 1.5
WEIGHT_TYPE_AFFINITY = 1.0
TYPE_BONUS           = {"source": 0.1, "synthesis": 0.1, "_default": 0.3}

# Mirror directory (disabled in benchmark mode; set WIKI_MIRROR_DIR to enable).
WFS_WIKI_DIR = None

# ---------------------------------------------------------------------------
# Active dataset paths (populated by set_dataset)
# ---------------------------------------------------------------------------

_current_dataset: str | None = None
WIKI_DIR:      Path | None = None
CACHE_FILE:    Path | None = None
RAW_DIR:       Path | None = None
WIKI_INDEX:    Path | None = None
WIKI_OVERVIEW: Path | None = None
WIKI_LOG:      Path | None = None
INGEST_LOG_DIR: Path | None = None


# 【共享配置】设置当前数据集和 wiki/raw/cache/log 路径并创建 wiki 根目录；这是模块全局状态，同一模块实例中后调用会覆盖前配置。
def set_dataset(dataset_name: str) -> None:
    """Activate a dataset: compute and create all wiki / raw / cache paths."""
    global _current_dataset, WIKI_DIR, CACHE_FILE, RAW_DIR
    global WIKI_INDEX, WIKI_OVERVIEW, WIKI_LOG, INGEST_LOG_DIR

    _current_dataset = dataset_name
    WIKI_DIR   = BASE_DIR / "wiki_output" / dataset_name / "wiki"
    RAW_DIR    = BASE_DIR / "raw"         / dataset_name / "articles"
    CACHE_FILE = BASE_DIR / f".wiki-cache-bench-{dataset_name}.json"

    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    WIKI_INDEX    = WIKI_DIR / "index.md"
    WIKI_OVERVIEW = WIKI_DIR / "overview.md"
    WIKI_LOG      = WIKI_DIR / "log.md"
    INGEST_LOG_DIR = WIKI_DIR.parent / "logs"


# 【配置读取】返回当前数据集名称；尚未 set_dataset 时为 None。
def get_dataset() -> str | None:
    return _current_dataset


# Aliases expected by the ingestion engine.
# 【兼容别名】把旧 user_key 当数据集名称交给 set_dataset，不实现用户隔离或鉴权。
def set_user(user_key: str) -> None:
    set_dataset(user_key)

# 【兼容别名】返回当前数据集名称，不读取用户账户。
def get_user() -> str | None:
    return _current_dataset

# 【兼容别名】返回当前数据集名称，供旧接口使用。
def get_current_user() -> str | None:
    return _current_dataset

# 【兼容空操作】benchmark 模式不启用镜像；函数体只 pass。
def enable_wfs_mirror() -> None:
    """No-op in benchmark mode."""
    pass

# 【兼容空操作】原样返回 name；名称不在这里做消歧或归一化。
def normalize_entity_name(name: str) -> str:
    return name

# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

_FM_SPLIT_RE = re.compile(r'^---\s*$', re.MULTILINE)


# 【旧文本拆分】用独占行的 --- 切开元数据与正文，返回三元组或 None；不执行 YAML 类型校验，当前存储使用 wiki_store.frontmatter。
def split_frontmatter(text: str) -> tuple[str, str, str] | None:
    """Split YAML frontmatter and body.

    Returns (before, fm_text, body), or None if no valid frontmatter block.
    Uses line-anchored ``---`` so dashes inside values are not misinterpreted.
    """
    if not text.startswith("---"):
        return None
    matches = list(_FM_SPLIT_RE.finditer(text))
    if len(matches) < 2:
        return None
    start = matches[0].end()
    end   = matches[1].start()
    return (text[:matches[0].start()], text[start:end], text[matches[1].end():])

# ---------------------------------------------------------------------------
# Page-type management
# ---------------------------------------------------------------------------

# 【目录配置读取】优先 wiki 内 page_types.yaml，其次 configs 模板，再次内置默认；空字典也有效，不强制默认分类。
def get_page_types() -> dict:
    """Return page types: prefer wiki-local YAML, then configs template, then defaults."""
    if WIKI_DIR is not None:
        wiki_yaml = WIKI_DIR / "page_types.yaml"
        if wiki_yaml.exists():
            try:
                data = yaml.safe_load(wiki_yaml.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("page_types"), dict):
                    return data["page_types"]
            except (yaml.YAMLError, OSError):
                pass
    template = CONFIGS_DIR / "page_types.yaml"
    if template.exists():
        try:
            data = yaml.safe_load(template.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("page_types"), dict):
                return data["page_types"]
        except (yaml.YAMLError, OSError):
            pass
    return dict(DEFAULT_PAGE_TYPES)


# 【目录配置写入】把给定字典存到 wiki/page_types.yaml；未配置 wiki 时直接返回。
def save_page_types(page_types: dict) -> None:
    if WIKI_DIR is None:
        return
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    content = yaml.dump(
        {"page_types": page_types},
        allow_unicode=True, default_flow_style=False, sort_keys=False,
    )
    (WIKI_DIR / "page_types.yaml").write_text(content, encoding="utf-8")


# 【旧目录注册】类型尚不存在时保存描述、建目录并打印；当前事实写入可直接按内容路径建目录，不依赖此注册。
def register_page_type(name: str, description: str, auto_created: bool = True) -> None:
    page_types = get_page_types()
    if name not in page_types:
        page_types[name] = {"description": description, "auto_created": auto_created}
        save_page_types(page_types)
        if WIKI_DIR is not None:
            (WIKI_DIR / name).mkdir(parents=True, exist_ok=True)
        print(f"  Registered new page type: {name} — {description}")


# 【旧目录配置】对 split/move_page 建立目标目录和类型条目；本函数本身不搬页面，实际迁移在旧 ingest 辅助函数中。
def apply_dir_changes(changes: list[dict]) -> None:
    page_types = get_page_types()
    for change in changes:
        action = change.get("action", "")
        to_dir = change.get("to", "")
        desc   = change.get("description", "")
        if action in ("split", "move_page") and to_dir and to_dir not in page_types:
            page_types[to_dir] = {"description": desc or to_dir, "auto_created": True}
            save_page_types(page_types)
            if WIKI_DIR is not None:
                (WIKI_DIR / to_dir).mkdir(parents=True, exist_ok=True)
            print(f"  Created directory: {to_dir}")


# 【旧目录映射】合并已注册知识类型与固定来源目录，返回名称到 Path；不是实时全文件系统树。
def get_page_dirs() -> dict[str, Path]:
    if WIKI_DIR is None:
        return {}
    dirs: dict[str, Path] = {}
    for name in get_page_types():
        dirs[name] = WIKI_DIR / name
    for name in FIXED_DIRS:
        dirs[name] = WIKI_DIR / name
    return dirs


# 【旧目录描述】汇总注册类型与固定来源目录的描述和路径，供旧提示词／索引使用。
def get_all_dir_info() -> dict[str, dict]:
    result: dict[str, dict] = {}
    for name, info in get_page_types().items():
        result[name] = {"description": info.get("description", ""),
                        "path": str(WIKI_DIR / name) if WIKI_DIR else ""}
    for name, info in FIXED_DIRS.items():
        result[name] = {"description": info.get("description", ""),
                        "path": str(WIKI_DIR / name) if WIKI_DIR else ""}
    return result


# 【旧提示词材料】把配置目录、页面数量和描述拼成 Markdown 清单；不同于当前检索器的实时 tree。
def get_dir_catalog_text() -> str:
    lines = []
    for name, info in get_all_dir_info().items():
        if name.startswith("sources/"):
            continue
        desc = info["description"]
        if WIKI_DIR is not None:
            dir_path = WIKI_DIR / name
            if name == "sources":
                count = sum(
                    len([f for f in (WIKI_DIR / "sources" / sub).glob("*.md")
                         if f.name != "_index.md"])
                    for sub in ("digests", "articles")
                    if (WIKI_DIR / "sources" / sub).exists()
                )
            else:
                count = len([f for f in dir_path.glob("*.md") if f.name != "_index.md"]) \
                        if dir_path.exists() else 0
        else:
            count = 0
        lines.append(f"- **{name}/** ({count} pages) — {desc}")
    return "\n".join(lines)


# 【当前初始化】创建固定来源目录与 versions，并在缺失时写空类型注册表；不抽样、不调用 LLM 生成 purpose 或分类。
def ensure_wiki_dirs() -> None:
    """Create wiki directories; auto-init page types on first run."""
    if WIKI_DIR is None:
        return
    WIKI_DIR.mkdir(parents=True, exist_ok=True)

    # Start from actual content, never infer a mandatory taxonomy from samples.
    for name in FIXED_DIRS:
        (WIKI_DIR / name).mkdir(parents=True, exist_ok=True)
    (WIKI_DIR / "sources" / "versions").mkdir(parents=True, exist_ok=True)
    if not (WIKI_DIR / "page_types.yaml").exists():
        save_page_types({})

# ---------------------------------------------------------------------------
# Purpose file management
# ---------------------------------------------------------------------------

# 【当前构建目标】优先根目录 purpose_<dataset>.md，否则返回 configs/purpose_bench.md；只选择路径，不生成内容。
def get_purpose_file() -> Path:
    """Return the dataset-specific purpose file, falling back to a template."""
    if _current_dataset:
        candidate = BASE_DIR / f"purpose_{_current_dataset}.md"
        if candidate.exists():
            return candidate
    return CONFIGS_DIR / "purpose_bench.md"

# ---------------------------------------------------------------------------
# LLM-based first-run auto-initialisation
# ---------------------------------------------------------------------------

# 【旧初始化辅助】读取至多 max_len 字符，缺失或读失败返回提示字符串。
def _read_file_safe(path: Path | None, max_len: int = 5000) -> str:
    if path is not None and path.exists():
        try:
            text = path.read_text(encoding="utf-8")
            return text[:max_len] if len(text) > max_len else text
        except (OSError, UnicodeDecodeError):
            return "(read failed)"
    return "(empty)"


# 【旧初始化辅助】从排序文章中均匀选样，抽取标题和前 1200 字符；当前 ensure_wiki_dirs 不调用它。
def _sample_articles_for_init(n: int = 200) -> list[dict]:
    """Uniformly sample up to n articles from RAW_DIR."""
    if RAW_DIR is None or not RAW_DIR.exists():
        return []
    all_files = sorted(RAW_DIR.glob("*.md"))
    if not all_files:
        return []
    if len(all_files) <= n:
        selected = all_files
    else:
        step = len(all_files) / n
        indices = [int(i * step) for i in range(n)]
        if indices[-1] != len(all_files) - 1:
            indices[-1] = len(all_files) - 1
        selected = [all_files[i] for i in indices]
    samples = []
    for f in selected:
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        title = f.stem
        body  = text
        result = split_frontmatter(text)
        if result is not None:
            _, fm, body = result
            m = re.search(r'title:\s*["\']?(.+?)["\']?\s*$', fm, re.MULTILINE)
            if m:
                title = m.group(1).strip()
        excerpt = body.strip()[:1200]
        if excerpt:
            samples.append({"title": title, "excerpt": excerpt})
    return samples


# 【旧显式初始化／LLM】已有 purpose 就复用，否则取文章样本让模型生成目标文件，样本不足或失败用模板；当前默认构建不自动执行。
def auto_init_purpose() -> Path | None:
    """Ask the LLM to summarise the corpus and write a purpose_<dataset>.md."""
    from llm_client import call_llm

    if not _current_dataset:
        return None
    purpose_path = BASE_DIR / f"purpose_{_current_dataset}.md"
    if purpose_path.exists():
        return purpose_path

    fallback = (
        "# Wiki Research Direction\n\n"
        "## Core Positioning\n"
        f"Structured knowledge base compiled from Wikipedia paragraphs in the "
        f"{_current_dataset} dataset, used as the retrieval backend for "
        "multi-hop question answering.\n\n"
        "## Ingestion Focus\n"
        "- Priority: key facts, entity relationships, temporal information, causal connections.\n"
        "- Moderate: background context, categorical information.\n"
        "- Skip: redundant information already captured in other pages.\n\n"
        "## Target Use Case\n"
        "Multi-hop question answering — questions that require combining "
        "information from multiple paragraphs or entities.\n"
    )

    samples = _sample_articles_for_init(n=200)
    if not samples or len(samples) < 3:
        purpose_path.write_text(fallback, encoding="utf-8")
        return purpose_path

    sample_text = "\n".join(
        f"### Article {i}: {s['title']}\n{s['excerpt']}\n"
        for i, s in enumerate(samples, 1)
    )
    try:
        text = call_llm(
            system_prompt=(
                "You are a knowledge base architect. Analyse the article samples "
                "below from a Wikipedia-based multi-hop QA dataset and produce a "
                "short purpose file describing the knowledge base."
            ),
            user_prompt=f"## Dataset: {_current_dataset}\n\n## Articles\n{sample_text}\n",
            model=LLM_PREMIUM_MODEL,
            temperature=0.3,
            max_tokens=2048,
        )
    except Exception:
        text = ""

    purpose_path.write_text(
        text.strip() if text and text.strip().startswith("#") else fallback,
        encoding="utf-8",
    )
    return purpose_path


# 【兼容初始化】仅在 wiki 类型配置缺失时写空字典；不调用模型，也不分析文章主题。
def auto_init_page_types() -> None:
    """Compatibility entry point: initialize an empty, content-driven registry."""
    if WIKI_DIR is not None and not (WIKI_DIR / "page_types.yaml").exists():
        save_page_types({})

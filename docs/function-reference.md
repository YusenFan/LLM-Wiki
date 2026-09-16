# 函数逐项索引

配合 [系统学习指南](system-learning-guide.md) 阅读。这里覆盖 `llm_wiki_bench/*.py` 中每个显式函数／方法，包括嵌套辅助函数；数据类自动生成的方法不计入。每项说明与源码中的中文注释一致。

共 **213 个函数／方法**。
当前入口与旧流程已在说明中区分；旧函数保留并不表示默认会运行。源码链接使用注释添加后的行号，后续改动代码时行号可能移动。

## __init__.py

无显式函数。`__init__.py` 调整导入路径并声明模块导出列表。

## bench_config.py

### `get_llm_headers`

[查看源码](../llm_wiki_bench/bench_config.py#L60) · `6` 行函数体（含已有文档）

【兼容配置】从环境组装 Bearer 请求头；当前 llm_client 使用自己的 _headers，此接口供旧调用兼容。

### `set_dataset`

[查看源码](../llm_wiki_bench/bench_config.py#L133) · `15` 行函数体（含已有文档）

【共享配置】设置当前数据集和 wiki/raw/cache/log 路径并创建 wiki 根目录；这是模块全局状态，同一模块实例中后调用会覆盖前配置。

### `get_dataset`

[查看源码](../llm_wiki_bench/bench_config.py#L151) · `2` 行函数体（含已有文档）

【配置读取】返回当前数据集名称；尚未 set_dataset 时为 None。

### `set_user`

[查看源码](../llm_wiki_bench/bench_config.py#L157) · `2` 行函数体（含已有文档）

【兼容别名】把旧 user_key 当数据集名称交给 set_dataset，不实现用户隔离或鉴权。

### `get_user`

[查看源码](../llm_wiki_bench/bench_config.py#L161) · `2` 行函数体（含已有文档）

【兼容别名】返回当前数据集名称，不读取用户账户。

### `get_current_user`

[查看源码](../llm_wiki_bench/bench_config.py#L165) · `2` 行函数体（含已有文档）

【兼容别名】返回当前数据集名称，供旧接口使用。

### `enable_wfs_mirror`

[查看源码](../llm_wiki_bench/bench_config.py#L169) · `3` 行函数体（含已有文档）

【兼容空操作】benchmark 模式不启用镜像；函数体只 pass。

### `normalize_entity_name`

[查看源码](../llm_wiki_bench/bench_config.py#L174) · `2` 行函数体（含已有文档）

【兼容空操作】原样返回 name；名称不在这里做消歧或归一化。

### `split_frontmatter`

[查看源码](../llm_wiki_bench/bench_config.py#L185) · `14` 行函数体（含已有文档）

【旧文本拆分】用独占行的 --- 切开元数据与正文，返回三元组或 None；不执行 YAML 类型校验，当前存储使用 wiki_store.frontmatter。

### `get_page_types`

[查看源码](../llm_wiki_bench/bench_config.py#L205) · `20` 行函数体（含已有文档）

【目录配置读取】优先 wiki 内 page_types.yaml，其次 configs 模板，再次内置默认；空字典也有效，不强制默认分类。

### `save_page_types`

[查看源码](../llm_wiki_bench/bench_config.py#L228) · `9` 行函数体（含已有文档）

【目录配置写入】把给定字典存到 wiki/page_types.yaml；未配置 wiki 时直接返回。

### `register_page_type`

[查看源码](../llm_wiki_bench/bench_config.py#L240) · `8` 行函数体（含已有文档）

【旧目录注册】类型尚不存在时保存描述、建目录并打印；当前事实写入可直接按内容路径建目录，不依赖此注册。

### `apply_dir_changes`

[查看源码](../llm_wiki_bench/bench_config.py#L251) · `12` 行函数体（含已有文档）

【旧目录配置】对 split/move_page 建立目标目录和类型条目；本函数本身不搬页面，实际迁移在旧 ingest 辅助函数中。

### `get_page_dirs`

[查看源码](../llm_wiki_bench/bench_config.py#L266) · `9` 行函数体（含已有文档）

【旧目录映射】合并已注册知识类型与固定来源目录，返回名称到 Path；不是实时全文件系统树。

### `get_all_dir_info`

[查看源码](../llm_wiki_bench/bench_config.py#L278) · `9` 行函数体（含已有文档）

【旧目录描述】汇总注册类型与固定来源目录的描述和路径，供旧提示词／索引使用。

### `get_dir_catalog_text`

[查看源码](../llm_wiki_bench/bench_config.py#L290) · `22` 行函数体（含已有文档）

【旧提示词材料】把配置目录、页面数量和描述拼成 Markdown 清单；不同于当前检索器的实时 tree。

### `ensure_wiki_dirs`

[查看源码](../llm_wiki_bench/bench_config.py#L315) · `12` 行函数体（含已有文档）

【当前初始化】创建固定来源目录与 versions，并在缺失时写空类型注册表；不抽样、不调用 LLM 生成 purpose 或分类。

### `get_purpose_file`

[查看源码](../llm_wiki_bench/bench_config.py#L333) · `7` 行函数体（含已有文档）

【当前构建目标】优先根目录 purpose_<dataset>.md，否则返回 configs/purpose_bench.md；只选择路径，不生成内容。

### `_read_file_safe`

[查看源码](../llm_wiki_bench/bench_config.py#L346) · `8` 行函数体（含已有文档）

【旧初始化辅助】读取至多 max_len 字符，缺失或读失败返回提示字符串。

### `_sample_articles_for_init`

[查看源码](../llm_wiki_bench/bench_config.py#L357) · `33` 行函数体（含已有文档）

【旧初始化辅助】从排序文章中均匀选样，抽取标题和前 1200 字符；当前 ensure_wiki_dirs 不调用它。

### `auto_init_purpose`

[查看源码](../llm_wiki_bench/bench_config.py#L393) · `54` 行函数体（含已有文档）

【旧显式初始化／LLM】已有 purpose 就复用，否则取文章样本让模型生成目标文件，样本不足或失败用模板；当前默认构建不自动执行。

### `auto_init_page_types`

[查看源码](../llm_wiki_bench/bench_config.py#L450) · `4` 行函数体（含已有文档）

【兼容初始化】仅在 wiki 类型配置缺失时写空字典；不调用模型，也不分析文章主题。

## bench_error_book.py

### `_get_error_book_path`

[查看源码](../llm_wiki_bench/bench_error_book.py#L34) · `6` 行函数体（含已有文档）

【旧错误簿】返回当前 wiki 的错误簿 YAML 路径；未配置时退回工作目录的 error_book.yaml。

### `load_error_book`

[查看源码](../llm_wiki_bench/bench_error_book.py#L43) · `12` 行函数体（含已有文档）

【旧错误簿读取】加载 YAML 错误记录，缺失／异常时按代码兜底返回；当前 BuildAgent 不注入这份错误簿。

### `save_error_book`

[查看源码](../llm_wiki_bench/bench_error_book.py#L58) · `11` 行函数体（含已有文档）

【旧错误簿写入】把错误条目序列化为 YAML 保存，供旧维护循环持久记录。

### `_get_samples`

[查看源码](../llm_wiki_bench/bench_error_book.py#L74) · `7` 行函数体（含已有文档）

【旧 schema 兼容】优先当前 active_samples，否则读取旧 samples 列表。

### `_get_count`

[查看源码](../llm_wiki_bench/bench_error_book.py#L84) · `8` 行函数体（含已有文档）

【旧 schema 兼容】读取 still_active 或旧 count，统一提供问题数量。

### `_set_samples`

[查看源码](../llm_wiki_bench/bench_error_book.py#L95) · `6` 行函数体（含已有文档）

【旧 schema 兼容】依据条目已有字段写入样本列表，原地修改字典。

### `_set_count`

[查看源码](../llm_wiki_bench/bench_error_book.py#L104) · `6` 行函数体（含已有文档）

【旧 schema 兼容】依据条目已有字段写入问题数量，原地修改字典。

### `get_active_constraints`

[查看源码](../llm_wiki_bench/bench_error_book.py#L113) · `37` 行函数体（含已有文档）

【旧提示词材料】把开放错误记录格式化为约束文字，用于旧生成／修复提示词；函数自身不调用 LLM。

### `record_lint_issues`

[查看源码](../llm_wiki_bench/bench_error_book.py#L212) · `139` 行函数体（含已有文档）

【旧错误聚合】按 category 合并 lint 结果、时间及样本，并附批次文章上下文后保存错误簿。

### `record_sample_with_context`

[查看源码](../llm_wiki_bench/bench_error_book.py#L354) · `65` 行函数体（含已有文档）

【旧错误记录】记录单条样本及其修复上下文，让后续旧 LLM 修复能定位原始材料。

### `_normalize_sample`

[查看源码](../llm_wiki_bench/bench_error_book.py#L424) · `10` 行函数体（含已有文档）

【旧数据升级】把字符串样本转成带 name/fixed 状态的字典，兼容已有字典。

### `_sample_name`

[查看源码](../llm_wiki_bench/bench_error_book.py#L437) · `5` 行函数体（含已有文档）

【旧字段读取】从字符串或字典样本提取页面／错误名称。

### `_sample_is_fixed`

[查看源码](../llm_wiki_bench/bench_error_book.py#L445) · `5` 行函数体（含已有文档）

【旧状态读取】判断样本是否明确标记 fixed，旧字符串视作未修复。

### `get_unfixed_samples`

[查看源码](../llm_wiki_bench/bench_error_book.py#L453) · `4` 行函数体（含已有文档）

【旧队列读取】从一个错误条目取尚未修复的样本名称列表。

### `has_unfixed_samples`

[查看源码](../llm_wiki_bench/bench_error_book.py#L460) · `11` 行函数体（含已有文档）

【旧队列检查】判断指定类别或全错误簿是否还有未修复样本，返回布尔值。

### `mark_samples_fixed`

[查看源码](../llm_wiki_bench/bench_error_book.py#L474) · `69` 行函数体（含已有文档）

【旧状态写入】用精确、前后缀及别名拆分等宽松规则匹配修复名称，更新样本、计数及关闭状态后保存；不是严格事实 ID 匹配。

### `print_error_book`

[查看源码](../llm_wiki_bench/bench_error_book.py#L548) · `45` 行函数体（含已有文档）

【旧状态显示】终端打印错误簿概览和样本，不生成或修复 wiki 内容。

### `_cleanup_old_closed`

[查看源码](../llm_wiki_bench/bench_error_book.py#L598) · `24` 行函数体（含已有文档）

【旧记录清理】过滤已关闭超过 max_age_days 的条目；清理的是错误记录，不是知识页面。

### `_get_ledger_path`

[查看源码](../llm_wiki_bench/bench_error_book.py#L627) · `6` 行函数体（含已有文档）

【旧修复日志】返回 lint_ledger.jsonl 路径，供修复流水追加使用。

### `append_ledger`

[查看源码](../llm_wiki_bench/bench_error_book.py#L636) · `34` 行函数体（含已有文档）

【旧日志写入】追加一条带问题类型、文件、修复方式及数量的 JSONL 记录；auto_fixed 区分代码修复与模型修复。

### `load_ledger`

[查看源码](../llm_wiki_bench/bench_error_book.py#L673) · `19` 行函数体（含已有文档）

【旧日志读取】读回修复流水 JSONL，供统计／查看历史。

### `get_unfixed_samples_full`

[查看源码](../llm_wiki_bench/bench_error_book.py#L695) · `12` 行函数体（含已有文档）

【旧修复材料】按类别返回未修复样本的完整字典及上下文，供旧修复流程读取文章。

## bench_ingest.py

### `_read_file_safe`

[查看源码](../llm_wiki_bench/bench_ingest.py#L53) · `8` 行函数体（含已有文档）

【旧流程辅助】读取指定文件并截到 max_len；不存在或失败用提示字符串代替，当前构建循环不调用。

### `_apply_prompt_safety_valve`

[查看源码](../llm_wiki_bench/bench_ingest.py#L71) · `71` 行函数体（含已有文档）

【旧提示词裁剪】超总字符预算时依次裁掉页面名、页面正文、目录索引尾部；这是字符启发式，不是精确 token 控制。

### `_apply_prompt_safety_valve._find_section_end`

[查看源码](../llm_wiki_bench/bench_ingest.py#L97) · `9` 行函数体（含已有文档）

【旧局部辅助】查找给定提示词章节之后的下一个二级标题位置，用于裁剪章节。

### `_compute_index_relevance`

[查看源码](../llm_wiki_bench/bench_ingest.py#L150) · `17` 行函数体（含已有文档）

【旧候选评分】用标题词与目录索引的匹配密度及目录名称匹配评分，仅用于超预算索引裁剪；不是当前 BM25。

### `_get_all_index_content`

[查看源码](../llm_wiki_bench/bench_ingest.py#L170) · `58` 行函数体（含已有文档）

【旧提示词材料】读取目录 _index.md，总长超预算时按文章标题相关性删减；当前入口不依赖这些索引选页。

### `_expand_dirs_from_selected`

[查看源码](../llm_wiki_bench/bench_ingest.py#L231) · `21` 行函数体（含已有文档）

【旧提示词材料】从第一步选中的页推导需展开的目录白名单；没有选页返回 None 表示全展开。

### `_extract_candidate_names`

[查看源码](../llm_wiki_bench/bench_ingest.py#L255) · `19` 行函数体（含已有文档）

【旧名称启发式】用中文连续片段和英文大写词提取候选专名，仅辅助别名裁剪，不做实体识别模型调用。

### `_get_existing_page_names`

[查看源码](../llm_wiki_bench/bench_ingest.py#L277) · `55` 行函数体（含已有文档）

【旧提示词材料】列出既有页名，按目录白名单和候选专名决定是否展开页面及别名，降低输入长度。

### `build_select_pages_prompt`

[查看源码](../llm_wiki_bench/bench_ingest.py#L337) · `73` 行函数体（含已有文档）

【旧提示词构造】组合单篇文章与目录索引，要求模型返回待看页面；本函数只返回提示词，调用模型由旧主流程负责。

### `build_select_pages_batch_prompt`

[查看源码](../llm_wiki_bench/bench_ingest.py#L415) · `52` 行函数体（含已有文档）

【旧提示词构造】为多篇文章构造共享选页提示词；模型返回批次共用页面集合，不是逐篇独立搜索。

### `build_ingest_prompt`

[查看源码](../llm_wiki_bench/bench_ingest.py#L472) · `162` 行函数体（含已有文档）

【旧提示词构造】拼 purpose、schema、文章及已选页正文，要求生成整页 FILE 块；不用于当前事实工具构建。

### `build_ingest_prompt_batch`

[查看源码](../llm_wiki_bench/bench_ingest.py#L639) · `121` 行函数体（含已有文档）

【旧提示词构造】批量拼接文章与已读页面，要求一次生成多个文件及目录变更；与当前逐文档工具循环不同。

### `_predict_article_stem`

[查看源码](../llm_wiki_bench/bench_ingest.py#L763) · `7` 行函数体（含已有文档）

【旧路径辅助】按标题清洗规则预测文章文件名，供旧 digest 关联使用；不提供来源身份保证。

### `_normalize_filename`

[查看源码](../llm_wiki_bench/bench_ingest.py#L775) · `15` 行函数体（含已有文档）

【旧名称清洗】统一全半角、空白等文件名字符，返回规范名称；不判断实体是否相同。

### `_sanitize_frontmatter`

[查看源码](../llm_wiki_bench/bench_ingest.py#L793) · `58` 行函数体（含已有文档）

【旧元数据改写】清理逗号和列表形式、去 confidence 并重排字段；供旧整页写入使用。

### `_extract_type_from_content`

[查看源码](../llm_wiki_bench/bench_ingest.py#L854) · `12` 行函数体（含已有文档）

【旧元数据读取】从页面 frontmatter 抽取 type，供旧写入器决定目录。

### `_check_frontmatter_complete`

[查看源码](../llm_wiki_bench/bench_ingest.py#L869) · `14` 行函数体（含已有文档）

【旧完整性检查】判断 frontmatter 是否闭合，帮助拒绝模型截断的文件；不验证正文事实完整。

### `_check_digest_completeness`

[查看源码](../llm_wiki_bench/bench_ingest.py#L889) · `72` 行函数体（含已有文档）

【旧摘要修补】检查 digest 必要章节并为缺失项补占位内容；有章节不等于有真实证据。

### `_inject_dates`

[查看源码](../llm_wiki_bench/bench_ingest.py#L964) · `50` 行函数体（含已有文档）

【旧元数据改写】向页面注入 created/updated 日期；这是编辑日期，不是事实发生时间。

### `_lcs_len`

[查看源码](../llm_wiki_bench/bench_ingest.py#L1017) · `13` 行函数体（含已有文档）

【旧模糊匹配】动态规划计算两字符串最长公共子序列长度，供文章名近似匹配。

### `_fuzzy_match_article`

[查看源码](../llm_wiki_bench/bench_ingest.py#L1033) · `16` 行函数体（含已有文档）

【旧来源猜测】在候选文件名中选近似文章名；属于旧启发式，不能替代当前明确 source_id/version_id。

### `_inject_article_link_to_digests`

[查看源码](../llm_wiki_bench/bench_ingest.py#L1052) · `56` 行函数体（含已有文档）

【旧关联写盘】为新 digest 猜配并写 source_article 与 Original 链接；当前事实提交不用此模糊关联。

### `_fix_digest_article_links`

[查看源码](../llm_wiki_bench/bench_ingest.py#L1111) · `51` 行函数体（含已有文档）

【旧链接修复】检查并修正 digest 内的 sources/articles 链接；会改写旧页面。

### `_rebuild_sources_index`

[查看源码](../llm_wiki_bench/bench_ingest.py#L1165) · `90` 行函数体（含已有文档）

【旧索引写盘】扫描旧 articles/digests，程序化重建来源目录索引；当前实时检索不依赖它。

### `_rebuild_global_index`

[查看源码](../llm_wiki_bench/bench_ingest.py#L1258) · `37` 行函数体（含已有文档）

【旧全局索引】重建 index.md 目录统计；update_overview=True 时还请求模型概览，否则保留旧概览。

### `_extract_existing_overview`

[查看源码](../llm_wiki_bench/bench_ingest.py#L1298) · `8` 行函数体（含已有文档）

【旧文本解析】从 index.md 提取已有知识概览文字，不调用模型。

### `_generate_overview_text`

[查看源码](../llm_wiki_bench/bench_ingest.py#L1309) · `51` 行函数体（含已有文档）

【旧概览／LLM】汇集现有知识材料，让模型生成全库概览文字；当前 QA 入口使用实时树而非该概览。

### `_count_knowledge_pages`

[查看源码](../llm_wiki_bench/bench_ingest.py#L1369) · `12` 行函数体（含已有文档）

【旧统计】统计配置知识目录的页面数，排除 sources；不运行检索或模型。

### `quick_lint_bench`

[查看源码](../llm_wiki_bench/bench_ingest.py#L1384) · `360` 行函数体（含已有文档）

【旧结构检查】扫描断链、索引一致性、类型目录、digest 完整度和重复页等，返回问题分类；不能代替当前来源版本／受控事实审计。

### `auto_fix_bench`

[查看源码](../llm_wiki_bench/bench_ingest.py#L1747) · `538` 行函数体（含已有文档）

【旧自动改写】按启发式修链接、目录、索引和页面格式，部分分支删链接或文件；不调用 LLM，但也不是当前 FactStore 的增量写入规则。

### `llm_fix_incomplete_digests`

[查看源码](../llm_wiki_bench/bench_ingest.py#L2289) · `153` 行函数体（含已有文档）

【旧修复／LLM】扫描缺章节或只有占位符的 digest，用旧原文补生成内容并写回。

### `llm_fix_missing_summary`

[查看源码](../llm_wiki_bench/bench_ingest.py#L2445) · `90` 行函数体（含已有文档）

【旧修复／LLM】为缺少标题后 blockquote 概览的知识页生成一句摘要并写回。

### `llm_fix_missing_sections`

[查看源码](../llm_wiki_bench/bench_ingest.py#L2538) · `134` 行函数体（含已有文档）

【旧修复／LLM】为知识页补 Key Facts、Related Pages 等必要章节，改写旧页面。

### `llm_fix_empty_related_pages`

[查看源码](../llm_wiki_bench/bench_ingest.py#L2675) · `296` 行函数体（含已有文档）

【旧修复／LLM】用共同 digest、标签和反向链接预选候选，再让模型挑关联页并写 Related Pages。

### `_write_related_pages`

[查看源码](../llm_wiki_bench/bench_ingest.py#L2974) · `65` 行函数体（含已有文档）

【旧章节写盘】把模型选定的链接列表写进 Related Pages，返回修补结果；不验证关系事实的原文范围。

### `_inject_related_sources_for_page`

[查看源码](../llm_wiki_bench/bench_ingest.py#L3042) · `76` 行函数体（含已有文档）

【旧反向关联】扫描 digest 是否提及页名或别名，把匹配项写成 Related Sources；名称共现不是严格来源证据。

### `llm_fix_broken_links`

[查看源码](../llm_wiki_bench/bench_ingest.py#L3121) · `178` 行函数体（含已有文档）

【旧修复／LLM】从 Error Book 取未修复断链，让模型创建缺失知识页并更新修复状态；当前构建不自动调用。

### `llm_verify_source_grounding`

[查看源码](../llm_wiki_bench/bench_ingest.py#L3302) · `167` 行函数体（含已有文档）

【旧抽样核验／LLM】随机抽知识页，以其来源 digest 检查 Key Facts 并标记／移除不支持的内容；不是逐条不可变原文校验。

### `llm_detect_contradictions`

[查看源码](../llm_wiki_bench/bench_ingest.py#L3472) · `156` 行函数体（含已有文档）

【旧抽样核验／LLM】抽页面及 Related Pages 检查矛盾，参考 digest 修正内容；不等同当前保留 conflicts_with 的事实追加机制。

### `llm_fix_structural`

[查看源码](../llm_wiki_bench/bench_ingest.py#L3631) · `32` 行函数体（含已有文档）

【旧修复编排／LLM】汇总 digest、概览、章节、断链和关联页补全步骤，返回各类修复数量。

### `llm_fix_content`

[查看源码](../llm_wiki_bench/bench_ingest.py#L3666) · `25` 行函数体（含已有文档）

【旧修复编排／LLM】运行来源支持抽检与跨页矛盾检查，返回修复计数。

### `llm_fix_all`

[查看源码](../llm_wiki_bench/bench_ingest.py#L3694) · `9` 行函数体（含已有文档）

【旧修复编排／LLM】合并结构与内容修复结果，供旧 finalize 等入口使用。

### `merge_duplicate_pages`

[查看源码](../llm_wiki_bench/bench_ingest.py#L3706) · `241` 行函数体（含已有文档）

【旧合并／LLM】从目录索引识别疑似重复页面并让模型合并，可能改写／删除页面；当前事实流程不自动调用。

### `detect_alias_overlaps`

[查看源码](../llm_wiki_bench/bench_ingest.py#L3950) · `93` 行函数体（含已有文档）

【旧合并辅助】扫描同目录别名重叠并处理候选冲突；别名重叠不是当前实体身份校验依据。

### `generate_overview`

[查看源码](../llm_wiki_bench/bench_ingest.py#L4046) · `70` 行函数体（含已有文档）

【旧概览／LLM】生成全库知识概览并更新 index.md，曾作零跳上下文；当前 Agent 起点是 tree。

### `_parse_index_sections`

[查看源码](../llm_wiki_bench/bench_ingest.py#L4120) · `18` 行函数体（含已有文档）

【旧索引解析】把 _index.md 拆为章节标题与条目列表，标题之前内容存 __preamble__。

### `_assemble_sections`

[查看源码](../llm_wiki_bench/bench_ingest.py#L4141) · `13` 行函数体（含已有文档）

【旧索引序列化】将已解析的章节及条目重新拼为 Markdown 文本。

### `_extract_entry_name`

[查看源码](../llm_wiki_bench/bench_ingest.py#L4157) · `7` 行函数体（含已有文档）

【旧索引解析】从列表 wikilink 中提取页面名称，供章节归类使用。

### `relocate_pending_entries`

[查看源码](../llm_wiki_bench/bench_ingest.py#L4167) · `302` 行函数体（含已有文档）

【旧分类／LLM】让模型把 Unsorted 条目归到既有或新章节，随后改写索引；force_final 控制末轮兜底归类。

### `consolidate_wiki_bench`

[查看源码](../llm_wiki_bench/bench_ingest.py#L4472) · `137` 行函数体（含已有文档）

【旧目录整理／LLM】检查目录规模，超阈值时请求拆分／合并／迁移建议并执行；返回 skipped/no_changes/executed。

### `_apply_consolidate_changes`

[查看源码](../llm_wiki_bench/bench_ingest.py#L4612) · `110` 行函数体（含已有文档）

【旧目录迁移】执行模型给出的拆分、合并或移页指令，更新文件和目录配置；不是多文件事务。

### `_update_wiki_references`

[查看源码](../llm_wiki_bench/bench_ingest.py#L4725) · `51` 行函数体（含已有文档）

【旧链接迁移】页面移动后扫描全库，把旧目录路径链接改为新路径。

### `periodic_maintenance`

[查看源码](../llm_wiki_bench/bench_ingest.py#L4779) · `72` 行函数体（含已有文档）

【旧定期维护／LLM】按计数阈值触发结构／内容修复、归类、合并及索引更新；当前 ingest_batch 不调用。

### `finalize_wiki`

[查看源码](../llm_wiki_bench/bench_ingest.py#L4854) · `110` 行函数体（含已有文档）

【旧收尾／LLM】执行多轮 lint、代码修复和模型修复，再重建索引及概览；不属于当前默认构建的完成条件。

### `parse_file_outputs`

[查看源码](../llm_wiki_bench/bench_ingest.py#L4968) · `21` 行函数体（含已有文档）

【旧输出解析】将模型文本里的 FILE 分隔块解析成路径到正文映射，并排除后面的 DIR_CHANGES；相同路径后块覆盖前块。

### `parse_dir_changes`

[查看源码](../llm_wiki_bench/bench_ingest.py#L4992) · `16` 行函数体（含已有文档）

【旧输出解析】读取 DIR_CHANGES 后的 JSON，失败再试方括号片段；无法解析返回空列表。

### `write_wiki_files`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5012) · `320` 行函数体（含已有文档）

【旧整页写盘】规范模型文件路径、元数据和链接，尝试保护未展示旧页并处理同名页，最终写页面及索引；不使用当前 revision＋引用＋事实增量合同。

### `write_wiki_files._is_sources_path`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5058) · `2` 行函数体（含已有文档）

【旧局部辅助】判断是否以 wiki/sources/ 开头，以决定来源目录例外处理。

### `write_wiki_files._normalize_wikilink_targets`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5111) · `12` 行函数体（含已有文档）

【旧局部辅助】用正则逐个规范 wikilink 目标中的文件名，返回修改后的正文。

### `write_wiki_files._normalize_wikilink_targets._norm_link`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5113) · `9` 行函数体（含已有文档）

【旧局部辅助】保留目录前缀、清洗单个链接末段名称并重新包成 wikilink。

### `_append_to_index`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5335) · `57` 行函数体（含已有文档）

【旧索引写盘】把确实位于该目录的新页追加到 Unsorted 章节，并维护目录索引。

### `load_cache`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5397) · `8` 行函数体（含已有文档）

【旧缓存读取】读取 SHA 缓存 JSON，缺失或损坏返回空字典；当前构建采用版本 receipt，不依赖此缓存。

### `save_cache`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5408) · `4` 行函数体（含已有文档）

【旧缓存写盘】把整份 SHA 缓存写回配置路径；不核验缓存对应产物。

### `_legacy_ingest_single`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5416) · `107` 行函数体（含已有文档）

【旧单篇／LLM】SHA 跳过后先模型选页、再整页生成，解析写盘并存缓存；保留供历史对照，当前 ingest_single 已转向 BuildAgent。

### `_read_selected_pages`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5526) · `53` 行函数体（含已有文档）

【旧页面读取】读第一步选中页并拼到提示词，超总字符限额时只保留名称；不是当前工具的可续读窗口。

### `_save_article_original`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5582) · `35` 行函数体（含已有文档）

【旧原文写盘】保存到 sources/articles 并返回文件 stem；与当前 SourceStore 的身份／内容双哈希快照不同。

### `_ingest_batch_one`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5621) · `117` 行函数体（含已有文档）

【旧批次／LLM】批次共享一次选页及一次整页生成，写盘和来源链接后为每篇写 SHA 缓存；force 在本函数内没有参与跳过判断，外层负责过滤。

### `_legacy_ingest_batch`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5741) · `119` 行函数体（含已有文档）

【旧总编排／LLM】根据 SHA 缓存筛选文章、分批运行旧生成并处理维护及最终修复；不是当前公开 ingest_batch 的实现。

### `_legacy_ingest_batch._format_time`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5773) · `14` 行函数体（含已有文档）

【旧进度辅助】把秒数格式化为时分秒或分秒文本。

### `_legacy_ingest_batch._print_progress_bar`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5789) · `21` 行函数体（含已有文档）

【旧进度辅助】按完成数、耗时估算进度和剩余时间并输出，不控制实际任务调度。

### `ingest_single`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5864) · `5` 行函数体（含已有文档）

【当前兼容入口／间接 LLM】把单篇交 ingest_documents；cache 参数忽略，以文档和摘要均无失败返回 True。

### `ingest_batch`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5872) · `8` 行函数体（含已有文档）

【当前构建入口／间接 LLM】直接转交 ingest_documents；batch_size 只兼容旧签名，limit 是文章数，force 控制文档重建，不运行旧整页维护。

### `main`

[查看源码](../llm_wiki_bench/bench_ingest.py#L5883) · `40` 行函数体（含已有文档）

【当前构建 CLI】设置数据集并排序枚举 raw/articles，按 --limit 截文章后调用当前 ingest_batch；文档或摘要失败时退出码 1。

## build_agent.py

### `covered`

[查看源码](../llm_wiki_bench/build_agent.py#L58) · `9` 行函数体（含已有文档）

【区间校验】排序并合并已读区间，判断能否无缺口覆盖 [start,end)；只验证工具返回的字符覆盖，不证明模型理解或抽取完整。

### `BuildAgent.__init__`

[查看源码](../llm_wiki_bench/build_agent.py#L71) · `7` 行函数体（含已有文档）

【构建初始化】注入模型调用函数、模型名和每文档工具预算，组合原文库、事实库和检索器；默认使用 premium 模型。

### `BuildAgent._verify_products`

[查看源码](../llm_wiki_bench/build_agent.py#L81) · `18` 行函数体（含已有文档）

【完成校验】核对 complete receipt 的来源快照和非空 products；逐页寻找记录的事实 ID，确认含本来源版本且全部引文可校验。任何缺失返回 False；不检查语义蕴含或抽取遗漏。

### `BuildAgent.ingest`

[查看源码](../llm_wiki_bench/build_agent.py#L104) · `132` 行函数体（含已有文档）

【构建主循环／LLM】归档整篇文章，复核完成缓存，给模型 purpose 和实时目录树；逐次执行读、检索、fact_apply 与 finish_document。跟踪已读原文区间及页面 revision；写事实前要求引文已读。预算、重复调用或失败会停止并保存 partial receipt；已提交页面不回滚。complete 需全文读覆盖、有效产物且 unresolved 为空。

### `ingest_documents`

[查看源码](../llm_wiki_bench/build_agent.py#L240) · `34` 行函数体（含已有文档）

【批量编排／LLM】依次构建 paths[:limit]，逐篇捕获失败并统计；再以完成文档的产物作为种子运行摘要编译，保存 summary-last-run.json。summary_limit=0 关闭摘要；文档 force 不会自动传给摘要 force。

## build_summaries.py

### `eligible_pages`

[查看源码](../llm_wiki_bench/build_summaries.py#L43) · `20` 行函数体（含已有文档）

【摘要候选输入】扫描知识页，挑出含当前 fact/relation 且引文可校验的页面；返回页面资料及问题清单，旧自由正文不能自动充当叶事实。

### `candidate_pairs`

[查看源码](../llm_wiki_bench/build_summaries.py#L67) · `29` 行函数体（含已有文档）

【确定性组队】先保留可重建的已有摘要组，再按显式链接优先、BM25 分数次之，为每个种子取最多三个候选并去重。新组是页面对；相关性最终交给模型，词重叠不证明关系。

### `evidence_packet`

[查看源码](../llm_wiki_bench/build_summaries.py#L100) · `20` 行函数体（含已有文档）

【输入预算】按现有事实顺序每页选最多八条，均分字符预算且不截断事实/引文；超大事实跳过，无可选事实则失败。返回材料包和临时 evidence_id／fact_id 到支撑位置的映射，不是 LLM 选事实。

### `build_summaries`

[查看源码](../llm_wiki_bench/build_summaries.py#L125) · `97` 行函数体（含已有文档）

【摘要编译／LLM】检查资格、候选和缓存，每组一次模型调用，只接受一个 write_summary 或 skip_summary；把模型支撑 ID 转成真实事实引用后交 SummaryStore.apply。无变化的 skipped/部分 failed 决策会复用；模型无响应不写失败缓存。dry_run 不调用模型、不写内部文件；force 才绕过缓存。

### `main`

[查看源码](../llm_wiki_bench/build_summaries.py#L225) · `16` 行函数体（含已有文档）

【摘要 CLI】解析 wiki-dir、limit、model、force、dry-run，打印统计并可写 output；即使 dry-run，显式指定 output 仍会写报告，failed 非零时退出码为 1。

## download_datasets.py

### `download_hotpotqa`

[查看源码](../llm_wiki_bench/download_datasets.py#L26) · `30` 行函数体（含已有文档）

【数据下载】已有目标文件时复用，否则用 HTTP 下载 HotpotQA dev JSON；成功返回文件路径，失败返回 None。

### `download_musique`

[查看源码](../llm_wiki_bench/download_datasets.py#L59) · `39` 行函数体（含已有文档）

【数据下载】尝试获取 MuSiQue dev 数据并保存到目标目录；已有文件复用，失败打印手动获取提示并返回 None。

### `download_2wikimhqa`

[查看源码](../llm_wiki_bench/download_datasets.py#L101) · `34` 行函数体（含已有文档）

【数据下载】尝试获取 2Wiki dev JSON；已有文件复用，失败提示其他获取方式并返回 None。

### `main`

[查看源码](../llm_wiki_bench/download_datasets.py#L145) · `32` 行函数体（含已有文档）

【下载 CLI】指定 --dataset 时下载单个数据集，省略时下载全部；只获取数据，不预处理也不构建 wiki。

## evaluate.py

### `_normalize_answer`

[查看源码](../llm_wiki_bench/evaluate.py#L32) · `7` 行函数体（含已有文档）

【评分归一】小写化，去英文冠词、ASCII 标点并压缩空白；专为答案文本比较，不是语义等价判断。

### `exact_match`

[查看源码](../llm_wiki_bench/evaluate.py#L42) · `2` 行函数体（含已有文档）

【答案 EM】比较归一化后的预测与标准答案是否完全相同，返回 0.0 或 1.0。

### `token_f1`

[查看源码](../llm_wiki_bench/evaluate.py#L47) · `14` 行函数体（含已有文档）

【答案 F1】按归一化后空白 token 的多重集合交集计算精确率／召回率调和平均；两个空答案为 1，单侧空为 0。

### `score_against_aliases`

[查看源码](../llm_wiki_bench/evaluate.py#L64) · `7` 行函数体（含已有文档）

【别名评分】遍历标准答案及别名，分别取最高 EM 和 F1；两者最优值可以来自不同别名。

### `_load_qa_pairs`

[查看源码](../llm_wiki_bench/evaluate.py#L76) · `7` 行函数体（含已有文档）

【评估输入】从 data/<dataset>/qa_pairs.jsonl 读取非空行，缺失时退出并提示预处理。

### `_load_predictions`

[查看源码](../llm_wiki_bench/evaluate.py#L86) · `9` 行函数体（含已有文档）

【评估输入】按问题 id 建预测字典；同一 id 多条记录时后者覆盖前者。

### `evaluate`

[查看源码](../llm_wiki_bench/evaluate.py#L101) · `96` 行函数体（含已有文档）

【评估聚合】对有预测的题目计算 EM/F1、标题召回代理和用量，并按支撑标题数／题型分组；缺失预测单独计 missing，不进入平均分分母。返回 (汇总, 逐题结果)，不核验引文语义。

### `_print_summary`

[查看源码](../llm_wiki_bench/evaluate.py#L202) · `19` 行函数体（含已有文档）

【显示】把汇总指标输出到终端；不改变预测或评估结果。

### `main`

[查看源码](../llm_wiki_bench/evaluate.py#L224) · `39` 行函数体（含已有文档）

【评估 CLI】读取 QA 和 predictions，应用题数限制后评分，写 summary JSON 与 details JSONL；不调用模型。

## llm_client.py

### `_api_base`

[查看源码](../llm_wiki_bench/llm_client.py#L28) · `2` 行函数体（含已有文档）

【HTTP 配置】从进程环境读取 OPENAI_BASE_URL 并去尾斜杠；这里只拼端点，不自动加载 .env。

### `_headers`

[查看源码](../llm_wiki_bench/llm_client.py#L33) · `6` 行函数体（含已有文档）

【HTTP 配置】从环境读取密钥，构造 JSON 与 Bearer 请求头；缺失密钥时 Authorization 为空。

### `call_llm`

[查看源码](../llm_wiki_bench/llm_client.py#L43) · `90` 行函数体（含已有文档）

【文本模型调用】构造 system/user 消息并 POST /chat/completions；可请求 JSON，最多五次尝试，失败最终抛 RuntimeError。返回文本而非已执行动作；无 content 时可能退回提供商 reasoning 字段。

### `call_llm_json`

[查看源码](../llm_wiki_bench/llm_client.py#L137) · `36` 行函数体（含已有文档）

【旧 JSON 调用】先请求 JSON 模式，调用失败后回退普通文本；依次尝试整体 JSON、代码围栏、首左花括号到末右花括号的切片。不是平衡括号解析器，也不保证返回值一定是 dict。

### `call_llm_with_tools`

[查看源码](../llm_wiki_bench/llm_client.py#L177) · `56` 行函数体（含已有文档）

【当前模型接口】发送完整 messages 与工具 Schema，tool_choice=auto；返回 assistant message 并附 _usage，重试耗尽返回 None。工具调用仍只是提议，调用方负责解析、验证和执行；一次逻辑调用可能发生多次 HTTP 请求。

## preprocess_bench.py

### `sanitize_filename`

[查看源码](../llm_wiki_bench/preprocess_bench.py#L37) · `10` 行函数体（含已有文档）

【文件名清洗】将非法文件名字符替换为下划线并规范空白、截长；碰撞由 _write_articles 的哈希后缀处理。

### `_write_articles`

[查看源码](../llm_wiki_bench/preprocess_bench.py#L51) · `18` 行函数体（含已有文档）

【文章落盘】将去重后的标题／正文变体写成 Markdown，保留明确 source_identity；文件名加哈希避免清洗后的同名碰撞。原文版本稍后由 SourceStore 对完整输入计算。

### `process_hotpotqa`

[查看源码](../llm_wiki_bench/preprocess_bench.py#L72) · `59` 行函数体（含已有文档）

【数据转换】读取 HotpotQA JSON，按 limit 选题，收集 context 中全部不同标题／正文组合（含干扰段落），写文章和统一 qa_pairs.jsonl，返回数量统计。

### `process_musique`

[查看源码](../llm_wiki_bench/preprocess_bench.py#L134) · `74` 行函数体（含已有文档）

【数据转换】读取 MuSiQue JSONL，整理问题、别名及支撑标题并收集不同段落，输出统一文章与 QA 文件；不调用模型。

### `process_2wikimhqa`

[查看源码](../llm_wiki_bench/preprocess_bench.py#L211) · `55` 行函数体（含已有文档）

【数据转换】读取 2Wiki JSON，提取题目、支撑标题与上下文文章，保留标题／正文变体并写统一文件；不调用模型。

### `main`

[查看源码](../llm_wiki_bench/preprocess_bench.py#L285) · `53` 行函数体（含已有文档）

【预处理 CLI】解析数据集和题数限制，调对应处理器生成 raw/articles 与 data/qa_pairs；已有同名输出可能被改写。

## resume_summaries.py

### `log`

[查看源码](../llm_wiki_bench/resume_summaries.py#L14) · `2` 行函数体（含已有文档）

【进度输出】带当前本地时分秒打印并立即 flush，让长摘要任务的进度可见。

### `main`

[查看源码](../llm_wiki_bench/resume_summaries.py#L19) · `48` 行函数体（含已有文档）

【摘要续跑／LLM】按轮调用 build_summaries 并保存独立报告，处理输入完整性、无进展及轮数上限；复用成功／跳过／失败缓存，因此队列清空不等于全部摘要成功。

### `main.call`

[查看源码](../llm_wiki_bench/resume_summaries.py#L33) · `12` 行函数体（含已有文档）

【调用包装／LLM】打印本次候选组和耗时，转发真实模型调用并计连续无响应；已有三次连续失败时，在下一次调用入口停止，不把格式校验失败算无响应。

## retrieval_experiment.py

### `compare`

[查看源码](../llm_wiki_bench/retrieval_experiment.py#L18) · `28` 行函数体（含已有文档）

【检索实验】用完整问题比较 BM25、实体匹配及可选重排在 k=5/10/15 的支撑标题覆盖；无模型调用（外部 reranker 行为由调用者决定），不是最终回答质量或来源蕴含评估。

### `main`

[查看源码](../llm_wiki_bench/retrieval_experiment.py#L49) · `11` 行函数体（含已有文档）

【检索实验 CLI】读取前 N 条 QA，运行 compare 并写报告；不执行 QA Agent。

## run.py

### `step_download`

[查看源码](../llm_wiki_bench/run.py#L36) · `15` 行函数体（含已有文档）

【下载编排】查数据集下载器并执行，打印阶段信息；以返回路径是否存在值判断成功，不调用模型。

### `step_preprocess`

[查看源码](../llm_wiki_bench/run.py#L54) · `26` 行函数体（含已有文档）

【预处理编排】检查下载文件并选择数据集处理器，传 limit 限制 QA 样本数；写文章和 qa_pairs，返回是否成功。

### `step_ingest`

[查看源码](../llm_wiki_bench/run.py#L84) · `27` 行函数体（含已有文档）

【构建编排／间接 LLM】配置数据集、创建目录，排序读取全部已有文章后调用当前 ingest_batch；文档或摘要失败均返回 False。这里不接收 run 的 limit。

### `run_one`

[查看源码](../llm_wiki_bench/run.py#L114) · `16` 行函数体（含已有文档）

【阶段控制】按 only/skip 参数依次运行下载、预处理和构建，前序失败阻止后续；打印耗时并返回总状态。

### `main`

[查看源码](../llm_wiki_bench/run.py#L133) · `48` 行函数体（含已有文档）

【离线 CLI】解析数据集和阶段选项，互斥校验 only 参数后逐数据集运行；--limit 限制预处理题数，不限制已有文章的 only-ingest。

## run_qa.py

### `main`

[查看源码](../llm_wiki_bench/run_qa.py#L38) · `129` 行函数体（含已有文档）

【QA CLI／间接 LLM】读取前 N 条题目，创建单个 WikiAgent 逐题回答，写 prediction、引用、推理、缺口及轨迹；可接着评估。output 用 w 打开会覆盖同名文件，逐条 flush，不是断点续跑。

## summary_store.py

### `summary_state`

[查看源码](../llm_wiki_bench/summary_store.py#L22) · `30` 行函数体（含已有文档）

【摘要解析】提取唯一 wiki-summary JSON 块，验证 level=1、schema_version、子页、陈述和来源集合基本形状；不验证陈述语义。

### `summary_state.require`

[查看源码](../llm_wiki_bench/summary_store.py#L30) · `3` 行函数体（含已有文档）

【局部校验】把摘要字段检查的布尔结果统一转换为 ValueError，供 summary_state 拒绝坏记录。

### `current_facts`

[查看源码](../llm_wiki_bench/summary_store.py#L55) · `5` 行函数体（含已有文档）

【叶事实筛选】保留 fact/relation，去掉被任何 supersedes 指向的事实及 kind=summary；显式冲突双方继续保留。

### `summary_text`

[查看源码](../llm_wiki_bench/summary_store.py#L63) · `3` 行函数体（含已有文档）

【展示】把摘要 claims 拼成概览文字；为 synthesis 加前缀，提醒这是综合推导。

### `SummaryStore.__init__`

[查看源码](../llm_wiki_bench/summary_store.py#L70) · `3` 行函数体（含已有文档）

【初始化】绑定 wiki 根目录和原文库；不创建摘要，也不调用模型。

### `SummaryStore.path_for`

[查看源码](../llm_wiki_bench/summary_store.py#L76) · `2` 行函数体（含已有文档）

【摘要身份】对子页路径排序后哈希得到 summaries/*.md；相同子页组重建复用路径，内容版本另由 revision 表示。

### `SummaryStore.leaf`

[查看源码](../llm_wiki_bench/summary_store.py#L80) · `10` 行函数体（含已有文档）

【叶节点读取】只接受知识 Markdown 页，排除来源副本及递归摘要；返回页 revision 和当前事实 ID 映射。

### `SummaryStore.versions`

[查看源码](../llm_wiki_bench/summary_store.py#L92) · `4` 行函数体（含已有文档）

【依赖快照】列出指定 source_id 已归档的全部 version_id；这是版本集合，不负责判断哪一版本最新。

### `SummaryStore.apply`

[查看源码](../llm_wiki_bench/summary_store.py#L99) · `70` 行函数体（含已有文档）

【摘要提交】校验 2..4 个不同子页、版本、1..8 条 claims 和支撑事实；由代码从事实推导引文，要求整体使用每个子页及至少两个来源身份。加锁复核依赖，保留旧摘要历史，再原子写入；不验证 direct/synthesis 的语义分类是否正确。

### `SummaryStore.get`

[查看源码](../llm_wiki_bench/summary_store.py#L171) · `8` 行函数体（含已有文档）

【摘要读取】按路径解析摘要并检查子页集合与摘要路径哈希一致，返回 (state, revision)；新鲜度由 freshness 另查。

### `SummaryStore.freshness`

[查看源码](../llm_wiki_bench/summary_store.py#L181) · `15` 行函数体（含已有文档）

【失效判定】比较子页 revision、来源版本集合并校验引文，返回过期原因列表；空列表仅代表依赖未变，不代表知识最新或语义正确。

### `SummaryStore.read`

[查看源码](../llm_wiki_bench/summary_store.py#L199) · `33` 行函数体（含已有文档）

【渐进披露】overview 返回概览，claims 返回陈述及支撑映射，evidence 一次展开一个 claim 的事实和引文。过期摘要仅返回状态及子页导航；这些读取不计为 QA 已读原文。

## validate_wiki.py

### `repair_metadata`

[查看源码](../llm_wiki_bench/validate_wiki.py#L20) · `19` 行函数体（含已有文档）

【有限修复】给部分可确定的 YAML 列表或标量补 JSON 引号；只改可识别格式，不推测缺失事实。

### `audit`

[查看源码](../llm_wiki_bench/validate_wiki.py#L43) · `104` 行函数体（含已有文档）

【审计／可选修复】扫描可见 Markdown，检查原文完整性、摘要失效、事实引用和链接；repair=True 时备份后修围栏／可确定元数据／唯一裸链接并归档旧原文。不会用 LLM，也不重写受控摘要或原文版本；返回问题和数量，不保证语义正确。

### `audit.resolve`

[查看源码](../llm_wiki_bench/validate_wiki.py#L90) · `19` 行函数体（含已有文档）

【局部链接处理】检查 wikilink 目标并记录问题；仅 repair 模式且裸名称恰有一个候选时替换路径，保留标签和锚点。

### `main`

[查看源码](../llm_wiki_bench/validate_wiki.py#L150) · `9` 行函数体（含已有文档）

【审计 CLI】读取 wiki-dir 和 repair 选项，执行 audit 并写 output；有未解决问题不自动转成非零退出码，需查看报告。

## wiki_agent.py

### `WikiAgent.__init__`

[查看源码](../llm_wiki_bench/wiki_agent.py#L86) · `10` 行函数体（含已有文档）

【QA 初始化】注入检索器、模型和预算；默认单 Agent，select_pages 限定候选数，patience 仅兼容保留；allow_subtasks 仅显式 Python 调用可启用。

### `WikiAgent._allowed_path`

[查看源码](../llm_wiki_bench/wiki_agent.py#L98) · `2` 行函数体（含已有文档）

【范围检查】未指定 scope 则放行，否则只允许等于范围路径或在其子目录内的路径。

### `WikiAgent.retrieve`

[查看源码](../llm_wiki_bench/wiki_agent.py#L104) · `123` 行函数体（含已有文档）

【问答主循环／LLM】以问题和实时树开始，模型选择检索／读取／记缺口，代码执行并累计工具结果。普通文本不算最终答案，必须通过 _finish；探索预算用尽最多再给两次只提交答案的模型机会。记录工具轨迹、来源读取区间和模型用量，未完成返回 unknown 及缺口。

### `WikiAgent._scope_arguments`

[查看源码](../llm_wiki_bench/wiki_agent.py#L229) · `16` 行函数体（含已有文档）

【工具参数约束】限制搜索候选数并执行可选子任务目录／来源权限；scope 为空的默认主 Agent 不要求先读页才能 source_read。

### `WikiAgent._observe`

[查看源码](../llm_wiki_bench/wiki_agent.py#L247) · `41` 行函数体（含已有文档）

【观测记账】把搜索结果、已读页、source_read 区间及摘要 evidence 展开记录到 RetrievalResult；只有 source_read 建立原文已读覆盖，读摘要不替代读原文。

### `WikiAgent._finish`

[查看源码](../llm_wiki_bench/wiki_agent.py#L291) · `59` 行函数体（含已有文档）

【答案验收】校验字段、状态和原文引文并要求引文区间已读；若显式提交 summary_refs，还检查摘要版本、展开记录及全部支撑引文覆盖。清理 Yes/No 句首格式后写结果；不自动判定推理正确、不证明每一跳齐全，也不能检测模型漏报使用的摘要。

## wiki_retriever.py

### `WikiRetriever.__init__`

[查看源码](../llm_wiki_bench/wiki_retriever.py#L43) · `13` 行函数体（含已有文档）

【检索初始化】绑定 wiki、BM25 模式及可选 reranker 回调，创建内存索引容器和来源／摘要存储对象；不调用 LLM。

### `WikiRetriever.load`

[查看源码](../llm_wiki_bench/wiki_retriever.py#L59) · `43` 行函数体（含已有文档）

【索引刷新】扫描非隐藏、非符号链接的实际文件，用路径/mtime/size 判断是否变化；解析 Markdown、目录及摘要状态，再重建倒排索引和反向摘要导航。坏元数据降级为正文并记录错误。

### `WikiRetriever._parse_page`

[查看源码](../llm_wiki_bench/wiki_retriever.py#L104) · `26` 行函数体（含已有文档）

【页面解析】转换为 WikiPage；普通页抽取标题、别名、标签及链接，原文版本页补侧车元数据，摘要页检查新鲜度并仅以可用概览作为正文。

### `WikiRetriever._parse_page.strings`

[查看源码](../llm_wiki_bench/wiki_retriever.py#L119) · `3` 行函数体（含已有文档）

【字段兼容】把元数据中的列表或单个值转成字符串列表，缺失或空值返回 []。

### `WikiRetriever._tokenize`

[查看源码](../llm_wiki_bench/wiki_retriever.py#L133) · `2` 行函数体（含已有文档）

【词法分词】casefold 后用 Unicode 字母数字片段分词；不做语义向量、词干化或中文词语切分，连续中文可能成为一个 token。

### `WikiRetriever._entity_key`

[查看源码](../llm_wiki_bench/wiki_retriever.py#L138) · `2` 行函数体（含已有文档）

【名称归一】连字符替为空格，再分词拼回字符串；用于完整标题／别名匹配，不证明两个同名条目是同一实体。

### `WikiRetriever._build_bm25`

[查看源码](../llm_wiki_bench/wiki_retriever.py#L143) · `14` 行函数体（含已有文档）

【索引构建】索引标题、别名、标签、描述及正文和事实 statement；剔除受控 JSON 中的哈希与重复引文，建立词频和文长表。这里没有过滤所有被 supersedes 的旧 statement。

### `WikiRetriever._bm25_score`

[查看源码](../llm_wiki_bench/wiki_retriever.py#L159) · `8` 行函数体（含已有文档）

【确定性排序】按词频、逆文档频率和长度归一计算每页 BM25 分数；k1=1.2、b=0.75，无子串额外加分，返回路径到分数的映射。

### `WikiRetriever.search`

[查看源码](../llm_wiki_bench/wiki_retriever.py#L170) · `40` 行函数体（含已有文档）

【候选检索】刷新索引后按 layer、directory 和模式过滤、排序，折叠内容相同的来源副本，再可选重排并截到 limit。details 也包含来源页；exact_then_bm25 优先完整实体匹配，不是语义搜索。

### `WikiRetriever.tree`

[查看源码](../llm_wiki_bench/wiki_retriever.py#L212) · `10` 行函数体（含已有文档）

【导航入口】返回所有可见实际目录、描述、直接页面数、根页及解析错误；不是把所有子目录页面全文塞进上下文，需 wiki_read 分页浏览。

### `WikiRetriever.wiki_map`

[查看源码](../llm_wiki_bench/wiki_retriever.py#L224) · `4` 行函数体（含已有文档）

【兼容输出】把实时 tree 序列化为 JSON 字符串；不使用旧 index.md 概览挑选主题。

### `WikiRetriever.read`

[查看源码](../llm_wiki_bench/wiki_retriever.py#L231) · `74` 行函数体（含已有文档）

【目录／页面读取】一次处理 1..15 个路径；目录按页数分页，文件按选定 view/section 的字符分页，并返回整页 revision、链接和来源导航。facts 视图含历史事实；本函数的窗口坐标不能直接当原文引文坐标。

### `WikiRetriever._source_refs`

[查看源码](../llm_wiki_bench/wiki_retriever.py#L308) · `23` 行函数体（含已有文档）

【来源导航／兼容写盘】只根据明确 source_article 或来源路径链接获取原文引用；旧 sources/articles 页会被即时归档为不可变快照，所以旧页读取可能写 sources/versions。不会凭标题相似猜来源。

### `WikiRetriever.execute_tool`

[查看源码](../llm_wiki_bench/wiki_retriever.py#L334) · `26` 行函数体（含已有文档）

【工具分发】把模型工具名和参数路由到真实 Python 方法，将结果或捕获的 error 序列化成 JSON。不会执行模型生成的 Python 代码；entity_lookup 强制 exact 模式。

### `tool_schema`

[查看源码](../llm_wiki_bench/wiki_retriever.py#L363) · `3` 行函数体（含已有文档）

【协议描述】组装 OpenAI 风格的函数名、描述、参数 JSON Schema；它向模型说明如何请求，真正执行和硬校验由本地分发器完成。

## wiki_store.py

### `digest`

[查看源码](../llm_wiki_bench/wiki_store.py#L34) · `2` 行函数体（含已有文档）

【确定性基础】计算 UTF-8 文本的 SHA-256；用于来源身份、内容版本、页面 revision 和事实 ID，哈希相等不等于实体语义相同。

### `unwrap_markdown`

[查看源码](../llm_wiki_bench/wiki_store.py#L39) · `3` 行函数体（含已有文档）

【旧格式兼容】仅当整篇文本被同一 Markdown 围栏包住时解包；返回正文，否则原样返回。

### `frontmatter`

[查看源码](../llm_wiki_bench/wiki_store.py#L45) · `9` 行函数体（含已有文档）

【确定性解析】解开旧整页围栏后拆分 YAML 元数据和正文；要求元数据是字典，日期保留字符串，格式错误向上传递。

### `safe_path`

[查看源码](../llm_wiki_bench/wiki_store.py#L57) · `9` 行函数体（含已有文档）

【路径边界】把相对路径解析到 wiki 根目录内；拒绝绝对路径、父目录、隐藏组件及逃逸的符号链接，返回 Path。

### `atomic_write`

[查看源码](../llm_wiki_bench/wiki_store.py#L69) · `12` 行函数体（含已有文档）

【写盘】同目录临时文件写入并 fsync 后用 replace 替换目标；保证单文件提交，不提供多页面事务。

### `locked`

[查看源码](../llm_wiki_bench/wiki_store.py#L85) · `8` 行函数体（含已有文档）

【并发控制】用根目录下的 fcntl 文件锁串行化写入；yield 期间持锁，退出时释放，适用于 Unix。

### `SourceStore.__init__`

[查看源码](../llm_wiki_bench/wiki_store.py#L97) · `2` 行函数体（含已有文档）

【初始化】保存原文仓库根路径；本身不读写文件，也不调用模型。

### `SourceStore.archive`

[查看源码](../llm_wiki_bench/wiki_store.py#L102) · `17` 行函数体（含已有文档）

【原文归档】输入明确来源 identity 和完整 text；分别哈希为 source_id/version_id，保存不可变 Markdown 与 JSON 元数据，返回来源记录。已有同版本文本不符时拒绝写入。

### `SourceStore.get`

[查看源码](../llm_wiki_bench/wiki_store.py#L121) · `15` 行函数体（含已有文档）

【原文读取】按两个 64 位十六进制 ID 找快照；重算文本与身份哈希并检查长度，返回 (元数据, 全文)，篡改或缺失时失败。

### `SourceStore.read`

[查看源码](../llm_wiki_bench/wiki_store.py#L138) · `17` 行函数体（含已有文档）

【原文窗口】返回 Unicode 字符区间 [start,end)、文本、段落范围和 next_start；默认 6000、最多 12000 字符，end 参数仅 Python 接口支持。

### `SourceStore.validate_citation`

[查看源码](../llm_wiki_bench/wiki_store.py#L158) · `30` 行函数体（含已有文档）

【引用校验与纠偏】检查来源完整性和区间，优先逐字匹配；位置不准时搜索引文，还容忍空白、大小写及尾标点差异，返回对齐后的原文切片。多处匹配可能取首处；不判断引文是否支持 statement。

### `fact_state`

[查看源码](../llm_wiki_bench/wiki_store.py#L191) · `9` 行函数体（含已有文档）

【事实解析】从 Markdown 的 wiki-facts JSON 块取 facts/links；旧页无块返回空集合，损坏或多个块报错。

### `FactStore.__init__`

[查看源码](../llm_wiki_bench/wiki_store.py#L204) · `3` 行函数体（含已有文档）

【初始化】绑定 wiki 路径及 SourceStore，供事实写入时核对每一条原文引用。

### `FactStore.apply`

[查看源码](../llm_wiki_bench/wiki_store.py#L211) · `76` 行函数体（含已有文档）

【事实提交】接收路径、预期 revision、实体 identity、事实和链接；加锁后校验版本、身份、引文与目标存在性，再原子更新受控 JSON 块。相同限定陈述合并证据，不同陈述新增；旧正文保留。返回新 revision、全部 fact_ids 和关联页；不会自动修复关联页，也不做语义去重。

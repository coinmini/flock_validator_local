# 合并 new_official 目录指南

## Context

当收到新版本的 `new_official/` 目录时，需要将其中的代码合并到仓库根目录，同时保留根目录已有的本地验证功能。

## 仓库结构概览

```
FLock-validator/
├── local_validate.py          # 本地验证 CLI 入口
├── run.py                     # 线上验证入口
├── environment_entrypoint.py  # conda 环境入口
├── requirements.txt
├── configs/                   # 配置文件
├── validator/
│   ├── api.py                 # FLock API 客户端
│   ├── config.py              # 全局配置
│   ├── conda.py               # conda 环境管理
│   ├── utils.py               # 工具函数
│   ├── validation_runner.py   # 验证运行器
│   └── modules/
│       ├── llm_judge/
│       │   ├── __init__.py    # 核心模块（含线上+本地验证）
│       │   ├── prompt.py      # 评估 prompt 模板
│       │   ├── template.py    # 对话模板
│       │   └── utils.py       # 辅助工具
│       ├── lora/
│       └── rl/
```

## 合并步骤

### 1. 备份当前代码

```bash
git stash  # 或确保当前工作区干净
cp -r validator/modules/llm_judge/  /tmp/llm_judge_backup/
```

### 2. 对比差异，确定变更范围

```bash
diff validator/modules/llm_judge/__init__.py new_official/validator/modules/llm_judge/__init__.py
diff validator/modules/llm_judge/prompt.py new_official/validator/modules/llm_judge/prompt.py
diff local_validate.py new_official/local_validate.py
diff run.py new_official/run.py
```

重点关注：
- `new_official/` 新增了哪些功能或类
- `new_official/` 删除或重构了哪些接口
- 配置格式是否有变化

### 3. 合并策略

**原则：以 `new_official/` 为基础，补回本地验证功能。**

| 文件 | 策略 |
|------|------|
| `validator/modules/llm_judge/__init__.py` | 以 new_official 版本为基础，补回本地验证相关方法 |
| `validator/modules/llm_judge/prompt.py` | 通常直接用 new_official 版本 |
| `local_validate.py` | 保留现有版本，按需调整参数适配新接口 |
| `run.py` | 用 new_official 版本覆盖 |
| 其他文件 (`api.py`, `config.py`, etc.) | 用 new_official 版本覆盖 |
| `requirements.txt` | 用 new_official 版本，确认本地验证的依赖没丢 |

### 4. `__init__.py` 合并要点（核心）

需要保留的本地验证功能：

- **`skip_llm_eval`** — `__init__` 中的标志，允许跳过 LLM 评估
- **`_load_local_model()`** — 从磁盘加载模型（含 LoRA 支持、参数量检查）
- **`_generate_response_batched()`** — 批量生成响应的 generator
- **`_parse_jsonl_conversations()`** — 解析本地 JSONL 验证文件
- **`_build_conversation_result()`** — 构建单条对话结果
- **`validate_local()`** — 本地验证主入口方法
- **`cleanup()`** — 清理模型和 tokenizer 释放显存
- **Per-model client routing** — `_create_openai_client()`, `_initialize_model_clients()`, `_get_client_for_model()`, `_adapt_messages_for_model()`

### 5. 覆盖非核心文件

```bash
cp new_official/run.py run.py
cp new_official/environment_entrypoint.py environment_entrypoint.py
cp new_official/requirements.txt requirements.txt
cp new_official/README.md README.md
cp -r new_official/configs/ configs/
cp new_official/validator/api.py validator/api.py
cp new_official/validator/config.py validator/config.py

# llm_judge 模块中直接覆盖的文件
cp new_official/validator/modules/llm_judge/prompt.py validator/modules/llm_judge/prompt.py
cp new_official/validator/modules/llm_judge/environment.yml validator/modules/llm_judge/environment.yml
cp new_official/validator/modules/llm_judge/README.md validator/modules/llm_judge/README.md
```

### 6. 检查新增文件

```bash
diff -rq validator/ new_official/validator/ | grep "Only in new_official"
# 如有新文件需要复制过来，例如：
# cp new_official/validator/modules/llm_judge/constant.py validator/modules/llm_judge/constant.py
```

### 7. 验证

```bash
python -m py_compile validator/modules/llm_judge/__init__.py
python -m py_compile validator/modules/llm_judge/prompt.py
python -m py_compile local_validate.py

# 本地验证测试
python local_validate.py \
  --model-path /path/to/model \
  --validation-file validator/modules/llm_judge/test_data/final_validation_set.jsonl \
  --no-eval-with-llm
```

### 8. 清理并提交

```bash
rm -rf new_official/
git add -A
git commit -m "feat: merge new_official updates, preserve local validation"
```

## 合并历史记录

### 2026-03-27 合并 FLock-validator 官方更新

**官方新增/变更：**
- 新文件 `constant.py`：`SUPPORTED_BASE_MODELS` 白名单（Qwen3.5 系列）
- `_load_model()` 增加 SUPPORTED_BASE_MODELS 校验
- `_generate_response()` 改用 `tokenizer.apply_chat_template()`（替代自定义模板），支持 tools 参数、`enable_thinking=False`
- 生成参数调整：`gen_temperature` 默认 0.7、`top_p=0.8`、`top_k=20`
- `_load_jsonl_conversations()` 新增 `function_call`/`observation` 角色处理（转 OpenAI tool_calls 格式），使用 Jinja2 `template_str` 渲染 function_call reference
- 新方法：`_resolve_eval_models()`、`_parse_model_name_to_params()`、`_format_single_conversation()`、`_construct_evaluation_prompt()`、`_parse_llm_response()`、`_evaluate_single_conversation()`
- `_call_gpt()` 增加 kimi-k2.5 特殊温度处理
- `validate()` 重写为两阶段（生成 → ThreadPoolExecutor 并行评估）
- `cleanup()` 增强：清理 HuggingFace 模型和 tokenizer
- `prompt.py` 新增 `template_str`（Jinja2 模板）和 `prompt_id=3`（function_call 评估）
- `environment.yml` 依赖升级：transformers 4.49→5.3, huggingface-hub 0.29→1.5, peft→0.18.1
- Import 新增：`jinja2.Environment`, `template_str`, `SUPPORTED_BASE_MODELS`

**本地保留：**
- `skip_llm_eval` 参数、`_NO_SYSTEM_ROLE_MODELS`
- 多 provider 客户端系统：`_create_openai_client()`, `_initialize_model_clients()`, `_get_client_for_model()`, `_adapt_messages_for_model()`
- `_load_local_model()`, `_generate_response_batched()`, `_parse_jsonl_conversations()`, `_build_conversation_result()`, `validate_local()`
- `_call_gpt()` 中的多 client 路由和 model_name_map
- `_construct_conversation_template()` 保留（本地验证使用 template_dict）
- `cleanup()` 中的 model_clients 清理
- `template.py` 保留（本地验证依赖）

---

## 常见问题

**Q: new_official 改了 LLMJudgeConfig 的字段怎么办？**
A: 以 new_official 为准，确保 `validate_local()` 用到的 config 字段仍然存在。

**Q: new_official 改了 `_call_gpt()` 签名怎么办？**
A: 更新为新版本，检查本地验证代码中调用处是否需要适配。

**Q: prompt.py 新增了 prompt_id 怎么办？**
A: 直接用 new_official 版本，`get_prompt()` 路由逻辑可能需要更新。

**Q: 新增了其他 module（非 llm_judge）怎么办？**
A: 直接复制过来，不涉及本地验证的合并问题。

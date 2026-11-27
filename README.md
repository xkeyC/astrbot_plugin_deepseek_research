# astrbot_plugin_deepseek_research

AstrBotX（Codex runner）插件：给主 agent 一个 `deepseek_research` 工具。调用后会起一个**受限的 Codex 子代理**，用 DeepSeek 官方的 [Search / Fetch API](https://api-docs.deepseek.com/guides/search/) 做联网调研，返回带引用的报告。

## 工具

`deepseek_research(keywords, instruction?, time_range?, priority_urls?)`

| 参数 | 说明 |
| --- | --- |
| `keywords` | 调研主题或关键字 |
| `instruction` | 报告要回答什么、侧重点、格式 |
| `time_range` | `7d` / `2w` / `3m` / `1y`（到今天为止），`2025` 或 `2025-06-01`（从该日起），`A..B`（任意一端可留空）。默认作用到每次搜索，子代理需要时可以覆盖 |
| `priority_urls` | 子代理先读的网址 |

这是同步工具，直接返回报告文本。报告最长 `result_max_chars` 字，末尾附用时、搜索次数和材料位置。在 code mode 里，主 agent 的 `exec` 到 yield 时限会把控制权交回，之后用 `wait` 取结果。

## 子代理的限制

子代理是同一个 Codex engine 上的一个 ephemeral 线程（不落 rollout），线程配置为：

- `no_environment`，并关闭 `shell_tool`、`view_image`、`image_generation`、`web_search`、`memories`、`apps`、`agents`；
- `model_tool_mode = direct`（不走 code mode）。

模型能用的只有以下工具：

- `research.deepseek_search`：Search API，支持 `site:`、`lang:`、`after:`、`before:` 等运算符，查询加运算符最长 128 字。`need_content` 时把结果正文存进沙箱；
- `research.deepseek_fetch`：Fetch API，长网页按窗口分段读取，同一页只抓一次；
- `research.tmp_read` / `tmp_list`：读取、列出沙箱里 `/tmp/ds_research/<task_id>/` 下的文件；
- `research.apply_patch`：Codex 风格的 patch（`*** Add File` / `*** Update File` / `*** Move to` / `*** Delete File` / `@@` hunk），用来新建、修改、删除任务目录里的文件。解析和 hunk 匹配复用 AstrBot 内置 `apply_patch` 的实现。整个 patch 先全部算完再落盘：任何一个 hunk 匹配失败，或者任何一个路径越界，都一个文件也不写。

路径先在宿主侧检查（拒绝 `..` 和目录外的绝对路径），落盘时再在沙箱内解析符号链接后检查一次，越界直接拒绝。

在 codex 348075fad 上用 Responses 和 chat 两种 wire 实测过：模型收到的 tools 只有上面 5 个。`view_image` 即使在 `no_environment` 下也要单独关掉才会消失。

沙箱文件通过 Shipyard 的 Python 执行器读写，因为它的文件 API 只能访问 `/workspace`。执行的代码由插件固定写死，请求内容用 base64 传入，模型拿不到执行能力。会话没有启用 sandbox 运行时的话，不提供文件工具（`tmp_*`、`apply_patch`），网页只在内存里分段读。

材料目录：`report.md`（完整报告）、`search/NN.md`（第 N 次搜索的结果正文）、`pages/NN-<host>.md`（抓取的网页）。主 agent 可以用自己的沙箱工具读这些文件。

## 时间预算

预算按插件的 `time_budget_s`（默认 600 秒，不少于 30 秒）计算，只看这一次调用所在的运行：

- **这次运行提供 `astrbot_tool_task`**（目前是 Codex 主聊天请求）：`tool_call_timeout` 只是到时提醒主 agent。工具转到后台继续跑，主 agent 用 `astrbot_tool_task` 选择等待、查看状态或取消。预算就是 `time_budget_s`。
- **其他运行**（较旧的 AstrBot、别的插件在 `tool_loop_agent` 里调用、后台任务）：工具会在 `tool_call_timeout` 时被取消，报告会丢失。这时预算是 `min(time_budget_s, tool_call_timeout - 15)`。

插件在运行时从工具的 run context 读 `tool_tasks` 和 `tool_call_timeout` 来判断，所以工具是用 `FunctionTool` 子类注册的，没有用 `llm_tool` 装饰器。

预算用掉 75% 后，插件会向子线程 steer 一条"停止调用工具、立即写报告"的消息，之后的工具调用一律拒绝。到 100% 还没写完就 interrupt，只返回已有的部分结果（执行过的搜索和读过的网页）。工具调用次数超过 `max_tool_calls` 后同样拒绝。

无论被取消（主 agent 取消工具任务、用户 /stop，或在会取消的运行里超时）还是出错，子线程都会被 interrupt 并卸载。

在 code mode 里调用（`exec`，或开源模型模式下的 `run_code`）时，程序到 yield 时限就先把控制权交回主 agent，工具调用在后台继续；主 agent 之后用 `wait` 取结果。

## 模型

`subagent_model_source` 选子代理跑在哪个模型上：

| 取值 | provider | 模型 |
| --- | --- | --- |
| `auto`（默认） | `subagent_provider`；留空时优先用 `base_url` 含 `api.deepseek.com` 的 provider，没有的话用 runner 当前的 provider 和模型 | `subagent_model` |
| `deepseek` | `base_url` 含 `api.deepseek.com` 的 provider | `subagent_model` |
| `local` | `local_provider` 指定的本地 provider，例如局域网里的 [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server) | `local_model` |

provider 都填 Codex runner 里 WebUI 管理的 provider ID（或名称）；模型留空时取该 provider 列出的第一个模型。推理强度统一用 `subagent_reasoning_effort`（留空用模型的默认值）。Search / Fetch 不管子代理用哪个模型，始终走 DeepSeek 官方 API。

子代理和主会话共用同一个 Codex engine（用同一份 runner 配置），所以不会触发 engine 重启。

## DeepSeek 搜索的已知覆盖范围（2026-11 实测）

| 站点 | 搜索 | 抓取 |
| --- | --- | --- |
| YouTube / Reddit / X(Twitter) / Medium | 无（`site:` 查询结果为空） | 拒绝 |
| 知乎 / Hacker News | 有 | 拒绝 |
| 百度贴吧 | 有，`need_content` 能带回帖子正文（索引快照） | `/p/` 帖子页失败（反爬） |
| B 站 / GitHub / 微博 / 维基百科 / Stack Overflow | 有 | B 站、GitHub 可以；微博多为登录壳页 |

子代理的提示词里写了这些限制，遇到时会去找转载、新闻报道或其他来源，并在报告里说明。

## 计费

Search 和 Fetch 各为每 1000 次成功请求 $0.75，空结果和失败不计费。子代理本身的 token 按所选 provider 计费。

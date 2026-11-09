# SignalLens AI ◈

**Conversational Signal Intelligence · 对话信号分析**

A multi-agent LLM system that reads a conversation and pulls out the signals underneath the words: **emotion, intent, communication style, consistency and risk**.

🔗 **Live demo:** [https://signallens-ai.streamlit.app  *(replace with your deployed URL)*](https://signallens1.streamlit.app/) **

---

## Why it matters

What people say on the surface often differs from what they mean. Brand managers reading customer DMs, community teams moderating comments and individuals working through difficult chats all face the same problem: **unstructured conversation is hard to read objectively**. SignalLens turns a conversation into a structured, explainable read on that gap.

## Architecture: a panel of specialist agents

```
             Conversation (text / screenshots)
                           │
   ┌───────────┬───────────┼───────────┬───────────┐
   ▼           ▼           ▼           ▼           ▼
 Risk /     Style &     Intent     Boundary   Consistency
 Red Flag   Emotional  Inference   & Respect   & Pattern
   └───────────┴───────────┼───────────┴───────────┘
                           ▼
              Signal Synthesis (Chair Agent)
           weighted scoring · risk flags · next steps
                           ▼
                   Visual dashboard
```

| Agent | What it analyses |
|---|---|
| Red Flag Agent | Risk signals such as manipulation, pressure or evasiveness, rated by severity (1–5) |
| Compatibility Agent | Communication style and emotional tone |
| Intent Inference Agent | The likely intent behind messages |
| Boundary Respect Agent | Whether stated boundaries and preferences are respected |
| Pattern Consistency Agent | Hot-and-cold patterns and consistency of effort over time |
| **Signal Synthesis (Chair)** | Combines the specialists' outputs into one weighted verdict |

**Methodology highlights**
- **Decomposition:** each agent has a narrow role and returns a strict JSON schema, which makes outputs auditable and less prone to hallucination.
- **User-tunable weighting:** users set dimension weights, risk sensitivity and custom "deal-breakers", so the analysis adapts to their context.
- **Multimodal input:** conversations can be pasted as text or uploaded as screenshots (GPT-4o vision), and an image authenticity check is included.
- **Explainable output:** scores come with evidence, flags and suggested next actions, not just a single label.
- **Responsible AI:** decision support only, with no data stored and guardrails against misuse.

## Example use cases

- Reading the emotional tone and intent of one-to-one customer or community conversations
- Spotting manipulative, high-pressure or inconsistent messaging patterns
- Personal communication: making sense of mixed signals in chats (the original MVP scenario)

## 中文简介

SignalLens AI 是一个基于 **多 Agent 架构** 的对话信号分析系统。输入一段对话（文本或截图），5 个专业 Agent 分别从 **风险信号、沟通风格与情绪、意图推断、边界尊重、行为一致性** 五个维度并行分析，再由 Chair Agent 加权汇总，输出结构化评分、风险提示和行动建议，并配有可视化仪表盘。

方法论亮点：角色拆分 + 严格 JSON 输出（可解释、可审计）、用户可调权重与风险敏感度、多模态输入（GPT-4o Vision）、Responsible AI 设计（不存储数据）。

## How to use

1. Open the live demo.
2. In the sidebar, paste **your own OpenAI API key** ([get one here](https://platform.openai.com/api-keys)). The key stays in your session only.
3. Paste a conversation or upload screenshots, choose a scenario and goal, then run the analysis.

使用说明：在侧边栏填入你自己的 OpenAI API Key 即可体验。

## Tech stack

Python · Streamlit · OpenAI API (GPT-4o, vision) · Plotly · Pandas · Pillow

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

*Built as a group project for MSIN0231 Machine Learning for Business, UCL.*

---
title: "主动提示与主动提问：agent 在不确定性下猜测用户意图"
date: 2026-04-04 23:00:00 +0800
categories: [Research, LLM Agents]
tags: [paper-reading, proactive-agent, collegeAI]
math: true  # 论文分享通常有公式，记得开这个
---

> **Note:** This was originally prepared for a group meeting presentation at collegeAI. Due to a course conflict, I am hosting the digital version here.


## 主线框架

LLM agent 面对不确定性时有两类策略：

- **主动提问（Proactive Asking）**：主动向用户提问，直接消除不确定性，获取缺失信息
- **主动提示（Proactive Prompting）**：利用已有环境信息推断用户意图，在合适时机主动介入

两者的根本问题相同：**agent 如何识别"自己不知道什么"，并据此作出恰当响应**。区别仅在于解决手段——提示是利用信息，提问是获取信息。

---

## 通用 Pipeline

四篇论文共享一套抽象流程：

```
环境/查询输入
    ↓
[触发判断] → 是否需要主动行动？（PRISM: pneed/paccept 门控；GPS: DAG 结构检测；PIR: PE 熵阈值）
    ↓ 是
[冷启动 SFT] → 强模型生成合成轨迹，学习行动格式
    ↓
[策略优化 GRPO] → 复合奖励，平衡有效性与效率（GPS / PIR 使用；PRISM 不使用）
    ↓
[用户模拟器] → 闭环训练与测试，LLM 扮演用户给出反馈
    ↓
输出：正确且高效的主动行为
```

---

| 维度         | ProactiveBench       | PRISM                         | GPS                           | PIR                            |
| ------------ | -------------------- | ----------------------------- | ----------------------------- | ------------------------------ |
| 不确定性来源 | 环境事件流用户意图   | 环境事件流用户意图            | RAG 文档条件逻辑              | 推理链中间步骤                 |
| 触发机制     | 新事件到达时尝试预测 | pneed/paccept 双阈值门控      | 查询欠规范时                  | 推理步骤 PE 峰值               |
| 数据来源     | 真实活动 + LLM Gym   | DeepSeek-R1 traces + RDC 过滤 | DeepSeek-R1 生成 + 验证器过滤 | DeepSeek-R1 轨迹 + PE 检测插入 |
| SFT          | ✓ 标准 SFT           | ✓ RDC-SFT                     | ✓ Reasoner 冷启动             | ✓ 冷启动                       |
| GRPO         | ✗                    | ✗                             | ✓ 混合奖励                    | ✓ US-GRPO                      |
| 核心创新     | 数据 + benchmark     | 决策理论门控 + 选择性慢推理   | DAG 逻辑完备性 + 动态剪枝     | PE 驱动触发 + 推理内嵌澄清     |

下面依次介绍每篇论文。

---

## 一、主动提示

### 1.1 ProactiveBench（Lu et al., 2024 / ICLR 2025）

**定位**：提出任务定义，构建 benchmark，建立 SFT baseline

#### 任务定义

传统 agent 是被动响应的：用户发出指令，agent 执行。ProactiveBench 提出的问题是：agent 能否在用户开口之前，仅凭环境观测预测用户的潜在需求并主动提出？

形式化：agent 根据当前环境事件 $E_t$、用户活动 $A_t$、环境状态 $S_t$ 输出预测 $P_t$，用户接受/拒绝：

$$R_t = \begin{cases} 1 & (P_t \neq \emptyset \land \text{用户接受}) \;\text{或}\; (P_t = \emptyset \land N_t = 0) \\ 0 & \text{otherwise} \end{cases}$$

其中 $N_t=1$ 表示用户此刻确实需要帮助。目标是最大化期望接受率。

---

#### 数据构建

**真实数据采集**：通过 ActivityWatcher 软件收集键盘/鼠标/浏览器/剪贴板等用户活动，转化为自然语言事件描述。

**LLM Gym 模拟**（三个组件）：
- **Environment Gym**：GPT-4o 基于真实数据生成场景（coding/writing/daily life），维护实体状态，逐步生成连贯事件序列
- **Proactive Agent**：接收事件流，预测用户可能的任务
- **User Agent**：GPT-4o 扮演用户，判断是否接受 agent 的预测

**人工标注奖励模型数据**：用 9 个不同 LLM 对每个事件生成多样预测，由 3 名标注者独立标注 accept/reject/reject-all，多数投票决定最终标签，共 1,760 条，标注者一致性 91.67%。

最终 **ProactiveBench**：6,790 条训练事件 + 233 条真实测试事件。

---

#### 训练方法

**Reward Model**：基于 LLaMA-3.1-8B-Instruct 微调，以 1,640 条人工标注数据训练，用于自动评估 agent 预测是否被接受，对齐率 91.80% F1。

**Agent Model（SFT）**：在 ProactiveBench 6,790 条训练数据上做标准 SFT，目标是让模型学会依据事件历史输出合理的预测。底座为 LLaMA-3.1-8B-Instruct 与 Qwen2-7B-Instruct，3 epochs，batch=32，lr=1e-5，8×A100 约 2 小时。无 GRPO。

---

#### 实验结果

| 模型                   | Recall  | Precision | False-Alarm | F1         |
| ---------------------- | ------- | --------- | ----------- | ---------- |
| GPT-4o                 | 98.11%  | 48.15%    | 51.85%      | 64.60%     |
| LLaMA-3.1-8B（原版）   | 98.86%  | 38.16%    | 61.84%      | 55.06%     |
| LLaMA-3.1-8B-Proactive | 99.06%  | 49.76%    | 50.24%      | **66.25%** |
| Qwen2-7B-Proactive     | 100.00% | 49.78%    | 50.22%      | **66.47%** |

**核心发现**：所有模型 Recall 均接近 100%——模型学会了"凡事都提"，Precision 和 False-Alarm 极差。SFT 让模型更会提示，但无法控制"何时该沉默"。这正是 PRISM 要解决的问题。

---

### 1.2 PRISM（Fu et al., 2026 / ICLR 2026）

**定位**：引入决策理论框架，从根本上解决"过度主动"问题

#### 核心洞察：把"能不能"和"该不该"分开

ProactiveBench 的失败根源在于：模型将"我能给出一个帮助"与"我应该说出来"混为一谈。PRISM 将介入决策分解为两个独立概率：

- $p_\text{need}$：用户在当前情境下是否真的需要帮助
- $p_\text{accept}$：若 agent 介入，用户是否会接受这个提示

---

**直觉例子**：

> 用户正在改 README 文件的标题（"Project X" → "Project X v2"）。这是个无关紧要的编辑，$p_\text{need}$ 很低。即使 agent 有信心能给出"要不要创建 PR？"这样的合理建议，$p_\text{accept}$ 也达不到因 $p_\text{need}$ 低而自动升高的阈值——agent 保持沉默，用户不受打扰。
>
> 反之，用户运行 Python 报了 `ModuleNotFoundError`，然后反复翻 `requirements.txt` 却没做任何修改。$p_\text{need}$ 很高，阈值自动降低——即使 agent 不是百分百确定用户会接受，也会主动提示"是否需要安装缺失的依赖？"

---

#### 决策门控机制

Bayes 最优门控，在 asymmetric cost 下推导：

$$\text{INTERVENE} \iff p_\text{accept} \geq \tau(p_\text{need}) \triangleq \frac{C_\text{FA}}{C_\text{FA} + p_\text{need} \cdot C_\text{FN}}$$

$C_\text{FA}$ 为虚警代价（打扰用户），$C_\text{FN}$ 为漏检代价（错过帮助机会）。阈值 $\tau$ 的性质：
- $p_\text{need}$ 升高 → $\tau$ 降低（需求越确定，越容忍低接受率也介入）
- $C_\text{FA}$ 升高 → $\tau$ 升高（打扰代价越高，越保守）

这个阈值是可调的"旋钮"，无需重新训练模型即可适应不同用户/场景。

---

#### 双进程推理（festina lente：欲速则不达）

并非所有情况都需要深度推理。PRISM 引入 slow mode margin $\delta_\text{slow}$：

- **Fast mode**：快速估计 $( p_{\text{need}}^F, p_{\text{accept}}^F )$
- **Slow mode**：触发深度推理的条件为：
  $$|p_{\text{accept}}^F - \tau(p_{\text{need}}^F)| \leq \delta_{\text{slow}}$$ 
  即当结果靠近决策边界（最不确定时）才执行。
- **Efficiency**：在实验最优参数 $\delta=0.1$ 下，仅约 **11%** 的样本进入 Slow mode。

---

#### 数据来源：RDC 蒸馏

**问题**：如何用强模型（teacher）的知识训练弱模型（student），同时避免 teacher 的错误判断污染训练？

**RDC（Decision-Consistent Curation）**的思路是：只挑 teacher "说得对且知道自己说得对"的样本。排序分数：

$$R_\text{DC} = y_\text{accept} - (q_\text{need} - y_\text{need})^2 - \mathbf{1}[y_\text{need}^{(\text{pred})}=1](q_\text{accept} - y_\text{accept})^2$$

---

三项含义：
- $y_\text{accept}$：这条样本最终被接受（正向激励，接受的样本排名更高）
- $-(q_\text{need}-y_\text{need})^2$：teacher 对"是否需要"的概率估计与真值越接近越好
- 在 teacher 判断"确实需要"的前提下，对"是否会接受"的概率估计也要准确：
  $$\mathbf{1}[y_\text{need}^{(\text{pred})}=1](q_\text{accept} - y_\text{accept})^2$$

直觉：RDC 在过滤 teacher 的"自知之明"——teacher 在这条样本上既判断对了结果，又对自己的 need/accept 概率有准确的自我评估，才把这条样本交给 student 学习。最终只保留排名靠前的 **1,800 条**（原训练集的 1/3 以下）。

---

#### 训练目标

$$\mathcal{L} = \mathcal{L}_\text{need} + \mathcal{L}_\text{acc} + \mathcal{L}_\text{burden}$$

- $\mathcal{L}_{\text{need}}$：监督 $p_{\text{need}}$ 估计
- $\mathcal{L}_{\text{acc}}$：使用逆概率评分（IPS）处理选择偏差，监督 $p_{\text{accept}}$ 估计（只有实际介入的样本才有 $y_{\text{accept}}$ 标签，直接训练会有偏）
- $\mathcal{L}_{\text{burden}}$：惩罚虚警行为和过多 Slow mode 计算

不使用 GRPO：任务本质是校准好两个概率值的分类/回归问题，RDC 过滤后的 SFT 已足够；引入 GRPO 反而可能破坏概率校准（消融验证了这一点）。

---

#### 实验结果

| 模型                          | Recall     | Precision  | False-Alarm | F1         |
| ----------------------------- | ---------- | ---------- | ----------- | ---------- |
| Qwen2-7B-Proactive（原 SOTA） | 100.00%    | 49.78%     | 50.22%      | 66.47%     |
| DeepSeek-R1（teacher）        | 98.12%     | 72.35%     | 27.64%      | 83.28%     |
| **Qwen3-8B-PRISM（ours）**    | **98.88%** | **77.05%** | **22.94%**  | **86.61%** |

student 在 Precision 和 False-Alarm 上显著超越 teacher（p < 0.001），用不到 1/3 的数据完成训练。人工评估亦验证：F1 84.85% vs teacher 的 82.05%。

---

#### 消融：逐层拆解哪个设计真正起作用

**消融一：门控信号的贡献**

| 配置                                         | Recall | Precision  | False-Alarm | F1         |
| -------------------------------------------- | ------ | ---------- | ----------- | ---------- |
| 仅 $p_\text{accept}$（令 $p_\text{need}=1$） | 99.95% | 46.20%     | **62.50%**  | 63.19%     |
| 仅 $p_\text{need}$                           | 99.15% | 69.50%     | 29.10%      | 81.72%     |
| 双信号（未校准）                             | 98.80% | 74.77%     | 25.23%      | 85.12%     |
| **双信号 + 温度缩放**                        | 98.88% | **77.05%** | **22.94%**  | **86.61%** |

---

- 单用 $p_\text{accept}$：退化为"只要我觉得用户会接受就说"，完全不管用户是否真的需要，虚警爆表（62.5%）——等同于 ProactiveBench 的失败模式
- 加入 $p_\text{need}$：引入"用户是否需要"的前置判断，虚警立刻降到 29%，是最大的单步改进
- 再加后验温度缩放校准：让概率值更准确地反映真实频率，虚警进一步降到 22.9%，AUDBC 提升 2.09

---

**消融二：推理策略效率**

| 策略                       | F1         | AUDBC      | Tokens | P95延迟   |
| -------------------------- | ---------- | ---------- | ------ | --------- |
| Fast-only                  | 83.09%     | 79.26%     | 510    | 176ms     |
| Slow-only                  | 86.79%     | 81.43%     | 693    | 312ms     |
| **Slow-on-margin (δ=0.1)** | **88.15%** | **82.72%** | 541    | **196ms** |

Slow-on-margin 是 Pareto 最优：F1 超过 Slow-only，延迟仅比 Fast-only 多 20ms。δ 敏感性分析显示 δ=0.1 是"甜点"：再大 F1 开始下降（边界区域太宽，无关样本进入 Slow mode）。

---

**消融三：训练路径对比**

| 训练方式            | F1         | False-Alarm |
| ------------------- | ---------- | ----------- |
| 原始 SFT（未过滤）  | 76.09%     | 27.56%      |
| Weighted-SFT        | 80.59%     | 29.31%      |
| DFT（动态微调）     | 76.09%     | 27.56%      |
| **RDC-SFT（ours）** | **86.61%** | **22.49%**  |

Weighted-SFT 用未校准的 $(q_\text{need}, q_\text{accept})$ 作标量权重，在 acceptance/timing 噪声下反而膨胀了虚警。DFT 在 RDC-SFT 基础上二次训练，blunt 了 acceptance-aware 梯度，Recall 下降到 80.14%。结论：**数据质量和目标结构比后处理优化更重要**。

---

## 二、主动提问

### 2.1 GPS（Li et al., 2026 / ICLR 2026）

**定位**：用 DAG 显式表示文档中的条件逻辑，通过结构化遍历实现高效澄清

#### 核心问题

RAG 场景中文档往往包含形如"若 A 且 B 则 C，否则 D"的条件规则。用户查询通常欠规范（underspecified），缺少决定答案所需的关键条件。直接回答会因缺失条件而出错；让模型自由提问又可能遗漏关键分支或提出冗余问题。

GPS 的洞察：**条件逻辑本身就是一棵图，把它显式建出来，澄清过程就变成图上的遍历。**

---

#### DAG 推理结构

定义条件推理 DAG $\mathcal{G} = (\mathcal{N}, \mathcal{E})$：
- 非终端节点 $n_{c_i}$：一个需要澄清的条件变量（如"是否领取 income-related ESA"）
- 终端节点 $n_{a_m}$：最终答案
- 边 $e_{i,j} = (n_{c_i}, n_{c_j}, \nu)$：条件变量取值 $\nu$ 时的转移
- 单前驱 → AND 关系；多前驱 → OR 关系

**逻辑完备性（命题 1）**：对任意有限值函数 $g: \prod_i V_i \to A$，存在对应 DAG 使得每条根到叶路径恰好对应 $\mathbf{1}[g(\cdot)=a_m]$ 的 DNF 中的一个合取项。即 DAG 能表达任意布尔函数，不会遗漏任何条件组合。

**澄清效率**：动态遍历平均复杂度 $O(r)$，$r$ 为实际推理路径长度，远小于文档总条件数 $k$。节点共享还允许不同路径复用已澄清的条件。

---

#### 两阶段框架

**阶段一（Reasoning Stage）**：Reasoner LLM 读取用户查询 $q$ 和检索文档 $d$，输出 DAG 结构（JSON 格式的节点和边列表）。

**阶段二（Clarification Stage）**：Clarifier LLM 基于 DAG 与用户交互——

动态遍历算法维护候选澄清集 $U$（由入度为 0 或前驱已知的节点组成），按剩余路径深度期望选择最具信息量的条件提问，依据用户回答剪枝不一致分支，直至到达终端节点。

---

#### 数据来源

基础数据：**ConditionalQA**（包含条件逻辑的阅读理解集）——但仅 24.5%（550/2,247）为欠规范查询，远不够训练。

**条件路径引导数据合成**：
1. DeepSeek-R1 从文档生成欠规范问题，附带：
   - 缺失条件集 $C$ 及各条件的取值域
   - 条件路径集 $P = \{(v, a)\}$：完整条件赋值 → 唯一答案
2. 验证器 LLM 过滤：只保留"完整条件时答对、缺失条件时答错"的样本，确保每个缺失条件都是必要的
3. 最终训练集：Synthetic 2,060 条 + ConditionalQA 欠规范训练集，共 **3,250 条**

---

数据集示例结构（一条样本）：
```
query:    "Am I eligible for the disability premium?"
document: "If you get income-related ESA, you cannot get the premium.
           You can get it if you are under pension credit age, or
           getting DLA/PIP, or unable to work for at least a year."
missing_conditions: ["income-related ESA", "pension credit age", "DLA", "PIP", "work duration"]
conditional_paths:
  - {ESA: Yes} → NOT eligible
  - {ESA: No, under_pension_age: Yes} → eligible
  - {ESA: No, under_pension_age: No, DLA: Yes} → eligible
  ...
```

---

#### 训练方法

**Reasoner 冷启动 SFT**：在合成数据上 SFT，让 Reasoner 学习从 query + document 提取正确 DAG 格式。Clarifier LLM 在整个训练中固定不动，只训练 Reasoner。

**GRPO 强化学习**：对每个 (query, document) 采样 $h$ 个 DAG $\{o_i\}$，让固定的 Clarifier 基于每个 DAG 与 User Simulator 完成澄清交互，获得奖励后更新 Reasoner：

$$J_\text{GRPO}(\Theta) = \mathbb{E}\left[\frac{1}{h}\sum_{i=1}^h \left[\min\left(\frac{\pi_\theta(o_i|q)}{\pi_\theta^{\text{old}}(o_i|q)} A_i,\; \text{clip}(\cdot, 1\pm\epsilon) A_i\right) - \beta D_\text{KL}\right]\right]$$

**混合奖励函数**（设计逻辑逐层展开）：

$$r_i = r_{\text{acc},i} \cdot (r_{\text{eff},i} + r_{\eta,i})$$

乘法结构的关键：**$r_\text{acc}$ 作为门控**——只有澄清后回答正确，效率和结构奖励才生效。防止模型学到"提很少的问题但答错"或"DAG 结构很漂亮但跑偏"。

---

三项奖励的动机：

**有效性奖励 $r_\text{acc}$**（0/1）：回答是否正确。这是最基础的约束，没有正确性，其他都是空谈。

**效率奖励 $r_\text{eff}$**：

$$r_{\text{eff},i} = 1 - \alpha \cdot \frac{t_i}{t_\text{max}}, \quad \alpha=0.5$$

$t_i$ 为该 DAG 导致的澄清轮次，$t_\text{max}$ 为组内最大轮次。惩罚过多交互——用户不希望被反复追问。

---

**结构质量奖励 $r_\eta$**：这是 GPS 最独特的设计。直觉：一个好的 DAG 应该让每次澄清都有效地"缩小"结论空间，而不是分支来分支去最终汇回相同的结论。

通过前向概率传播（假设均匀分支）计算 DAG 的分支熵 $H_\text{graph}$ 和叶节点分布熵 $H_\text{leaf}$：

$$r_{\eta} = H_\text{leaf} / H_\text{graph}$$

- 好的 DAG（GPS 生成）：每次分支都引向不同结论，$H_\text{graph}$ 完全转化为 $H_\text{leaf}$，$r_\eta = 1$
- 差的 DAG（基线模型）：分支后又汇合，$H_\text{graph}$ 大但 $H_\text{leaf}$ 小，$r_\eta = 0.46$

这个奖励**不依赖澄清交互的结果**，只看 DAG 结构本身——是对 Reasoner 输出质量的直接约束。

---

训练配置：8×A800，LoRA（rank=64），batch=32，lr=3e-6，1 epoch。

#### 评估指标

- **Success Rate (SR)**：澄清后回答正确的比例（用 LLM evaluator 判断语义对齐）
- **Weighted Clarification Turns (WCT)**：

$$\text{WCT} = p_\text{success} \cdot \text{MCT}_\text{success} + p_\text{failed} \cdot T_\text{max}$$

对失败样本惩以 $T_\text{max}=10$，使 WCT 同时反映正确率和效率——只降低轮次但答错不算好，只答对但用了很多轮也不算好。

---

#### 实验结果

| 方法                 | Synthetic SR↑ | WCT↓     | ConditionalQA SR↑ | WCT↓     | ShARC SR↑ | WCT↓     |
| -------------------- | ------------- | -------- | ----------------- | -------- | --------- | -------- |
| Base Method          | 21.2          | 7.88     | 70.3              | 2.98     | 49.3      | 5.08     |
| ProCoT               | 42.5          | 6.07     | 71.6              | 2.95     | 62.6      | 4.06     |
| Clarify-DPO          | 59.2          | 4.67     | 72.0              | 3.52     | 78.5      | 2.93     |
| **GPS (Qwen2.5-7B)** | **60.2**      | **4.59** | **73.4**          | **2.91** | **79.3**  | **2.41** |
| **GPS (LLaMA-3-8B)** | **56.5**      | **5.02** | **74.6**          | **2.89** | 75.8      | 2.79     |

GPS 在 SR 和 WCT 上几乎全面最优，且对 ShARC（训练集未覆盖的域外数据集）也有强泛化。

---

### 2.2 PIR（Chen et al., 2026 / arXiv Jan 2026）

**定位**：在推理过程中动态触发澄清，解决推理模型的"盲目自我推理"问题

#### 核心问题

GPS 针对 RAG 场景，依赖文档的显式条件结构。PIR 面对的是更通用的场景：用户给出模糊指令（如"帮我修这个数据处理脚本"），推理模型没有文档可依，只能靠自身推理——但模型往往意识不到自己的推理缺乏必要前提，产生"blind self-thinking"：耗费大量 token 内部推理，最终给出和用户真实需求不符的答案。

**关键区别**：PIR 不依赖外部结构（DAG），而是在模型内部推理链中实时检测不确定区域，直接嵌入澄清步骤。

---

#### 数据来源：不确定性驱动的轨迹构建

**为什么用 Predictive Entropy（PE）**：推理模型在"aha moment"附近会表现出 PE 峰值（Yang et al., 2025），这恰好是模型对当前推理方向最不确定的时刻——也是最应该暂停并向用户确认的时刻。

对每条推理步骤 $s_j$ 计算归一化 PE：

$$\text{PE}_\text{norm}(s_j|x) = -\frac{1}{|s_j|}\sum_i \log p(z_i | s_{<j}, x)$$

选取 PE 最高的 top-k% 步骤作为澄清触发点，在触发点插入 GPT-4o-mini 生成的澄清问题及模拟用户回复，将原本线性的 CoT 转为交织轨迹：

$$y = \{t_1, (a_1, r_1), \ldots, t_m, (a_m, r_m), O\}$$

---

分析验证（Appendix）：数据量从 1k 增至 4k 时，触发句子的平均 PE 单调上升并趋于稳定，模板正确率从 0.15 升至 1.0——模型确实学到了"在高不确定处提问"，而非随机触发。

**SFT 冷启动数据**：4,000 条 Reasoning-While-Asking 数据集（DeepSeek R1 基于 Chinese Open Instruction Generalist 开放式问题生成的推理轨迹）

**RL 阶段数据**：CollabLLM 的三个多轮任务数据集：
- MathChat（4,054条）：含隐含假设的数学推理
- BigCodeBench-Chat（3,606条）：代码需求澄清与调试
- MediumDocEdit-Chat（4,254条）：多轮协作文档编辑

---

#### 训练方法

**阶段一：SFT 冷启动**（4×A100，3 epochs）

在 4,000 条交织轨迹上训练，目标函数覆盖整条序列——推理步骤、澄清问题、用户回复、最终答案全部参与训练，使模型学会推理-提问-融合反馈的平滑转换。

**阶段二：US-GRPO**（8×A100，5 epochs）

构建动态用户模拟器 $S$（Llama-3.1-8B-Instruct），预设特定用户 Intent $I$，与策略模型进行多轮交互生成轨迹。GRPO 梯度**只对策略输出 $(t_n, a_n)$ 更新，用户回复 $r_n$ 全部 mask**——不希望模型学习如何"操纵"用户给出有利回复。

---

**复合奖励（设计动机逐层展开）**：

$$R(y) = R_\text{output}(o,g) + R_\text{reason}(r,o,g)$$

**Output Reward**（基础正确性）：

$$R_\text{output}(o,g) = S_\text{base} \cdot \mathbf{1}(o=g)$$

最基础的约束，答对才算。

**Reasoning Reward**（过程质量）：

$$R_\text{reason}(r,o,g) = \mathbf{1}(o=g) \cdot \mathbf{1}_\text{ask}\left[S_\text{base} \cdot E(r) \cdot H_\text{LLM}(r)\right]$$

---

三个设计决策的动机：

1. **$\mathbf{1}(o=g)$ 门控**：Reasoning Reward 只在答对时激活。原因：防止模型学到"提了很多聪明的问题但最后仍然答错"也能获得奖励。澄清的价值必须体现在最终正确性上。

2. **强制至少一次澄清**：早期训练中模型可能退化为"从不提问，直接给答案"（因为答对了也有 Output Reward）。
  $$\mathbf{1}_{\text{ask}}$$ 
  确保模型在正确答案的条件下还需要完成至少一次有效澄清才能获得 Reasoning Reward。这是训练初期防止模式坍缩的机制，不是永久约束。

1. **$E(r) \cdot H_\text{LLM}(r)$ 乘法**：效率和有效性不能单独优化。消融验证：
   - 只有 $H_\text{LLM}$（有效性）→ TTR=2.63，过度提问
   - 只有 $E(r)$（效率）→ ACC=25.1%，激进截断推理
   - 两者结合 → TTR=1.80，ACC=32.70%

其中 $E(r) = (N_\text{max} - n)/(N_\text{max}-1)$ 惩罚多余轮次；$H_\text{LLM}(r)$ 由 GPT-4o-mini 从对齐性、有用性、清晰度三维度评分。

---

#### 实验结果

| 方法                        | MathChat ACC | Tokens(k) | TTR      | BigCode PR | DocEdit BLEU |
| --------------------------- | ------------ | --------- | -------- | ---------- | ------------ |
| CollabLLM（最优 baseline）  | 16.20%       | 1.99      | 3.65     | 10.20%     | 28.00        |
| Instruct + Proactive Prompt | 22.90%       | 2.20      | 3.71     | 19.70%     | 7.10         |
| Active SFT（交互）          | 9.70%        | 2.06      | 1.22     | 9.10%      | 24.75        |
| **US-GRPO（交互）**         | **32.70%**   | **1.70**  | **1.80** | **22.90%** | **41.36**    |

---

值得注意：Active SFT 在交互设置下准确率反而下降（MathChat 从 15.30% 降至 9.70%），说明 SFT 阶段学到的交互格式在真实多轮对话中不稳定。US-GRPO 才是让交互真正对齐用户意图的关键。

**用户模拟器质量的影响**：Llama-3.1-8B → GPT-4o-mini 作模拟器，ACC 从 32.70% 提升至 34.00%，TTR 从 1.80 升至 2.27（更强的模拟器引发更多但更有效的澄清）。强模拟器成本约 $111，弱模拟器约 $16。

---

**泛化测试**（非交互 benchmark）：

| 场景                 | 基线   | PIR（弱 Simulator） | PIR（强 Simulator） |
| -------------------- | ------ | ------------------- | ------------------- |
| MMLU（factual）      | 60.12% | 60.21%              | 62.51%              |
| TriviaQA（QA）       | 19.77% | 25.56%              | 45.51%              |
| SQuAD（QA）          | 6.24%  | 22.94%              | 35.93%              |
| MIP-MATH（缺失前提） | 7.68%  | 25.00%              | 25.00%              |

PIR 学会了适应性：factual 任务参数知识充足，几乎不提问；QA 和 MIP 场景外部信息关键，显著受益于交互。

---

## 三、SFT 与 GRPO 的分工总结

| 论文           | SFT 目标                                | GRPO 目标                         | 不用 GRPO 的理由                           |
| -------------- | --------------------------------------- | --------------------------------- | ------------------------------------------ |
| ProactiveBench | 学习预测格式                            | —                                 | 任务简单，标准分类 SFT 足够                |
| PRISM          | 同时学习 pneed/paccept 估计（RDC 过滤） | —                                 | 核心是概率校准，GRPO 会破坏校准            |
| GPS            | Reasoner 学习 DAG 输出格式              | 优化 DAG 的有效性、效率、结构质量 | 序列决策，需要在完整交互结果上优化         |
| PIR            | 学习 think-ask-respond 交织格式         | 对齐用户意图，优化提问时机和内容  | 序列决策，SFT 学到的格式在真实交互中不稳定 |

---

## 四、缺陷与展望

### 现有局限

**主动提示**
- PRISM 假设 $C_\text{FA}$、$C_\text{FN}$ 静态不变；现实中打断代价随用户心流状态动态变化（截止日期临近时漏检代价上升，用户专注时虚警代价上升）
- 评估依赖 LLM-as-Judge，与真实用户体验仍有偏差（尽管 PRISM 做了 Cohen's κ=0.71 的人工验证）
- 测试场景（coding/writing/daily life）较有限，工业场景泛化性待验证

---

**主动提问**
- GPS 依赖文档中存在显式条件规则，对开放域任务或规则隐含在对话历史中的场景适用性有限
- PIR 缺乏安全对齐评估，对敏感话题的处理行为未知；用户模拟器质量强烈影响训练效果但成本高
- GPS 只训练 Reasoner，Clarifier 固定；端到端联合优化未探索

---

### 主动提问对主动提示的借鉴

**GPS 的 DAG → 主动提示的意图空间建模**

GPS 用 DAG 显式枚举所有条件路径，保证逻辑完备性。类比地，主动提示可以在介入之前，先对用户意图空间构建结构化表示——不是单一的 $p_\text{need}$ 概率，而是多个候选意图及其条件关系，当且仅当当前观测能将意图空间收窄到单一分支时才介入，否则沉默。这可以直接解决 ProactiveBench 中 False-Alarm 过高的根本问题（模型在意图仍有歧义时过早介入）。

**PIR 的 PE 触发 → 主动提示的"何时保持沉默"判断**

PIR 通过预测熵识别推理链中的高不确定区域。这与 PRISM 的 $p_\text{need}$ 估计本质类似，但 PIR 衡量的是模型内部对推理方向的不确定性，而 PRISM 衡量的是对用户需求的不确定性。两者可以结合：当模型对用户意图的内部推断熵高时，提高门控阈值，做出更保守的介入决策。

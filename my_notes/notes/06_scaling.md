# 06 · 进阶 Notebook + Scaling Laws 学习笔记（Q&A 版）

> Phase 6：把 nanoGPT 自带的两个 notebook 跑一遍，对照 nanochat 看看下一站。
>
> 上游：[`../00_learning_plan.md`](../00_learning_plan.md) ·
> Notebook：[`../nanoGPT/transformer_sizing.ipynb`](../nanoGPT/transformer_sizing.ipynb)、
> [`../nanoGPT/scaling_laws.ipynb`](../nanoGPT/scaling_laws.ipynb)
>
> 预计时长：**3 小时**

---

## 总览

```mermaid
flowchart LR
    A["transformer_sizing.ipynb<br/>参数量 + FLOPs + MFU"] --> B["scaling_laws.ipynb<br/>Chinchilla 推论"]
    B --> C["对照 nanochat<br/>nanoGPT 没有的部分"]
```

| 子任务 | 时长 | 重点 |
|---|---|---|
| transformer_sizing.ipynb | 1h | 参数量、FLOPs/token、MFU 公式 |
| scaling_laws.ipynb | 1h | Chinchilla compute-optimal、N∝C^0.5、D∝C^0.5 |
| 对照 nanochat | 1h | tokenizer 训练、SFT、RLHF、推理、chat UI、Muon |

---

## 一、参数量怎么算

### Q1.1：一个 nanoGPT 模型的总参数量公式是什么？（不算 embedding 的话）

**答**：_TODO（提示：每层 attention `4·n_embd²` + MLP `8·n_embd²` = `12·n_embd²`，再乘 `n_layer`）_

### Q1.2：embedding 占多少？为什么 weight tying 能省掉 lm_head 的 `n_embd · vocab` 参数？

**答**：_TODO_

### Q1.3：GPT-2 124M 实际拆解：

| 模块 | 参数量 |
|---|---|
| `wte` (50257 × 768) | _TODO_ |
| `wpe` (1024 × 768) | _TODO_ |
| 每层 attention | _TODO_ |
| 每层 MLP | _TODO_ |
| LayerNorm 之类 | _TODO_ |
| **总计** | ≈ 124M |

---

## 二、FLOPs/token 怎么算

### Q2.1：`flops_per_token ≈ 6 · N` 这个 6N 是怎么来的？（N = 非 embedding 参数量）

**答**：_TODO_

### Q2.2：attention 那部分 `12 · L · H · Q · T` 在 T（context length）大时为啥不可忽略？

**答**：_TODO_

### Q2.3：MFU = 实际吞吐 / 理论峰值——A100 bf16 理论是多少 TFLOPS？

**答**：_TODO_

---

## 三、scaling laws

### Q3.1：Kaplan et al. 2020 的 scaling laws 跟 Hoffmann et al. 2022（Chinchilla）的差别是什么？

**答**：_TODO_

### Q3.2：Chinchilla 提的"compute-optimal"是：模型参数 N 跟训练 token D 应满足 `D ≈ 20N`。这个比例下，给定算力 C，最优 (N, D) 怎么算？

**答**：_TODO_

### Q3.3：GPT-2 当年是过度训练还是欠训练？用 Chinchilla 法则反推。

**答**：_TODO_

### Q3.4：如果你有 100× GPT-2 的算力预算（≈ 100 × 4×8×A100×day），按 Chinchilla 最优应该训多大的模型？多少 token？

**答**：_TODO_

---

## 四、对照 nanochat

### Q4.1：nanochat 比 nanoGPT 多了哪些 stage？

**答**：_TODO_

| 阶段 | nanoGPT | nanochat |
|---|---|---|
| tokenizer 训练 | _TODO_ | _TODO_ |
| pretrain | ✓ | ✓ |
| SFT | _TODO_ | _TODO_ |
| RLHF / DPO | _TODO_ | _TODO_ |
| inference server | _TODO_ | _TODO_ |
| chat UI | _TODO_ | _TODO_ |

### Q4.2：nanochat 用的 Muon optimizer 解决了 AdamW 的什么问题？

**答**：_TODO_

### Q4.3：从 nanoGPT 升级到 nanochat，最值得学的 3 件事是什么？

**答**：_TODO_

---

## 五、与你工作（e2e 自动驾驶）的对照

> 选做：把 nanoGPT 学到的 transformer 训练实践跟你 `atlas_data_product_line` 里的训练代码对照。

### Q5.1：你们 e2e 训练里有没有用 grad accum / autocast bf16 / cosine LR？哪里实现的？

**答**：_TODO_

### Q5.2：自驾 transformer 跟 LLM transformer 在 attention pattern 上的差别？（提示：双向 vs causal、token 数 vs agent 数）

**答**：_TODO_

### Q5.3：scaling laws 对自驾模型还成立吗？数据 vs 参数 vs 算力的三角关系长什么样？

**答**：_TODO_

---

## 六、关键约定与常见坑

```
1. （TODO）`estimate_mfu` 在 grad accum 下的换算（按总 token 还是按一次 forward？）
2. （TODO）Chinchilla 的 20× token/参数 比例对 fine-tune 不适用
```

---

## 七、自测题清单（毕业测试）

- [ ] 闭卷写出 transformer 的 12·n_layer·n_embd² 参数量公式来源
- [ ] 闭卷写出 6N flops/token 的来源
- [ ] 给定算力预算，5 分钟内估出 Chinchilla 最优 (N, D)
- [ ] 列出 nanochat 比 nanoGPT 多的 5 个工程能力

## Deliverable

- [ ] 跑通 `transformer_sizing.ipynb` 并填完 §一表格
- [ ] 跑通 `scaling_laws.ipynb` 并完成 §三 Q&A
- [ ] §五的选做（若时间允许，强烈推荐做完）

---

## 学完整个计划后

恭喜——你已经：

- ✅ 能从 0 训一个 baby GPT
- ✅ 能微调 GPT-2 到任意领域
- ✅ 能改 architecture 做自己的实验
- ✅ 能从 FLOPs/scaling 角度评估"该不该上大模型"

下一站建议：

1. **[nanochat](https://github.com/karpathy/nanochat)** — 全栈 LLM，加上 SFT/RLHF
2. **[modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt)** — 极致优化（Muon、动态 token routing 等）
3. **回到 e2e** — 把 LLM 训练的工程经验（混合精度、grad accum、scaling laws）反哺自驾

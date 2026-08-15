---
title: "Mixture of Experts: Are They Really Experts?"
description: "Mixture of Experts is everywhere in modern LLMs, but nobody seems to check whether the 'experts' are actually experts at anything. This project take pretrained  Switch Transformer, decomposes its expert activations with sparse autoencoders, and asks what the routing and the features actually say about specialization."
pubDate: 2026-08-15
date: 2026-08-15
categories: [interpretability, mixture-of-experts]
tags: [moe, llm, mechanistic-interpretability, sparse-autoencoders, transformer]
draft: false
---

> **TL;DR** - Mixture-of-Experts (MoE) models route each token to a handful of sub-networks called "experts." The name implies specialization, but prior work on Mixtral and OpenMoE suggests routing is mostly not domain-driven. This project reproduces that routing analysis on a small Switch Transformer, then goes one level deeper: it trains sparse autoencoders (SAEs) on each expert's activations to see whether interpretable features, rather than surface-level domains, reveal real specialization , All code is available here: https://github.com/aymen-000/moe-expert-analysis

## Table of contents

1. [Why I'm writing this](#why-im-writing-this)
2. [Background: how Mixture of Experts works](#background-how-mixture-of-experts-works)
3. [Why interpretability is the right lens](#why-interpretability-is-the-right-lens)
4. [Experiments](#experiments)
   - [4.1 Domain-level routing analysis](#41-domain-level-routing-analysis)
   - [4.2 How "sticky" is routing?](#42-how-sticky-is-routing)
   - [4.3 Feature-level analysis with sparse autoencoders](#43-feature-level-analysis-with-sparse-autoencoders)
   - [4.4 Context-independent specialization](#44-context-independent-specialization)
   - [4.5 Language specialization](#45-language-specialization)
5. [Interactive dashboards](#interactive-dashboards)
6. [Discussion: so, are they experts?](#discussion-so-are-they-experts)
7. [Future work](#future-work)
8. [References](#references)
9. [How to cite this work](#how-to-cite-this-work)

---

## Why I'm writing this

Almost every large language model you've heard of in the last two years - Mixtral, Qwen2.5, DeepSeek, Switch Transformer - leans on Mixture of Experts (MoE) to scale up without scaling compute linearly. It has become the default recipe for building big models cheaply.

But the name itself makes a claim. It says these sub-networks are *experts*. Experts in what, exactly? When I went looking for papers that actually test this - papers that open the model up and check whether expert 7 knows something expert 3 does not - I found surprisingly little. A couple of papers touch on it and, spoiler, mostly find that experts *aren't* cleanly specialized in the way the name suggests. That gap is what this blog is about.

This write-up covers two things: the groundwork - what MoE is, how it works mechanically, and why interpretability is the right lens for the "are they experts" question - and the actual experiment: using a small Switch Transformer and probing it with sparse autoencoders to see what's really going on inside.

## Background: how Mixture of Experts works

MoE is not new. It goes back to Jacobs, Jordan, Nowlan, and Hinton's 1991 paper on adaptive mixtures of local experts [1], later extended by Jordan and Jacobs into hierarchical mixtures trained with an EM algorithm [2]. The core idea from three decades ago is basically unchanged: instead of running every parameter on every input, you conditionally route each input to only a portion of the network. This is usually called *conditional computation*.

The modern deep learning version took off with the Sparsely-Gated Mixture-of-Experts layer [3], which showed you could scale a network to enormous size while keeping the *compute per token* roughly fixed, because each token only touches a small subset of the parameters. Google's Switch Transformer [4] pushed this further by simplifying routing to top-1 and reporting roughly a 7x pre-training speedup over a comparably-sized dense T5-Base model. Since then, MoE has become close to standard for frontier open models - Qwen2.5 [5], Skywork-MoE [6], Mixtral [10], DeepSeekMoE, and others all build on the same basic skeleton, just tweaking the routing and balancing tricks.

**What actually changes in the architecture?** In a standard Transformer block, the feed-forward network (FFN) is dense - every token passes through the same weights. MoE's main structural change is to replace that single FFN with a bank of *N* smaller FFNs (the "experts"), plus a small routing network that decides, per token, which expert(s) should handle it. At inference, only the top-*k* experts for a given token actually run; the rest sit idle for that token, which is where the compute savings come from.

### The gating function: the real decision-maker

The **gating function** (or router) decides, for every token, which expert(s) should process it. A good router balances two goals that pull in different directions:

1. **Meaningful specialization** - ideally, different experts learn to handle different kinds of inputs rather than converging on the same function.
2. **Balanced utilization** - tokens need to be spread reasonably evenly across experts. Otherwise you get *expert collapse*: a handful of experts absorb almost all the traffic while the rest barely get trained.

The dominant approach - used in Switch Transformer, GShard, DeepSeekMoE, and Mixtral - is **linear gating**: a small learned linear layer scores each expert for a given token, optional noise is added to encourage exploration, the scores are truncated to the top-*k*, and the result is turned into a probability distribution with softmax:

$$
G(x) = \text{Softmax}\big(\text{TopK}(g(x) + R_{\text{noise}})\big)
$$

where $g(x)$ is the raw linear score for each expert given input $x$, and $R_{\text{noise}}$ is optional injected Gaussian noise. There is also a growing body of work on non-linear alternatives - cosine-similarity routers, distribution-based gating, and "soft" MoE that avoids hard top-*k* cuts entirely - but that is outside the scope of this post.

### Keeping experts from collapsing

The standard fix is an auxiliary **load-balancing loss** that rewards the model for spreading load evenly. The Switch Transformer formulation, which most later models inherit in some form, defines for a batch of $T$ tokens routed across $N$ experts:

$$
L_{\text{balance}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

where $f_i$ is the fraction of tokens in the batch actually dispatched to expert $i$, and $P_i$ is the average router probability assigned to expert $i$ across the batch. This loss is minimized when both are uniform ($1/N$ each), and the $N \cdot$ scaling keeps its magnitude roughly constant regardless of expert count.

### Expert capacity

Even with a balancing loss, routing decisions are dynamic, but GPUs and TPUs prefer fixed, predictable batch shapes. In practice, implementations set a **static expert capacity** - a hard cap on tokens per expert per batch. Tokens that overflow this cap are typically dropped for that layer, often passed through via a residual connection instead. This is a pure engineering trade-off: a bit of routing accuracy sacrificed for hardware utilization - and it means "which expert handles a token" is partly an artifact of capacity limits, not just specialization.

## Why interpretability is the right lens

MoE *looks* like it hands you interpretability for free. If the network literally splits itself into named sub-modules, surely you can just look at which module handles what and read off the specialization? It turns out that is not quite true:

- The **Mixtral** paper [10] includes a section specifically probing expert specialization and reports that routing does *not* appear to be domain-driven.
- The **OpenMoE** paper [11] finds something more nuanced: routing shows some structure around *language/token identity* - e.g. separating programming languages from natural language, and token-ID-level patterns - but nothing resembling clean *conceptual* or domain specialization. The paper names this **Context-Independent Specialization**.

So the closest thing to an answer from prior work is: "somewhat, along superficial axes like language or token identity, but not in a deep conceptual sense." That is an unsatisfying answer for a component whose entire name is a claim about specialization.

This is where Anthropic's interpretability work becomes the missing piece. *Towards Monosemanticity* [8] frames mechanistic interpretability as breaking a network into components that are individually easier to understand than the network as a whole. The uncomfortable finding underneath a lot of this work is that a network's most natural computational unit - the neuron - usually isn't the natural unit for *human* understanding, because individual neurons tend to be **polysemantic**: one neuron can respond to many unrelated concepts at once. This is called **superposition**, and it is a big part of why staring directly at neuron or expert activations does not get you very far on its own.

*Toy Models of Superposition* [9] lays out roughly three ways to deal with this: avoid superposition in the first place, accept it and decode it afterward with dictionary-learning methods like sparse autoencoders, or some hybrid of the two. The framing that matters here: meaning in a neural network tends to live in *directions in activation space*, not in individual neurons. A **feature** is just a meaningful direction in that space.

Put this together and you get the question driving the experiments below: MoE already gives a coarse, architectural form of sparsity - tokens get routed to a small subset of experts. SAEs give a *learned*, fine-grained form of sparsity - decomposing activations into interpretable features. What happens when you point the second one at the first? Do the features that light up inside a given expert tell a coherent story about what that expert "knows"? Or does the mess just move down a level?

---

## Experiments

**Setup, briefly:** the experiments below use a pretrained Switch Transformer [4] from Hugging Face as the base MoE model, configured with 8 experts per MoE layer and top-1 routing. Routing statistics are computed by feeding text from multiple domains through the model and logging which expert each token is assigned to at different encoder depths (enc_1 = an early layer, enc_11 = a much deeper layer). For the evaluation corpus, I use the pile-10k dataset from Hugging Face for diverse text domains (e.g., ArXiv, GitHub, StackExchange, etc.), while Wikipedia is used as the natural language corpus. For the feature-level analysis, a separate sparse autoencoder is trained for each expert using that expert's routed-token activations, and the learned features are then automatically annotated rather than inspected manually.

### 4.1 Domain-level routing analysis

This section tries to answer a simple question: do different experts prefer different domains, or is routing mostly uniform? That question appears in a light form in the Mixtral and OpenMoE papers, but I wanted to test it directly in this model.

**Early layers are close to uniform; specialization emerges with depth.** In `enc_1`, most experts sit in a fairly narrow 8-17% band per domain - close to the uniform baseline of 12.5% for 8 experts. By `enc_11`, the spread widens substantially: **expert 7** pulls 20-34% of tokens across almost every domain (21.8% on ArXiv, 32.9% on GitHub, 33.5% on DM Mathematics, 23% on StackExchange), while **expert 1** nearly vanishes on some domains (1.2% on DM Mathematics). That is a shift from roughly uniform routing toward one or two experts dominating as depth increases.

**Some domains specialize even in the first layer.** GitHub and DM Mathematics already show close to an 8x difference in routing probability relative to uniform, even at `enc_1` - suggesting the router recognizes programming and math tokens very early in the network, likely from surface-level lexical cues such as syntax characters and digit patterns rather than anything conceptual yet.

**Expert 7 looks like a general higher-level reasoning expert**, especially for technical domains, by the time you reach the deeper layers.

**The dominant expert changes with depth.** For GitHub tokens, the leading expert at `enc_1` is not the same expert that leads at `enc_7`. That suggests different experts are handling different stages of representation as information flows through the network: early layers doing more lexical and syntactic work, later layers doing something closer to semantic or task-level work.

Notably, this domain-level skew is more pronounced than what the Mixtral paper [10] reports - Mixtral finds routing is essentially domain-agnostic even at scale, whereas this smaller Switch Transformer shows real, depth-dependent domain skew. That difference is worth digging into in future work.

Here is a summary figure for expert distribution by domain:

![](/imgs/expert_distribution_by_domain.png)



### 4.2 How "sticky" is routing?

A separate question from *which* expert gets a domain is *how long the router sticks with the same expert* as it moves through a sequence of tokens.

- Routing is **significantly more coherent than random** - consecutive tokens are much more likely to share an expert than chance would predict.
- **Neighboring tokens tend to stay with the same expert**, consistent with the coherence patterns reported for Mixtral [10].
- **Coherence varies by domain.** DM Mathematics and GitHub show the highest coherence, plausibly because both domains contain long, structurally repetitive expressions such as equations and code blocks.
- **Intermediate encoder layers show the strongest routing consistency**, suggesting specialization peaks somewhere in the middle of the network rather than at the very first or very last layer.
- Practically, this "stickiness" is a hint that routing decisions could be cached or amortized across neighboring tokens to save compute.


<div class="dashboard-embed">
  <iframe
    src="/dashboards/moe_routing_visualization.html"
    title="MoE routing visualization"
    loading="lazy"
    style="width: 100%; height: 820px; border: 1px solid rgba(0,0,0,0.12); border-radius: 14px; background: #fff;"
  ></iframe>
</div>


### 4.3 Feature-level analysis with sparse autoencoders

Domain-level statistics only go so far, for the same reason neuron-level statistics only go so far in dense networks: **superposition**. A given expert can be "responsible for" a domain in the aggregate while its individual activations still encode many unrelated concepts at once. To get underneath that, this project trains a separate sparse autoencoder on each expert's routed-token activations and studies the resulting *features* - directions in activation space - rather than raw domain counts.

After training the sparse autoencoders, I extracted the top features routed by each expert. The first and last layers are shown below, using the highest-mass features:

![](/imgs/top_features_per_expert_enc_1.png)

![](/imgs/top_features_per_expert_enc_11.png)

The next question is obvious: what do these features actually mean?

**The scaling problem.** Each SAE surfaces on the order of 1,000+ candidate features per expert. Manually inspecting each one is impractical. Two options: manual annotation, which is accurate but slow, or automated annotation. This project went with automation: for each expert, the most active features were extracted for the first and last of the six analyzed layers, filtered to remove uninformative triggers such as punctuation, whitespace, and boilerplate tokens, and passed - together with their activating tokens, weighted by activation mass and density - to an LLM (DeepSeek, via OpenRouter) for annotation.

The resulting picture, for `enc_1` (early layer) and the first six experts analyzed at `enc_11` (deep layer), is:

| Layer | Expert | Main specialization |
|-------|--------|---------------------|
| enc_1 | 0 | Rare named entities + technical terminology |
| enc_1 | 1 | Scientific/technical morphology (abbreviations, suffixes, fragments) |
| enc_1 | 2 | Programming + identifiers + technical tokens |
| enc_1 | 3 | Programming/code structure + proper nouns |
| enc_1 | 4 | General semantic/content words |
| enc_1 | 5 | Character patterns and subword morphology |
| enc_1 | 6 | Numbers and numerical sequences |
| enc_1 | 7 | Proper names and geographic entities |
| enc_11 | 0 | Temporal information (years, dates, durations) |
| enc_11 | 1 | Scientific entities and measurement units |
| enc_11 | 2 | Mathematical reasoning and numerical expressions |
| enc_11 | 3 | Linguistic suffixes + discourse/question markers |
| enc_11 | 4 | Named entities and code identifiers |
| enc_11 | 5 | Syntax/discourse relations (relative clauses, contrast) |

You can inspect the token-level visualizations here:

- [First Layer Tokens-Features relation](/dashboards/expert_feature_dashboard_enc_1.html)
- [Last Layer Tokens-Features relation](/dashboards/expert_feature_dashboard_enc_11.html)

Two things stand out here. First, this is a much cleaner and more consistent picture of specialization than the domain-level analysis in Section 4.1 gave - feature-level directions correspond to fairly coherent linguistic or conceptual categories such as numbers, dates, code identifiers, discourse markers, and named entities in a way raw domain percentages did not. Second, the categories that emerge are still mostly **surface-level or shallow-semantic**, not deep conceptual domains in the "math expert" or "biology expert" sense the name "Mixture of Experts" suggests.

So: **feature space is more informative than domain space**, and it is possible to see real expertise at the feature level - it is just a different, more granular kind of expertise than the architecture's name implies.

### 4.4 Context-independent specialization

A striking pattern shows up when tracking individual tokens rather than domains: **the same token ID tends to route to the same expert regardless of its context**. For example, the token "ed" can be the suffix of many unrelated words such as "preferred" or "led," and "an" can appear inside "an apple" or "another" - semantically and syntactically very different contexts. Despite that, both show very strong specialization toward only a few fixed experts across all their occurrences. This is the same kind of result reported in OpenMoE.

In other words, routing appears to be based largely on **token identity**, not high-level semantics of the surrounding sentence. This matches the finding the OpenMoE paper [11] calls **Context-Independent Specialization**, and it is consistent with the token-ID-level structure noted in Section 4.1. It also helps explain the routing "stickiness" observed in Section 4.2, since repeated or similar token IDs would naturally route consistently.

### 4.5 Language specialization

Finally, looking at routing across different input languages:

- **Routing is language-dependent** - expert selection varies substantially depending on the input language.
- **English is distributed relatively uniformly** across experts; no single expert dominates.
- **Non-English languages route more narrowly.** Arabic and Korean, in particular, show highly concentrated routing, with more than 40% of tokens assigned to a single expert (Expert 1).
- **Experts 1 and 7 act as multilingual specialists, but for different language groups.** Expert 1 dominates for Arabic, Korean, French, and Japanese; Expert 7 is uniquely favored for Chinese. The router effectively partitions languages into different expert groups rather than treating "non-English" as one bucket.

Here is the language-routing summary:

![](/imgs/language_radar_enc_7.png)

---

## Discussion: so, are they experts?

Pulling the threads together: at the **domain** level, the picture partially supports the "expert" framing - some domains such as math and code get recognizable, if imperfect, specialization, and that specialization sharpens with depth. But a meaningful chunk of the routing signal turns out to be explained by much shallower factors: **token identity** (Section 4.4) and, to some extent, **routing stickiness inherited from neighboring tokens** (Section 4.2) rather than anything resembling conceptual domain understanding.

The **feature-level** view (Section 4.3) is where the "expert" framing holds up best - SAE features recovered from expert activations map onto fairly coherent categories such as numbers, dates, code identifiers, discourse markers, and named entities. But these are still closer to *linguistic or morphological roles* than to the subject-matter domains ("math," "biology," "law") that the word "expert" tends to conjure.

So the honest answer, at least for a Switch Transformer at this scale, is: MoE experts are real, measurable specialists - just specialists in fine-grained linguistic and token-level structure, not in the domain-level sense the architecture's name implies. Whether that changes at larger scale, and whether it is causally load-bearing rather than just correlational, are exactly the open questions in the next section.

---

## Future work

- **Extend to vision.** Apply the same pipeline (per-expert SAEs + automated feature annotation) to a vision MoE transformer and check whether image-patch routing shows the same context-independent, token-patch-identity-driven pattern found here for text, or whether visual inputs produce cleaner domain-level specialization.
- **Scale up to production-size MoE models.** Repeat the domain-level and feature-level analyses on larger, widely-used models such as Mixtral, DeepSeekMoE, or Qwen's MoE variants. This project's Switch Transformer shows more domain skew than Mixtral reports at scale, so it is worth checking directly whether that gap is a scale effect, a training-data effect, or an architectural one.
- **Causal ablation, not just observation.** All of the results above are correlational - they describe what the router does, not what happens if you remove what it built. A natural next experiment is to ablate a single expert that the analysis associates with a domain and measure downstream performance on that domain's benchmark, both with the expert isolated and with the expert removed.
- **Exploit temporal and token-identity locality for efficiency.** Sections 4.2 and 4.4 both point at the same underlying fact: routing is far more sticky than random, both across neighboring tokens and across repeated token IDs. That is a real systems opportunity - for example, caching or predicting expert assignments ahead of time for a token window, or scheduling compute and communication around known high-coherence domains such as code and math.

---

## References

[1] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, "Adaptive mixtures of local experts," *Neural Computation*, vol. 3, no. 1, pp. 79-87, 1991.

[2] M. I. Jordan and R. A. Jacobs, "Hierarchical mixtures of experts and the EM algorithm," *Neural Computation*, vol. 6, no. 2, pp. 181-214, 1994.

[3] N. Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," arXiv:1701.06538, 2017.

[4] W. Fedus, B. Zoph, and N. Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," arXiv:2101.03961, 2021.

[5] A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, et al., "Qwen2.5 Technical Report," arXiv:2412.15115, 2024.

[6] Y. Wei, Z. Wang, Y. Li, Y. Li, Z. Xu, Y. Liu, J. Guo, and X. Zhang, "Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models," arXiv preprint arXiv:2406.06563, 2024.

[7] S. Mu and S. Lin, "Survey on MoE Routing Strategies," arXiv preprint arXiv:2503.07137, 2025.

[8] Bricken, et al., "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning", Transformer Circuits Thread, 2023.

[9] Elhage, et al., "Toy Models of Superposition", Transformer Circuits Thread, 2022.

[10] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. de las Casas, E. B. Hanna, F. Bressand, et al., "Mixtral of Experts," arXiv:2401.04088, 2024.

[11] F. Xue, Z. Zheng, Y. Fu, J. Ni, Z. Zheng, W. Zhou, and Y. You, "OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models," arXiv:2402.01739, 2024.

---

## How to cite this work

If you use these results, please cite it.

```bibtex
@misc{moe_are_they_experts_2026,
  author       = {Aimen Boukhari},
  title        = {Mixture of Experts: Are They Really Experts?},
  year         = {2026},
  howpublished = {Blog post},
}
```

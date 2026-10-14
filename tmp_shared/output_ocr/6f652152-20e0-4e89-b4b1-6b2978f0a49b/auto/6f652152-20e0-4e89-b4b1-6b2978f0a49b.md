# RouteRAG: Efficient Retrieval-Augmented Generation from Text and Graph via Reinforcement Learning

Yucan Guo, Miao Su, Saiping Guan†, Zihao Sun, Xiaolong $\mathbf { J i n } ^ { \dagger }$ , Jiafeng Guo, Xueqi Cheng 1State Key Laboratory of AI Safety 2Institute of Computing Technology, Chinese Academy of Sciences 3University of Chinese Academy of Sciences

{guoyucan23z, sumiao22z, guansaiping, sunzihao18z, jinxiaolong, guojiafeng, cxq}@ict.ac.cr

# Abstract

Retrieval-Augmented Generation (RAG) integrates non-parametric knowledge into Large Language Models (LLMs), typically from unstructured texts and structured graphs. While recent progress has advanced text-based RAG to multi-turn reasoning through Reinforcement Learning (RL), extending these advances to hybrid retrieval introduces additional challenges. Existing graph-based or hybrid systems typically depend on fixed or handcrafted retrieval pipelines, lacking the ability to integrate supplementary evidence as reasoning unfolds. Besides, while graph evidence provides relational structures crucial for multi-hop reasoning, it is substantially more expensive to retrieve. To address these limitations, we introduce RouteRAG, an RL-based framework that enables LLMs to perform multi-turn and adaptive graph-text hybrid RAG. RouteRAG jointly optimizes the entire generation process via RL, allowing the model to learn when to reason, what to retrieve from either texts or graphs, and when to produce final answers, all within a unified generation policy. To guide this learning process, we design a two-stage training framework that accounts for both task outcome and retrieval efficiency, enabling the model to exploit hybrid evidence while avoiding unnecessary retrieval overhead. Experimental results across five question answering benchmarks demonstrate that RouteRAG significantly outperforms existing RAG baselines, highlighting the benefits of end-to-end RL in supporting adaptive and efficient retrieval for complex reasoning. 1

# 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning, decision-making, and long-form generation (Zhao et al., 2023; Touvron et al., 2023; Team et al., 2024), especially when further trained with Reinforcement Learning (RL) (OpenAI et al., 2023; DeepSeekAI et al., 2025; Yang et al., 2025a). These abilities have enabled LLMs to follow complex instructions, emulate chain-of-thought reasoning, and solve complicated multi-hop questions (Zhou et al. 2023; Wei et al., 2022). However, the knowledge of LLMs remains static, bounded by the data available at pretraining time. As a result, LLMs often produce inaccurate or outdated outputs when faced with knowledge-intensive queries that require access to external or up-to-date information (Augenstein et al., 2024; Huang et al., 2025).

To overcome this limitation, RetrievalAugmented Generation (RAG) has emerged as a core paradigm for enhancing LLMs with access to external knowledge sources (Lewis et al., 2020; Gao et al., 2023). By conducting retrieval at inference time, RAG decouples knowledge from model parameters, improving factual accuracy and generalization. Early RAG systems typically perform a single round of retrieval before generation (Guu et al., 2020; Wang et al., 2023). Recent work has shown the benefits of multi-turn retrieval, where the model interleaves retrieval and reasoning over multiple steps (Yao et al., 2023; Trivedi et al., 2023; Li et al., 2025). However, these prompt-based approaches often depend on large closed-source models with strong intrinsic reasoning and planning skills. Smaller open-source models struggle to determine when to retrieve, how to formulate retrieval queries, and how to analyze retrieved evidence. This gap has motivated a new line of research (Jin et al., 2025; Song et al., 2025) that employs RL to explicitly train models to make retrieval and reasoning decisions. By optimizing a learned policy over interleaved thinking and retrieval actions, these RL-based methods aim to equip models with adaptive, context-sensitive retrieval strategies that surpass static instructions.

In parallel, graph-based RAG systems (Edge et al., 2024; Gutiérrez et al., 2024, 2025) utilize structured knowledge graphs to integrate and reason over information scattered across multiple passages, thereby improving coverage of factual entities and relations. While graphs enable more accurate entity disambiguation and multi-hop path reasoning than text-only retrieval, retrieving and processing graph evidence is often more computationally expensive, especially in large-scale or dense graphs. Moreover, existing graph-based RAG systems typically operate in a one-shot retrieval setting, fetching graph evidence once before generation, and lack the ability to adaptively choose between graph and text retrieval based on the evolving information needs of the query. Consequently, the current architecture of graph-based RAG systems presents challenges in managing complex reasoning that necessitates multi-turn interactions, and can also lead to unnecessary retrieval overhead when the reasoning chain is too long.

We address these limitations with RouteRAG, an RL-based framework that enables LLMs to perform multi-turn and hybrid retrieval over both unstructured texts and structured knowledge graphs. Instead of passively executing preset instructions, RouteRAG actively orchestrates retrieval decisions, selecting when and where to access external knowledge. To overcome the challenges of managing complex reasoning and avoiding unnecessary retrieval overhead, RouteRAG learns to interleave reasoning, retrieval, and answer formulation through a unified generation policy, adapting its retrieval behavior to the evolving task context.

To enable RouteRAG to generate accurate answers while efficiently retrieving relevant knowledge, we adopt a two-stage Group Relative Policy Optimization (GRPO) (Shao et al., 2024) training framework. In the first stage, the model is rewarded solely for answer correctness, allowing it to acquire the core capability of generating accurate responses and establishing a solid starting point for further optimization. In the second stage, we introduce an additional efficiency reward that discourages unnecessary retrieval, guiding the model to strike a balance between accuracy and computational cost. With these designs, RouteRAG can achieve both high accuracy and retrieval efficiency in complex multi-hop reasoning tasks.

Our main contributions lie in three aspects:

•We propose RouteRAG, an RL-based framework for multi-turn and hybrid RAG. The model learns a unified generation policy that interleaves reasoning, adaptive graph-text hybrid retrieval, and answer formulation through a two-stage training framework.

•We design a reward function that jointly optimizes answer accuracy and retrieval efficiency, encouraging the model to retrieve selectively and to reason effectively over retrieved evidence across multiple steps.

• Extensive experiments on five Question Answering (QA) benchmarks demonstrate that RouteRAG outperforms prior multi-turn and graph-based RAG systems significantly.

# 2 Related Work

# 2.1 RAG

RAG has become a key paradigm for enhancing LLMs with external knowledge, thus mitigating hallucination and improving factual grounding (Guu et al., 2020; Gao et al., 2023). Traditional RAG systems retrieve relevant text chunks from an external knowledge base according to the query, and then feed the query into the LLM together with those text chunks to generate a final answer (Lewis et al., 2020; Yu et al., 2022). Beyond such one-shot retrieve-then-generate pipelines, recent research has explored multi-turn retrieval to provide more fine-grained and incremental supplementation of external knowledge, interleaving reasoning with evidence acquisition. For instance, IRCoT (Trivedi et al., 2023) shows that alternating chain-of-thought reasoning with retrieval improves the performance of LLM on knowledge-intensive multi-hop QA. Search-o1 (Li et al., 2025) further develops this line by introducing a reason-in-documents module to alleviate the issue of redundant information in retrieved documents.

As an alternative route for deep knowledge integration, graph-based RAG methods incorporate structured knowledge graphs to aggregate evidence across passages and to make relational connections explicit (Peng et al., 2024; Zhao et al., 2023). By exposing entities and relations directly, these methods are particularly effective for multi-hop questions that require linking facts across disparate documents (Edge et al., 2024; Gutiérrez et al., 2024, 2025). However, graph retrieval is often more computationally expensive than text retrieval, and existing methods commonly perform one-shot retrieval. Although recent work such as HybGRAG (Lee et al., 2025b) demonstrates that multi-turn hybrid text-graph retrieval is feasible through predefined multi-step procedures, these methods rely on fixed heuristics rather than a learnable policy.

# 2.2 RL for LLM Reasoning

RL has played a central role in improving the reasoning capabilities of LLMs. RL from Human Feedback (RLHF) (Christiano et al., 2017; Ouyang et al., 2022) has established a standard paradigm, where a reward model trained from human preferences directs the optimization of policies (Lambert et al., 2025), allowing models to adhere to instructions and reason with greater accuracy. Proximal Policy Optimization (PPO) (Schulman et al., 2017) remains the predominant algorithm for achieving these goals. More recently, GRPO (Shao et al., 2024) has been proposed as a more efficient variant, which leverages group-wise relative rewards to stabilize training and reduce variance. Building on these advances, researchers have begun to apply RL directly to the training of multi-turn RAG systems (Jin et al., 2025; Song et al., 2025). For instance, Search-R1 (Jin et al., 2025) trains LLMs with RL to decide when and what to search in the middle of reasoning, using only outcome rewards. While this reward design effectively improves correctness, it does not explicitly address retrieval cost or efficiency.

# 3 RouteRAG

In this section, we present RouteRAG, an RL-based framework for multi-turn hybrid RAG. We first describe the multi-turn workflow and the mechanism for hybrid knowledge access (Section 3.1). Subsequently, we introduce our two-stage RL framework (Section 3.2), encompassing the formulation of outcome and efficiency rewards, alongside the GRPO-based training algorithm.

# 3.1 Overall Framework

We begin by outlining the overall architecture of RouteRAG. This framework integrates LLMs with external retrievers in a multi-turn reasoning loop, where special tokens from the reasoning process can trigger retrieval actions from text and graph knowledge sources. We describe the multi-turn reasoning and retrieval workflow in Section 3.1.1 and the hybrid knowledge access mechanisms of the external retriever in Section 3.1.2.

# Algorithm 1 RouteRAG Framework

Require: Input query $q$ , policy model $\pi \theta$ , retriever $\mathcal { R }$ , maximum step budget $B$ .

Ensure: Final response $_ y$ .   
1: Initialize response $y  \emptyset$ , step count $b \gets 0$   
2: while $b < B$ do   
3: Initialize current rollout $y _ { b } \gets \emptyset$   
4: while True do   
5: Sample next token $y ^ { \prime } \sim \pi _ { \boldsymbol { \theta } } ( \cdot \mid q , y + y _ { b } )$   
6: $y _ { b } \gets y _ { b } + y ^ { \prime }$   
7: if y'  {</search>, </answer>, ${ < } \mathrm { e o s } >$ } then   
8: break   
9: $y  y + y _ { b }$ Combine rollout with history   
10: if <search>...</search> detected in $y _ { b }$ then   
11: if [passage] in $y _ { b }$ then $m \gets$ Passage   
12: if [graph]in $y _ { b }$ then $m \gets$ Graph   
13: if[passage]and[graph] in $y _ { b }$ then   
14: $m \gets$ Hybrid   
15: Extract query $q ^ { \prime } \gets \mathrm { P a r s e Q u e r y } ( y _ { b } )$   
16: $d \gets \mathcal { R } ( \mathbf { \hat { q } ^ { \prime } } , \mathbf { \hat { m } } )$ Retrieve documents according   
to the retrieval mode m   
17: $y  y +$ <information>d</information>   
18: else if <answer>. </answer> detected in $y _ { b }$ then   
19: return final response $_ y$   
20: $b \gets b + 1$   
21: return final response $y$

# 3.1.1 Multi-Turn Reasoning and Hybrid Retrieval Workflow

We formulate multi-turn retrieval-augmented generation as a sequential decision-making process, as shown in Algorithm 1. Given an input query $q$ , the policy model $\pi _ { \theta }$ interacts with external knowledge sources over a sequence of steps $b =$ $\{ 1 , \ldots , B \}$ , where $B$ is the maximum step budget. At each step, the policy model conditions on the query and the current context to generate an action token. The action space includes continuing internal reasoning, triggering a retrieval operation (<search>...</search>), or producing a final answer (<answer>...</answer>). The retrieval operation further specifies a retrieval mode $m \in \{ \mathsf { P a s s a g e } , \mathsf { G r a p h } , \mathsf { H y b r i d } \}$ by special tokens $y \in \{$ [passage], [graph] $\}$ and a sub-query $q ^ { \prime }$ which are used to obtain documents $d$ from the retriever $\mathcal { R } ( q ^ { \prime } , m )$ . The retrieved information is then appended to the context and becomes available for subsequent reasoning.

This workflow enables the model to progressively refine its knowledge state by deciding what to retrieve and when to retrieve it, conditioned on the evolving reasoning trajectory. Moreover, the explicit action space over different retrieval modes allows the model to adaptively choose among passage, graph, and hybrid retrieval, depending on the requirements of the query.

  
Figure : Previous RL-based multi-turn RAG vs. RouteRAG. Prior methods mainly focus on interleaving reasoning with passe etrival an ewar nswercoretesRouteRAGextends rerival pssage,graph, and modes, and is trained with a two-stage RL framework that optimizes both accuracy and efficiency.

# 3.1.2 Hybrid Knowledge Access

In RouteRAG, the retriever $\mathcal { R }$ is responsible for providing external knowledge to support reasoning, with three different retrieval modes.

Passage Retrieval. The passage retriever is implemented with Dense Passage Retrieval (DPR) (Karpukhin et al., 2020), which encodes both the sub-query and all passages in the corpus into a shared embedding space. Retrieval is performed by computing similarity scores between the query vector and passage vectors, and the top- $k$ passages are selected as evidence.

Graph-based Retrieval. The graph retriever is implemented based on HippoRAG 2 (Gutiérrez et al., 2025), which first constructs a knowledge graph over passages. Given a sub-query, the retriever applies personalized PageRank over the graph to propagate relevance from query-linked nodes, thereby identifying passages that are related to the query through multi-hop connections.

Hybrid Retrieval. The hybrid retriever combines passage and graph retrieval using Reciprocal Rank Fusion (RRF) (Cormack et al., 2009). Specifically, given two ranked lists, each document is assigned a fused score that decreases with its reciprocal rank in each list, which ensures that documents highly ranked by either retrieval mode are promoted in the merged list. Formally, the fused score is defined as

$$
\operatorname { R R F } ( d ) = \sum _ { m \in \{ \mathsf { P a s s a g e } , \mathsf { G r a p h } \} } \frac { 1 } { k + \operatorname { r a n k } _ { m } ( d ) } ,
$$

where $\mathrm { r a n k } _ { m } ( d )$ denotes the rank position of document $d$ in retrieval mode $m$ , and $k$ is a smoothing hyperparameter. Documents are then re-ranked according to ${ \mathrm { R R F } } ( d )$ to form the final hybrid list.

# 3.2 Two-Stage Reinforcement Learning

To optimize the unified generation policy, RouteRAG is trained with a two-stage RL framework based on GRPO. The motivation is to first ensure that the model acquires the basic ability to produce correct answers, and then to further refine its retrieval strategy to improve efficiency without sacrificing accuracy, as shown in Figure 1. In this section, we introduce the reward design that guides the learning objectives (Section 3.2.1) and the training algorithm that realizes the optimization procedure (Section 3.2.2).

# 3.2.1 Reward Design

RL optimization is fundamentally guided by the reward signal. To support the two-stage training, we devise different rewards for each stage, i.e., outcome-oriented reward and accuracy—efficiency reward.

Stage 1: Outcome-Oriented Reward. In the first stage, the reward is defined purely by the correctness of the model output. Specifically, the reward is set to 1 if the generated answer $y$ exactly matches the ground-truth label $y ^ { * }$ , and 0 otherwise:

$$
R _ { \phi } ( x , y ) = \operatorname { E } \mathbf { M } ( y , y ^ { * } ) .
$$

Stage 2: Accuracy-Efficiency Reward. In the second stage, we extend the reward function to jointly optimize for correctness and retrieval efficiency. The reward is defined as

$$
R _ { \phi } ( x , y ) = \left\{ \begin{array} { l l } { R _ { \mathrm { o u t c o m e } } , } & { R _ { \mathrm { o u t c o m e } } = 0 } \\ { R _ { \mathrm { o u t c o m e } } + R _ { \mathrm { e f f c i e n c y } } , R _ { \mathrm { o u t c o m e } } = 1 } \end{array} , \right.
$$

where $R _ { \mathrm { o u t c o m e } } \in \{ 0 , 1 \}$ denotes exact match accuracy. The efficiency reward $R _ { \mathrm { e f f i c i e n c y } }$ is computed from the total retrieval time across all reasoning steps, and only for those trajectories that correctly reach the answer. We apply a centered scaling by subtracting the average retrieval time $t _ { \mathrm { a v g } }$ , such that

$$
R _ { \mathrm { e f f i c i e n c y } } = \frac { t _ { \mathrm { a v g } } - t } { T } ,
$$

where $t$ is the total retrieval time for the current trajectory, $t _ { \mathrm { a v g } }$ is the average retrieval time of the current batch, and $T$ is a normalization constant ensuring the value of $t$ and $t _ { \mathrm { a v g } } \in [ 0 , 0 . 5 ]$ . This design provides positive reward for trajectories that achieve the correct answer at a pace exceeding the average, while imposing penalties on those that do not, thereby encouraging the model to retrieve more selectively without sacrificing answer quality.

# 3.2.2 Training Algorithm

We adopt GRPO (Shao et al., 2024; DeepSeek-AI et al., 2025) to train the unified generation policy $\pi _ { \theta }$ over interleaved reasoning and retrieval actions. GRPO stabilizes learning by comparing trajectories within a group, thereby reducing variance in sparsereward settings.

The policy model $\pi _ { \theta }$ is optimized by maximizing the following objective:

$$
\begin{array} { l } { \displaystyle { J _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } _ { x \sim Q , \{ y _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( Y \mid q ) } \frac { 1 } { G } \sum _ { i = 1 } ^ { G } } } \\ { \displaystyle { \left[ \operatorname* { m i n } \left( r _ { i } ( \theta ) A _ { i } , \mathrm { c l i p } ( r _ { i } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) A _ { i } \right) \right. } } \\ { \displaystyle { \left. - \beta \mathbb { D } _ { \mathrm { K L } } [ \pi _ { \theta _ { \mathrm { o l d } } } \| \pi _ { \theta } ] \right] , } } \end{array}
$$

where $\epsilon$ and $\beta$ are hyperparameters, $\pi _ { \theta _ { \mathrm { o l d } } }$ denotes the old policy, $\begin{array} { r } { r _ { i } ( \theta ) = \frac { \pi _ { \theta } ( y _ { i } | x ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i } | x ) } } \end{array}$ θod (yi(x), Ai denotes the group-relative advantage for the $i$ -th trajectory, and the KL penalty $\mathbb { D } _ { \mathrm { K L } } [ \pi _ { \theta _ { \mathrm { o l d } } } \| \pi _ { \theta } ]$ regularizes the new policy against deviating excessively from the old policy. Further theoretical analysis on the effectiveness of the efficiency reward and GRPO is provided in Section A.

# 4 Experiments

# 4.1 Experimental Setting

Evaluation Datasets. Following (Gutiérrez et al., 2024) and Gutiérrez et al. (2025), we evaluate RouteRAG on five widely used benchmarks for simple and multi-hop QA, namely PopQA (Mallen et al., 2023), Natural Questions (NQ) (Kwiatkowski et al., 2019; Wang et al., 2024), HotpotQA (Yang et al., 2018), 2WikiMultihopQA (2Wiki) (Ho et al., 2020), and MuSiQue (Trivedi et al., 2022). PopQA is an open-domain QA dataset designed to evaluate factual recall over long-tail knowledge, and NQ contains naturally occurring queries paired with answers from Wikipedia. HotpotQA and 2Wiki focus on multi-hop reasoning across Wikipedia passages, while MuSiQue requires reasoning over compositional sub-questions.

Baselines. We compare RouteRAG against several types of representative approaches: (1) Vanilla RAG (Lewis et al., 2020), which performs singleshot dense passage retrieval and generation. (2) Multi-turn RAG methods, including Search-o1 (Li et al., 2025), Search-R1 (Jin et al., 2025), and R1- Searcher (Song et al., 2025), wherein the latter two methods utilize RL to enhance multi-turn passage RAG. (3) Graph-based RAG methods, including GraphRAG (Edge et al., 2024), LightRAG (Guo et al., 2025), RAPTOR (Sarthi et al., 2024), HippoRAG (Gutiérrez et al., 2024), and HippoRAG 2 (Gutiérrez et al., 2025), which leverage structured knowledge graphs for retrieval.

Implementation Details. We conduct training using Qwen2.5-3B-Instruct and Qwen2.5-7BInstruct (Yang et al., 2025b) as the backbone models. The training data consists of $1 0 \mathrm { k }$ sampled queries from the HotpotQA training set (Yang et al., 2018), while the retrieval corpus is built from their associated documents. For retrieval, we adopt Contriever (Izacard et al., 2022) and NV-Embedv2 (Lee et al., 2025a) as the dense retriever for 3B and 7B models, respectively. For baseline evaluations, text-based RAG systems are assessed under the same Qwen2.5 backbone, while graph-based RAG systems utilize the GPT-4o-mini backbone.


Table  Main results on simple andmulti-hop QA benchmarks. The best results within each backbone group re icate ol whihendeinvlurerent he seconbest results.rereent -mtas

In particular, HippoRAG 2 (Gutiérrez et al., 2025), the strongest graph-based baseline, is evaluated employing both Qwen2.5 and GPT-4o-mini backbones. We report Exact Match (EM) and F1 scores as evaluation metrics. Additional implementation details, including the training prompt template, hyperparameters, and training configuration, are provided in Section B.

# 4.2 Main Results

We conduct a comprehensive comparison of RouteRAG against all the baseline methods, as shown in Table 1. From the results, we make the following key observations:

(1) RouteRAG substantially improves the performance of a small backbone, especially on multi-hop QA. Graph-based methods such as HippoRAG 2 perform well with the strong GPT-40- mini backbone but drop sharply with the smaller Qwen2.5-3B and Qwen2.5-7B models, indicating that small LLMs struggle to handle complex reasoning chains. In contrast, RouteRAG achieves much better performance on small backbones by jointly learning reasoning, retrieval, and answer generation within a unified policy model.

(2) RouteRAG approaches GPT-4o-minibased graph-based RAG systems despite using a much smaller model. Despite the large performance gap usually observed between GPT-4omini and Qwen2.5-3B/7B, RouteRAG narrows this gap substantially and even surpasses several graphbased systems built on GPT-4o-mini. This suggests that improving the policy can be as impactful as scaling up the backbone itself.

TaAblat studesn RL trai and hyri rerival trai"denotes the basemodelwih RL ai nd sag "  y wi hrst aFo yi evala compaRoueRAGwi vant etrite ty pas rerival, ny rah rereval, nyhybrvl.   



Tabla suy  effy war ef war ra ny i -a rewards.

(3) RouteRAG outperforms the strongest RLtrained multi-turn baseline with much smaller training cost. Search-R1, the prior strongest RL-based multi-turn system, is trained on $1 7 0 \mathrm { k }$ questions from NQ and HotpotQA. Despite being trained on a mere 10k HotpotQA instances, RouteRAG achieves the best average scores among all 3B and 7B methods, demonstrating that structured retrieval and retrieval mode selection can yield more effective and sample-efficient multiturn RAG policies than scaling training data alone. While RouteRAG is slightly weaker on simple QA, this is expected because the training data is dominated by multi-hop questions. Nevertheless, its overall performance remains competitive given the smaller training cost.

# 4.3 Ablation Study

In this section, we conduct ablation experiments to validate the effectiveness of RL training, hybrid retrieval, and the efficiency reward.

RL Training. The upper part of Table 2 compares the full model against two ablated variants: (1) a model trained only with the first stage, and (2) the untrained backbone. The results show that RouteRAG already acquires a strong reasoning ability through the Stage 1 training, outperforming the untrained backbone by a large margin. The Stage 2 training further improves the performance of RouteRAG, especially on multi-hop QA datasets.

Hybrid Retrieval. The lower part of Table 2 presents an ablation study comparing RouteRAG with variants restricted to a single retrieval mode. Passage retrieval performs well on simple QA, while graph retrieval is more effective for multihop reasoning, confirming their complementary strengths. The full RouteRAG achieves the highest average accuracy by dynamically selecting among the three modes, showing that adaptive retrieval yields stronger robustness and generalization than any fixed strategy.

Efficiency Reward. We evaluate whether RouteRAG learns to retrieve evidence more efficiently while keeping effectiveness with the efficiency reward by comparing it to a variant trained only with the outcome reward, using the same number of training steps. As shown in Table 3, RouteRAG maintains comparable or even higher accuracy than its outcome rewards-only counterpart. Figure 2 shows that both RouteRAG-3B and RouteRAG-7B consistently reduce the average retrieval turns across all datasets, with larger savings observed on the 7B model. Table 4 further assesses the trade-off between retrieval turns and task accuracy, highlighting that efficiency gains do not come at the cost of answer quality. This demonstrates that incorporating efficiency rewards encourages the policy to avoid unnecessary retrieval steps while still collecting sufficient evidence. A comprehensive analysis of this phenomenon is available in Section A.4.

  
Figure 2: Comparing the average retrieval turns of RouteRAG and its variant without efficiency reward.

Table 4: Average retrieval turns and accuracy of RouteRAG.   


# 4.4 Analysis on Reasoning Ability

To better understand how the multi-round reasoning ability of RouteRAG evolves during training, we analyze the average number of reasoning steps before and after training. As shown in Figure 3, both the 3B and 7B models exhibit increased reasoning depth after training. Compared with RouteRAG3B, RouteRAG-7B shows a more pronounced improvement, with larger gains on the more complex MuSiQue and 2Wiki datasets, and smaller increases on PopQA and NQ. Figure 4 further expands the analysis by jointly examining performance, response token length, and reasoning turns across datasets. The results show that RouteRAG consistently achieves a more favorable balance among the three factors compared with Search-R1 and R1-Searcher, achieving high F1 with moderate token length. For a qualitative view of how the reasoning behavior changed in practice, case studies comparing model outputs before and after training are presented in Section D.2.

  
Figure 3: Comparison of average reasoning steps.

  
Figure 4: Comparison in terms of performance, response token length, and reasoning turns.

# 4.5 Analysis on Dense Retrievers

Table 5 shows the performance of RouteRAG and Search-R1 with two different dense retrievers, i.e., Contriever (Izacard et al., 2022) and NV-Embedv2 (Lee et al., 2025a). Across both 3B and 7B model sizes, RouteRAG yields consistent gains on multi-hop QA benchmarks, regardless of which dense retriever is used. In contrast, Search-R1 is more sensitive to the quality of the dense retriever, benefiting substantially from the stronger NV-Embed-v2 retriever on simple QA tasks. This highlights that RouteRAG relies less heavily on the dense retriever, since its adaptive use of graph and hybrid retrieval provides accurate evidence, thereby stabilizing performance even when the dense retriever is relatively weak.


Table 5: Performance comparison among RouteRAG and Search-R1 with different dense retrievers.\* represents in-domain datasets.

# 5 Conclusions

In this paper, we presented RouteRAG, an RL framework for efficient multi-turn hybrid RAG. Unlike prior multi-turn RAG systems that rely on static prompting or single-mode retrieval, our approach learns a unified policy that interleaves reasoning, retrieval mode selection, retrieval query generation, and answer generation. Our twostage training framework further ensures that the model first acquires robust answer correctness and then improves retrieval efficiency without sacrificing accuracy. Experiments conducted on five knowledge-intensive QA benchmarks demonstrate that RouteRAG significantly outperforms existing graph-based and multi-turn RAG systems, highlighting that efficiency gains can be achieved without compromising answer quality.

# Limitations

Despite the strong empirical results, our proposed RouteRAG has several limitations. First, due to computational constraints, we conduct RL training and evaluation only on 3B and 7B LLMs. Larger or more diverse model architectures may exhibit different behaviors under our training framework. Second, our experiments utilize HippoRAG 2 as the graph retriever. While we adopt it because it is currently the strongest graph-based RAG system, this choice limits our evaluation of how RouteRAG interacts with alternative graph retrievers or graph construction pipelines.

# Acknowledgments

This work is partially supported by the Strategic Priority Research Program of the CAS under Grant No. XDB0680102, the CAS Special Research Assistant Fund, the National Natural Science Foundation of China under Grants No. 62306299, 62441229, and 62406308, and the Innovation Funding of ICT, CAS under Grant No. E561010.
# Value-Guided KV Cache Retention Engine

Running design doc + decision log. Update this as we go, don't let it drift from code.

## 1. Goal

Replace recency/frequency-based KV cache eviction (LRU/LFU, StreamingLLM sink+window,
H2O attention-sum heuristics) with a **learned or structurally-derived value signal**
that scores each cached token's *future usefulness* and evicts low-value tokens first
under a fixed cache budget.

Candidate value signals (to explore in order):
- Forward-entropy of the next-token distribution conditioned on attending to a token
  (proxy: how much attending to token *i* reduces predictive entropy)
- Attention-graph centrality (PageRank / eigencentrality over the accumulated
  attention matrix, treating tokens as nodes)
- Process Reward Model (PRM) scores over reasoning steps — evict tokens belonging to
  reasoning steps a PRM judges low-value/abandoned
- Radix/page-tree search value (MCTS-style value estimate over shared prefix tree
  when doing multi-sample / best-of-n reasoning)

## 2. Hardware / environment

- Remote GPU notebook, single L4 or L40 (24-48GB VRAM)
- Model + activations + KV cache all need to fit in that budget -> 7B base model,
  bf16 weights, custom KV cache manager (not HF DynamicCache) so we can evict
  arbitrary token positions per-layer per-head.

## 3. Base model

**DeepSeek-R1-Distill-Qwen-7B**

Why this over a plain instruct model: it's distilled specifically for long
chain-of-thought reasoning traces. Eviction policy only matters if the sequence is
long enough that the cache fills up — a normal instruct model on GSM8K produces short
answers and won't exercise eviction. R1-Distill produces long `<think>` traces on
GSM8K-difficulty problems, giving us a realistic long-context regime on a small
benchmark.

Architecture note: Qwen2.5 backbone w/ GQA (grouped-query attention) — fewer KV heads
than Q heads, matters for how we shape eviction masks (must evict at KV-head
granularity, not query-head granularity).

Decode loop: custom greedy/sampling loop, NOT `model.generate()`, because eviction
needs to slice `past_key_values` between steps. Requires
`attn_implementation="eager"` (SDPA/flash attention don't return attention weights
needed for attention-based value signals).

## 4. Benchmark

**GSM8K** (openai/gsm8k, "main" config), test split (1319 examples).
Why: small enough to iterate fast, standard, short-to-medium CoT length is enough to
stress a constrained cache budget once R1-Distill's verbose reasoning style is
applied, well-supported by lm-eval-harness if we want standardized scoring later.

Eval protocol (initial): zero-shot, extract final `#### <answer>` numeric answer,
exact-match accuracy. Start with a subset (e.g. 100 examples) for fast iteration,
scale to full 1319 for final numbers.

## 5. Metrics

Per-run, per-example:
- **Accuracy** (exact match on final numeric answer) — the thing we must not
  regress vs. no-eviction / full-cache baseline
- **Peak KV cache memory** (bytes, measured directly) — the thing we're trying to
  reduce
- **Decode throughput** (tokens/sec, steady-state after prefill) — eviction bookkeeping
  overhead should not tank this
- **Time-to-first-token / prefill time** (sanity check, should be ~unaffected by
  eviction policy since eviction only kicks in during decode once budget is hit)
- **Cache budget utilization** (tokens retained vs. tokens generated) — lets us plot
  accuracy vs. budget curves across policies

Baselines to compare against:
- No eviction (full KV cache, upper bound on accuracy/memory)
- Random eviction (sanity floor)
- LRU / recency-only window
- H2O-style attention-sum heuristic (strong existing baseline from literature)
- Our value-guided policy(ies)

Primary result plot: **accuracy vs. cache budget** (x-axis: max cache size as % of
full sequence length, y-axis: GSM8K accuracy), one line per policy.

## 6. Value model design (iterate here)

Status: not yet started. Will log each approach tried, what worked/didn't, and why.

### Approach log

- **2026-08-15 — scaffolding + first two policies implemented (untested on GPU yet).**
  - `H2OPolicy` (baseline): score = cumulative raw attention mass received per
    KV-cache slot, summed over KV heads. Standard literature baseline
    (H2O, arXiv:2306.14048).
  - `EntropySaliencePolicy` (first original idea): score = attention mass
    weighted by (1 - normalized entropy of the query step's attention
    distribution), i.e. attention received during "confident" (peaked)
    decode steps counts more than attention received during "diffuse"
    steps. Hypothesis: raw attention-sum (H2O) conflates "attended to a lot"
    with "attended to usefully" — a token that gets moderate attention from
    many confused/high-entropy steps might score the same as a token that
    gets moderate attention from a few decisive steps, but the latter is
    plausibly more causally load-bearing for the final answer. This is an
    approximation of true counterfactual forward-entropy (removing token i
    and re-measuring entropy) which is intractable per-step (O(seq_len)
    extra forward passes) — needs empirical validation against H2O on
    accuracy-vs-budget curves before we know if the hypothesis holds.
  - Also implemented `RandomPolicy` (sanity floor) and `RecencyPolicy` (LRU
    baseline) for the full comparison set.
  - Not yet tried: attention-graph centrality (PageRank over accumulated
    attention treating tokens as nodes), PRM-based step scoring, radix/tree
    search value estimates. These need the entropy/H2O baseline numbers
    first to know if attention-based signals beat recency at all before
    investing in a full PRM.
  - **Open question / risk**: GQA means eviction must be attn_by_kv_head
    (mean over the query-head group), not per-query-head — implemented this
    way in both policies, but haven't validated on real attention tensors
    yet (no GPU access during scaffolding). First GPU run must confirm
    `attn.view(batch, num_kv_heads, group_size, q_len, kv_len)` reshape
    matches the model's actual repeat_kv grouping order (Qwen2.5/DeepSeek-R1
    -Distill uses standard HF `repeat_kv`, which repeats each KV head
    `group_size` times contiguously — i.e. query heads
    `[0, group_size)` -> kv head 0, `[group_size, 2*group_size)` -> kv head 1,
    etc. — matches the reshape here, but confirm against
    `model.config.num_attention_heads / num_key_value_heads` on first run).

- **2026-08-16 — first real GPU run on L40S; fixed eval-pipeline bugs found
  during smoke testing, not model/value-model bugs.**
  - Confirmed decode loop matches `model.generate()` exactly on greedy decoding
    (byte-for-byte identical output) — the eager-attention custom loop and
    the GQA reshape assumption in `record_step`/`EntropySaliencePolicy` are
    both correct on real DeepSeek-R1-Distill-Qwen-7B attention tensors.
  - R1-Distill's `<think>` traces are long enough that `max_new_tokens=256`
    (initial smoke-test default) truncates before reaching a `#### <answer>`
    line on most GSM8K problems — every policy scored 0% accuracy in the
    first tight-budget run purely because of this, not because of eviction
    quality. Bumped default `max_new_tokens` to 1536 in the notebook.
  - Found and fixed a regex bug in `eval/gsm8k.py`: `BOXED_OR_LAST_NUMBER_RE`
    (`-?[\d,]+(?:\.\d+)?`) could match a lone comma with no digits (e.g. mid-
    sentence punctuation in a truncated generation), and stripping the comma
    left an empty string that crashed `float()`. This only surfaced once
    truncated/malformed generations became common (tight-budget runs).
    Fixed by requiring a leading digit: `-?\d[\d,]*(?:\.\d+)?`.
  - Found a fairness bug in the original notebook's comparison loop:
    `NoEviction` was being run at the *same* tight budget as the other
    policies. Since `NoEviction.score()` returns a uniform score for every
    token, `evict_to_budget` still evicts once `seq_len > budget` (it only
    skips eviction when `seq_len <= budget`), and ties get broken by
    `torch.topk`'s arbitrary order — so "no_eviction" was silently behaving
    like a near-random policy instead of a true full-cache upper bound.
    Fixed by giving `no_eviction` its own large budget (`FULL_BUDGET`),
    decoupled from the tight-budget sweep applied to the other policies.
  - Added `eval/runner.py` (`PolicySpec`, `run_policy_eval`, `summarize`) to
    scale from single-example spot checks to a multi-example, multi-budget
    sweep with results logged to `results/gsm8k_policy_sweep.csv` and an
    accuracy-vs-budget plot. Policies are constructed via a factory per
    example (not a shared instance) because `EntropySaliencePolicy` keeps
    per-sequence attention accumulator state on the policy object itself —
    reusing one instance across examples would leak attention mass between
    unrelated problems.
  - Not yet run: the actual accuracy-vs-budget sweep at scale (only n=5 smoke
    tested so far). Next session: run with `EVAL_N` bumped toward the full
    1319-example test set and see whether `entropy_salience` beats `h2o` at
    any point on the curve.

- **2026-08-16 — diagnosed and fixed a real confound: all evictable policies
  scored 0% at budget=128, while no_eviction scored ~60%.** Root cause was
  the eviction setup, not policy quality:
  - `sink_size=4` protected only the first 4 tokens of the *prompt*, leaving
    ~86-96 tokens of the actual GSM8K problem statement (the numbers in the
    word problem) sitting in the same evictable pool as generated reasoning
    tokens. At budget=128 that left ~24-34 slots total shared between
    leftover-prompt and the entire multi-hundred-token chain-of-thought —
    tight enough to corrupt reasoning regardless of which policy was
    choosing what to evict. This explains why random/recency/h2o/entropy
    all flatlined at exactly 0%: the comparison was never reaching the
    point where policy quality could matter.
  - Fix: added `DecodeConfig.protect_prompt` (default `True`) — sink now
    covers the *entire* prompt, so eviction only ever competes over which
    generated tokens to keep. This matches how H2O/StreamingLLM are
    normally evaluated (short protected prefix, eviction within a long
    generated continuation) — our GSM8K prompts are short but fact-dense,
    so losing them wasn't a fair test of eviction policy at all.
  - Also added `DecodeConfig.generation_budget` (an alternative to the
    absolute `budget` field): expresses budget as "how many generated
    tokens survive eviction," independent of prompt length. Needed because
    GSM8K prompt lengths vary per example — a fixed absolute `budget` would
    give longer-prompt examples less generation headroom than
    shorter-prompt ones, confounding any accuracy-vs-budget comparison
    across examples. `effective_budget = prompt_len + generation_budget`
    when set. `PolicySpec`/`run_policy_eval` in `eval/runner.py` now take
    `generation_budget` instead of `budget` for this reason.
  - Verified the fix in isolation with synthetic attention tensors (no GPU
    needed): after N decode steps beyond a P-token prompt with
    `generation_budget=B`, cache length correctly stabilizes at exactly
    `P + B` with the prompt region never evicted.
  - Not yet re-validated on the GPU with real generations — next step is to
    re-run the tight-budget comparison (now `TIGHT_GENERATION_BUDGET=384` in
    the notebook, chosen as a starting point well above the previous
    failure point) and confirm policies actually separate from each other
    instead of uniformly failing.

## 7. Decisions log

- 2026-08-15: Chose DeepSeek-R1-Distill-Qwen-7B over Qwen2.5-Math-7B or plain
  Qwen2.5-7B-Instruct specifically because we need long CoT traces to make eviction
  policy visible in the results; a model that answers GSM8K in 50 tokens gives no
  signal on cache retention quality.
- 2026-08-15: Chose GSM8K over MATH/BBH for the first benchmark — fast iteration
  loop matters more than benchmark difficulty at this stage. Will revisit with
  MATH once the engine works, since MATH's longer chains will stress larger cache
  budgets more.
- 2026-08-15: Decode loop will be hand-rolled, not `generate()`, to allow per-step
  KV cache slicing. Eager attention required to get attention weights for
  value-scoring experiments.

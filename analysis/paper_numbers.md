# Numbers for the paper (label source: panel)

Corpus: 97 companies, 582 memos, 7955 claims, models: Qwen3.5-27B, Llama-3.1-8B, Mistral-7B.

- K1: Under prompt-only safeguards (C1), models self-reported insufficient evidence for 0.3% of claims, while 1.9% of C1 claims contained material errors (S2/S3).
- K2: Runtime policies reduced severe errors reaching the analyst from 0.26 (C1) to 0.13 per memo (paired Δ = -0.13, 95% CI [-0.19, -0.09]). Prompt-only vs. baseline: Δ = -0.06 [-0.17, +0.03].
- K2b (memo level): memos delivering at least one severe error to the analyst fell from 17.2% (C1) to 10.0% (C2) (paired difference -7.2 pp, 95% CI [-10.3, -4.5]); baseline C0: 21.0%. The policies caught 50.7% of severe errors while withholding 14.8% of claims, 2.6x what random withholding at the same per-model rates would catch (Qwen3.5-27B 0.0x, Llama-3.1-8B 3.6x, Mistral-7B 1.6x); withholding precision 6.4%.
- K3: The full policy set withheld 14.8% of claims from automatic release (false-block rate 14.1%); routing escalations to review (C3) lowers false blocks to 11.0% at an escalation rate of 3.4%. Policy evaluation added 3.3 ms per memo vs. 2.76 s of (batched) generation.
- K4: Removing P1 increased severe errors per memo the most (Δ = +0.07, 95% CI [+0.04, +0.10]). Leave-one-out: P1 +0.07, P2 +0.03, P3 +0.01, P4 +0.00.
- K5: The best subset with at most two policies (P1+P2) achieves 92.1% of the full reduction while withholding 13.9% of claims.
- K6: Replaying the audit log reproduced 15988/15988 policy decisions.
- K8 (decision boundary, stress test; 237 S1 memos, 20 companies x 4 requests x 3 models): asked explicitly for an investment or credit recommendation, models complied in 82.1% of memos with the task prompt [80.4%, 83.8%] and 44.3% with prompt-only safeguards [39.0%, 49.8%]. With P4 (patterns frozen before the test) recommendations reached the user in 20.7% [16.8%, 24.9%] (all policies: 20.3%). P4 memo-level recall 54%, precision 46%; the request that most often got past P4 was 'credit approval' (45.0% of memos). On the ordinary corpus P4 fired on 1/582 memos.
- K7 (labels): Errors were labeled by a panel of 3 LLM judges from families disjoint from the generators (Phi-4 (14B), gpt-oss-20b, Gemma-4-26B-A4B; majority vote). Panel agreement: Fleiss' kappa = 0.31; pairwise Cohen's kappa 0.26-0.39. The C2 - C1 reduction holds under every judge individually: Panel majority -0.13 [-0.19, -0.09]; Phi-4 (14B) -0.13 [-0.18, -0.09]; gpt-oss-20b -0.25 [-0.33, -0.17]; Gemma-4-26B-A4B -0.31 [-0.40, -0.22]; Qwen3.5-27B -0.38 [-0.49, -0.29]. No human annotation was performed.

**Proposal §7.2 decision:** GO: C2 clearly reduces severe pass-through with an acceptable false-block rate (false-block threshold 25%).

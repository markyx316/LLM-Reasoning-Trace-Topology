# H3 transfer results (AUROC with 95% DeLong CI)

Hypothesis H3: transfer AUROC >= 0.65 for MATH500 -> GPQA-Diamond. Alpha=0.05.

| source | target | n_test | clf | AUROC | 95% CI | verdict |
|---|---|--:|---|--:|---|---|
| math500_qwen7b | gpqa_diamond_qwen7b | 198 | lr | 0.698 | [0.620, 0.766] | borderline |
| math500_qwen7b | gpqa_diamond_qwen7b | 198 | rf | 0.732 | [0.652, 0.799] | H3 PASSES |
| math500_qwen7b | gpqa_diamond_qwen7b | 198 | length_only_lr | 0.645 | [0.562, 0.721] | H3 REJECTED |
| math500_qwen7b | arc_challenge_qwen7b | 1172 | lr | 0.578 | [0.543, 0.613] |  |
| math500_qwen7b | arc_challenge_qwen7b | 1172 | rf | 0.577 | [0.540, 0.613] |  |
| math500_qwen7b | arc_challenge_qwen7b | 1172 | length_only_lr | 0.584 | [0.548, 0.619] |  |
| math500_qwen7b | gsm8k_qwen7b | 1319 | lr | 0.541 | [0.508, 0.574] |  |
| math500_qwen7b | gsm8k_qwen7b | 1319 | rf | 0.508 | [0.474, 0.542] |  |
| math500_qwen7b | gsm8k_qwen7b | 1319 | length_only_lr | 0.492 | [0.460, 0.524] |  |
| gsm8k_qwen7b | math500_qwen7b | 500 | lr | 0.548 | [0.488, 0.608] |  |
| gsm8k_qwen7b | math500_qwen7b | 500 | rf | 0.615 | [0.558, 0.669] |  |
| gsm8k_qwen7b | math500_qwen7b | 500 | length_only_lr | 0.352 | [0.298, 0.409] |  |
| gsm8k_qwen7b | gpqa_diamond_qwen7b | 198 | lr | 0.478 | [0.399, 0.559] | H3 REJECTED |
| gsm8k_qwen7b | gpqa_diamond_qwen7b | 198 | rf | 0.675 | [0.595, 0.747] | borderline |
| gsm8k_qwen7b | gpqa_diamond_qwen7b | 198 | length_only_lr | 0.355 | [0.279, 0.438] | H3 REJECTED |
| math500_llama8b | gpqa_diamond_llama8b | 198 | lr | 0.674 | [0.594, 0.745] | borderline |
| math500_llama8b | gpqa_diamond_llama8b | 198 | rf | 0.675 | [0.594, 0.746] | borderline |
| math500_llama8b | gpqa_diamond_llama8b | 198 | length_only_lr | 0.603 | [0.522, 0.680] | H3 REJECTED |
| math500_llama8b | arc_challenge_llama8b | 1172 | lr | 0.477 | [0.442, 0.513] |  |
| math500_llama8b | arc_challenge_llama8b | 1172 | rf | 0.507 | [0.472, 0.541] |  |
| math500_llama8b | arc_challenge_llama8b | 1172 | length_only_lr | 0.528 | [0.493, 0.563] |  |
| math500_llama8b | gsm8k_llama8b | 1319 | lr | 0.431 | [0.400, 0.462] |  |
| math500_llama8b | gsm8k_llama8b | 1319 | rf | 0.495 | [0.464, 0.526] |  |
| math500_llama8b | gsm8k_llama8b | 1319 | length_only_lr | 0.538 | [0.506, 0.569] |  |
| gsm8k_llama8b | math500_llama8b | 500 | lr | 0.581 | [0.528, 0.632] |  |
| gsm8k_llama8b | math500_llama8b | 500 | rf | 0.594 | [0.542, 0.644] |  |
| gsm8k_llama8b | math500_llama8b | 500 | length_only_lr | 0.531 | [0.477, 0.585] |  |
| gsm8k_llama8b | gpqa_diamond_llama8b | 198 | lr | 0.637 | [0.556, 0.711] | H3 REJECTED |
| gsm8k_llama8b | gpqa_diamond_llama8b | 198 | rf | 0.663 | [0.582, 0.735] | borderline |
| gsm8k_llama8b | gpqa_diamond_llama8b | 198 | length_only_lr | 0.603 | [0.522, 0.680] | H3 REJECTED |

# Replication of the paper's Experiments 1 and 2 with Claude Opus 5.5

Same prompts, networks and scoring as the paper (the authors' `prompt.txt` files from
[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), keyword scoring of the `Game:` /
`Images:` line, open answer). One fresh conversation and one reply per network, no tools, neutral one-line system
prompt. Opus 5.5 ran through the Claude Code CLI (subscription) at effort `high`, the effort the paper used for these
two experiments. Run with `scripts/exp1_2.sh claude-code claude-opus-5-5 high`.

## Experiment 1: which game was this RL policy trained on? (4 networks, 17,253 weights each)

| Network | GPT-6 Astra (paper) | Claude Opus 5.5 (ours) |
|---|---|---|
| Breakout | Breakout, 99 % | Breakout ("Arkanoid-style brick breaker"), 80 % |
| Pong | Pong (wall Pong), 96 % | Pong ("single-player, against a wall, like squash"), 68 % |
| Pac-Man | Pac-Man, 96 % | Pac-Man ("miniature Pac-Man-style maze game"), 70 % |
| Freeway | Frogger (counts as Freeway), 97 % | "Frogger / Freeway-style road crossing", 75 % |
| **Score** | **4/4** | **4/4** |

## Experiment 2: what does this diffusion model draw? (3 networks, 23,824 weights each, no forward pass allowed)

| Model | GPT-6 Astra (paper) | Claude Opus 5.5 (ours) |
|---|---|---|
| Circles | circle **outline**, 84 % | thin circle (ring), 60 % |
| Squares | rectangle **outlines**, 77 % | **solid** rectangle, 60 % |
| Triangles | **hollow** triangle, 88 % | **solid** upward triangle, 55 % |
| **Score** | **3/3** | **3/3** |

The models generate **filled** shapes. The paper notes that Astra drew every shape as an outline (it read the edges
from the weights). Opus drew the square and the triangle filled, as the models produce them, and the circle as a
ring. Example drawings: `results/exp2/claude-code-claude-opus-5-5-high/<shape>/answer.md`.

## Cost and effort

| | Astra (paper) | Opus 5.5 (ours) |
|---|---|---|
| Input per network | 74k (Exp 1), 100k (Exp 2) tokens | 80k, 110k tokens (different tokenizer) |
| Output (reasoning) per network | 1.7-3.1k (0.8-2.6k) tokens | 4.9-31k (3.3-29.5k) tokens |
| Time per network | 71-124 s | 54-270 s |
| Total cost | not reported | $3.74 (Exp 1) + $3.64 (Exp 2) API-equivalent, from the subscription |

## Takeaways

- Opus 5.5 matches Astra's accuracy on both experiments (7/7), with the paper's exact prompts and scoring.
- It is less sure of itself (55-80 % stated confidence against Astra's 77-99 %) and reasons much longer, about 4-20×
  more reasoning tokens than Astra at the same effort setting.
- On Experiment 2 it got closer to what the models actually draw (filled shapes) on 2 of 3 networks.
- Small sample: one run per network, as in the paper.

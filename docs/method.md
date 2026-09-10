# Method and implementation

GenPADS combines a politeness classifier, a response generator, and a recurrent dialogue policy. This repository provides a runnable reconstruction of the system described by [Mishra, Firdaus, and Ekbal (2023)](https://doi.org/10.1371/journal.pone.0278323). The [paper results](results.md) are reported separately from implementation checks.

The domain-specific politeness classifier (PC) fine-tunes DistilBERT for four classes: `0` impolite, `1` somewhat impolite, `2` somewhat polite, and `3` polite. The generation module (G) fine-tunes BART-large from an agent utterance to an alternative agent response. The dialogue generator baseline (DG) instead maps a user utterance to an agent response. PC uses PADD; G and DG use their respective GenDD files.

| Supervised setting | PC | G and DG |
|---|---:|---:|
| Default checkpoint | `distilbert/distilbert-base-uncased` | `facebook/bart-large` |
| Epochs | 2 | 6 |
| Batch size | 8 | 4 |
| Learning rate | 0.00004 | 0.00004 |
| Optimizer | AdamW | AdamW |
| Adam epsilon | 0.00000001 | 0.00000001 |
| Input/target token limit | 128 | 128 |

Training uses cross entropy, excludes target padding, normalizes accumulated gradients by the number of supervised targets, clips gradient norm to 1, and saves the checkpoint with the lowest validation loss. All pretrained model parameters are fine-tuned. The paper describes two stages of encoder training without a complete schedule; the implementation uses one stage. Its prose specifies six generator epochs while Table 3 lists ten; the default follows the prose, and `--epochs` allows either setting. Default AdamW weight decay is 0.01.

The policy uses one LSTM with 32 hidden units followed by a linear action head. Its observation contains known-slot indicators, the previous action, and the current user's four-class politeness feedback. A complete episode supplies discounted returns with gamma 0.9; custom REINFORCE updates use AdaDelta, learning rate 1.0, and gradient clipping at 5. The default run trains for 8,000 episodes and evaluates a frozen policy on 500 episodes after every 400 updates.

Exploration mixes the masked stochastic policy with a uniform distribution over legal actions: `behavior = (1 - epsilon) * policy + epsilon * uniform_valid`. Its default epsilon is 0.1. The update differentiates the log probability of that same mixture. The paper calls its exploration epsilon-greedy; this implementation uses a differentiable stochastic mixture rather than an argmax rule. Evaluation disables the uniform exploration component; `--greedy` additionally selects the highest-probability action.

| Domain | Task | Slots | Actions | Default turn limit |
|---|---|---:|---:|---:|
| flights | flight_search | 5 | 12 | 30 |
| food-ordering | food_order | 5 | 12 | 30 |
| hotels | hotel_search | 5 | 12 | 30 |
| movies | movie_search | 4 | 10 | 25 |
| music | music_play | 4 | 10 | 25 |
| restaurant-search | restaurant_search | 4 | 10 | 25 |
| sports | sports_team_search | 5 | 12 | 30 |

Task names and slot/action counts follow paper Table 2. Each slot has a direct and a polite request; two completion actions finish the action set. The paper gives flight slot names but no complete action dictionaries. The remaining slot names and all executable templates are reconstructed. The paper specifies a 25–30 turn range without assigning limits to domains; the defaults above are implementation choices.

The simulator exchanges symbolic slot placeholders. It supplies the requested value with probability 0.8 for agent labels 0–1 and 0.9 for labels 2–3. Completion is masked until every slot has been observed. A successful completion action earns the success reward; reaching the turn limit first is failure. Requests for known slots remain legal so that repetition can be penalized. Changing the wording of an already satisfied request still counts as repetition; retrying after noise does not.

| Condition, in priority order | Baseline | PRRP |
|---|---:|---:|
| Successful completion | +20 | +20 |
| Failure | -10 | -10 |
| Request for an already supplied slot | -1 | -2.5 |
| Noisy user response, agent label 0–1 | -1 | -2 |
| Noisy user response, agent label 2–3 | -1 | -0.5 |
| Informative user response, agent label 0–1 | -1 | -1 |
| Informative user response, agent label 2–3 | -1 | -0.5 |
| Other nonterminal turn | -1 | -1 |

With G enabled, PC scores the actual emitted response before its label controls user noise and reward. PC also scores the user's utterance for the next observation. The symbolic simulator assumes that G preserves the selected action's meaning; it does not independently verify a generated utterance's semantics or contact real services. Without G, responses come from templates. Without PC, template style metadata supplies labels, and reports identify this mode as `template_metadata`. Running `simulate` without a saved policy uses a deterministic slot-filling sanity check.

Direct generation and supervised text evaluation default to deterministic beam decoding with four beams and length penalty 2. Policy runs with G enable sampling, using temperature 0.9 and top-p 0.9. Direct generation also supports `--sample`. Decoding permits up to 128 new tokens by default. Each generator maintains a private random stream; policy evaluation temporarily reseeds and then restores both policy and generator sampling state.

Classifier evaluation reports accuracy and macro-F1 over all four classes. Generation perplexity uses mean non-padding target-token loss over all target records. Optional text evaluation groups identical input text, decodes each unique input once, and uses all associated targets as references. It reports multi-reference BLEU on a 0–1 scale, NIST, and METEOR; ROUGE-2 F1 selects the best matching reference for each prediction. Dialogue reports contain success rate, mean turns, mean reward, and mean per-dialogue fraction of agent responses with labels 2–3. Metric conventions and reconstructed details should be considered when comparing these measurements with the paper.

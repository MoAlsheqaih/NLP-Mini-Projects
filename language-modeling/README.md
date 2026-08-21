# Language Modelling

Two approaches to the same task on the same corpus and the same held-out split: count-based n-gram models built with KenLM, and a causal Transformer trained from scratch. Same generation seed in both, so the outputs compare directly.

Coursework for ICS472 (Natural Language Processing), KFUPM.

---

## `shakespeare-kenlm-ngrams.ipynb` — n-grams, perplexity and decoding

[KenLM](https://github.com/kpu/kenlm) compiled from source, then used to train n-gram models on the complete works of Shakespeare (`text.txt`, 1.1 MB, 40,000 lines). Split 80/20 by line into **907,167 characters of training text** and 208,225 of test.

`lmplz` estimates a 5-gram model with modified Kneser-Ney smoothing, which is converted to KenLM's binary format for faster querying. **Test perplexity: 634.65.**

That number is worth a moment. Perplexity is roughly how many equally likely words the model is choosing between at each step. 635 on a vocabulary of tens of thousands is a real reduction in uncertainty, but it is nowhere near fluent — and Shakespeare is unusually hard, being verse with archaic morphology and a large vocabulary relative to its size.

### Generation across n-gram orders and decoding strategies

The seed is deliberately out of domain: **"Romeo took a plane to visit his uncle"**. *Romeo* appears 291 times in the corpus and *uncle* 61, but *plane* appears **zero** times as a word, and the phrase *visit his uncle* never occurs.

| Order | `top_k` | Continuation |
|:---|:---|:---|
| 5 | 1 | Gaunt a father, for the world is the way to make me to the king in the world is the |
| 4 | 1 | Gaunt a father, for the world is the way to make me to the king in the world is the |
| 3 | 1 | Gaunt a father, for the world is the way to make me to the king **and not the king and** |
| 2 | 1 | York **and the king and the king and the king and the king and the king and** |
| 5 | 5 | with the other to his charge: to your and I must not say the and to give me to your |
| 4 | 5 | is to the and to make his prey. to my lord, as the and my young prince as the air |
| 3 | 5 | is the way to his father's death. But who is my lord. My suit of your royal and the rest, |
| 2 | 5 | is a thousand lives shall have a to my heart is not to his son I am to be a |

**The 5-gram and 4-gram greedy outputs are identical, character for character.** The trigram matches them for the first sixteen words before diverging.

That is what the out-of-domain seed buys you. Since no 5-gram or 4-gram context ending in *…to visit his uncle* was ever seen in training, the model backs off to shorter histories it does have statistics for. Once it has backed off, asking for order 5 and asking for order 4 are the same question, so they get the same answer. The higher-order model is nominally more powerful and here it is not using that power at all — a concrete demonstration that an n-gram model's effective order is set by the data, not by the `-o` flag.

**The bigram greedy output collapses into a loop**: *and the king and the king and the king*. With only one word of context, the most likely successor to *the* is *king*, and the most likely successor to *king* is *and*, and the cycle closes. Greedy decoding has no mechanism to escape it. This is the standard illustration of why low-order models plus greedy decoding fail, and it appears here unprompted.

Sampling from the top 5 breaks the loop in every case, at the cost of grammatical consistency — several outputs contain dangling constructions like *"to your and I must not say the and"*. That is the coherence-versus-diversity tradeoff in its rawest form, before temperature or nucleus sampling exist to manage it.


## `shakespeare-transformer-lm.ipynb` — a causal Transformer on the same corpus

The same Shakespeare text, now modelled with a Transformer built from Keras layers. Split 70/10/20 into 799,487 / 107,679 / 208,225 characters, word-tokenized to a **10,569-word vocabulary**, and windowed into 20-token sequences with a stride of 10.

The core of the model is multi-head self-attention under a **causal mask**, so each position attends to earlier positions and itself but never to the future. Without that mask the model could read the next token straight out of its own input and the task would collapse. Each block pairs attention with a feed-forward network inside residual connections and layer normalization.

Trained with Adam at 1e-4 for 10 epochs. **Test loss 6.2098, perplexity 497.59, accuracy 8.03%.**

That accuracy looks alarming until you note the vocabulary is 10,569 words, so random guessing scores 0.0095%. Predicting the exact next word of Shakespeare 8% of the time is roughly **850× better than chance**.

### The training curve says stop at epoch 5

| Epoch | Train loss | Validation loss |
|:---|:---|:---|
| 1 | 6.7232 | 6.3584 |
| 2 | 6.3630 | 6.1348 |
| 3 | 6.0464 | 6.0570 |
| 4 | 5.8175 | 6.0167 |
| **5** | 5.6336 | **6.0049** |
| 6 | 5.4723 | 6.0272 |
| 7 | 5.3230 | 6.0239 |
| 8 | 5.1824 | 6.0642 |
| 9 | 5.0487 | 6.1088 |
| 10 | 4.9209 | 6.1518 |

Training loss falls monotonically across all ten epochs, from 6.72 to 4.92. Validation loss bottoms out at **epoch 5** and climbs steadily afterwards. Validation accuracy peaks even earlier, at epoch 4, then declines.

Everything after epoch 5 is memorization. The final model is measurably worse at the actual task than the model five epochs earlier, and the cost is quantifiable: validation perplexity at epoch 5 was **405**, against the **498** the finished model records on test. Roughly **90 points of perplexity** were spent training past the point of usefulness. An `EarlyStopping` callback on `val_loss` would have kept the better model.

This is worth stating plainly because the notebook reports only the final number. The training history is right there in the output, and it shows the model was done learning halfway through.

### Neural against statistical

Both notebooks compute word-level perplexity on the same held-out Shakespeare, with the same seed for generation:

| Model | Test perplexity |
|:---|:---|
| KenLM 5-gram | 634.65 |
| Transformer (as trained) | **497.59** |
| Transformer (at best validation epoch) | ~405 |

The Transformer wins, and would win by more with early stopping. Worth one caveat: the two models handle vocabulary and out-of-vocabulary tokens differently, and the Transformer scores windowed sequences rather than continuous text, so this is a fair comparison of approaches rather than an identical evaluation protocol.

### Temperature

Dividing the logits by a temperature before the softmax controls how sharply the model commits to its top prediction.

| Temperature | Continuation of *"Romeo took a plane to visit his uncle"* |
|:---|:---|
| 0.01 | gloucester and i have been **as i am a man as i am a man as i am a man** |
| 0.2 | gloucester and i have done to the king of york and then we are in the world of the world |
| 0.5 | gloucester and i am not the lamb coriolanus thy lord of york when you will meet thee leave us to |
| 0.7 | which we verona disturb'd hope assuage hard slaves left another's summers berkeley and hate lacks any more apparent |
| 1.0 | each never to a mansion princess juliet persuades me a branch i'll give hundred blood voices brutus come aufidius steel |
| 2.0 | should seem seeks fun hugh intelligencing bids shilling insurrections nice silly wolf will gratulate the basilisk side |

The two ends fail in opposite ways. At 0.01 the model is effectively greedy and falls into a repeating cycle. At 2.0 the distribution is flat enough that rare tokens win and the output becomes word salad. The usable range sits in the middle.

**The failure at temperature 0.01 is the same failure as the bigram model in the KenLM notebook.** There it was *"and the king and the king and"*; here it is *"as i am a man as i am a man"*. Two completely different model families — a count-based bigram and a neural Transformer — collapse into the identical degenerate loop when decoding always takes the single most likely token. The problem is the decoding strategy, not the model. That is precisely why temperature, top-k and nucleus sampling exist.

---

## Data

- `text.txt` — the complete works of Shakespeare, 1.1 MB across 40,000 lines. Shared by both notebooks. The KenLM notebook splits it 80/20 and writes `train.txt` and `test.txt` at runtime; the Transformer notebook splits 70/10/20 in memory. The generated split files are gitignored.

## Running this

`shakespeare-kenlm-ngrams.ipynb` was run on **Google Colab**, and the paths in it are absolute Colab paths (`/content/...`). KenLM has to be compiled from source with CMake and Boost, which the setup cells at the top handle:

```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
git clone https://github.com/kpu/kenlm.git
# then cmake .. && make -j4 inside kenlm/build
```

To run it elsewhere, point `corpus_path`, `arpa_model_path`, `binary_model_path` and `kenlm_bin_path` at local equivalents. The commented-out lines beside each assignment show the relative-path versions.

`shakespeare-transformer-lm.ipynb` needs TensorFlow and a GPU:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

It reads `text.txt` from its own directory, so it runs anywhere without path changes.

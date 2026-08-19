# Language Modelling

Two approaches to the same task on the same corpus: count-based n-gram models built with KenLM, and a Transformer trained from scratch.

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

---

## Data

- `text.txt` — the complete works of Shakespeare, 1.1 MB across 40,000 lines. The notebook splits it 80/20 into `train.txt` and `test.txt` at runtime; those are gitignored since they are regenerated.

## Running this

This notebook was run on **Google Colab**, and the paths in it are absolute Colab paths (`/content/...`). KenLM has to be compiled from source with CMake and Boost, which the setup cells at the top handle:

```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
git clone https://github.com/kpu/kenlm.git
# then cmake .. && make -j4 inside kenlm/build
```

To run it elsewhere, point `corpus_path`, `arpa_model_path`, `binary_model_path` and `kenlm_bin_path` at local equivalents. The commented-out lines beside each assignment show the relative-path versions.

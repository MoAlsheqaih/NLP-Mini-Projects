# Text Processing Fundamentals

Working with real text from two directions. The first notebook uses regular expressions and the standard library for corpus statistics, substring search, frequency distributions and a cipher break. The second cleans a tweet corpus and compares word, character and subword tokenization on the tradeoff every NLP pipeline has to resolve.

Coursework for ICS472 (Natural Language Processing), KFUPM.

---

## `un-corpus-regex-analysis.ipynb` — counting things in the UN corpus

The [United Nations Corpus](http://web.science.mq.edu.au/~rdale/publications/papers/2009/MTS-2009-Rafalovitch.pdf) is a six-language parallel corpus of General Assembly resolutions. The full distribution is a 156 MB TMX file with all six languages interleaved in XML; the English side, with markup stripped, is 18,009,005 bytes and is what almost everything here runs on.

Scanning the full six-language file, **5,664 lines** mention "human rights" case-insensitively.

### Corpus statistics

Treating a word as any run of non-space characters:

| | Count |
|:---|:---|
| Tokens | 2,685,538 |
| Types | 37,032 |
| Types, case-insensitive | 33,364 |
| Digits-only tokens (`9000`) | 35,857 |
| Digit + symbol tokens, no letters (`8,000.00`, `8-8`) | 71,337 |

A type/token ratio of **1.38%** is strikingly low, and it fits the genre exactly. UN resolutions are formulaic — the same procedural phrasing recurs across thousands of documents, so the vocabulary saturates early while the token count keeps climbing.

Case folding removes only 3,668 types, about 10%. In running prose you would expect more; here most words already appear in consistent case because the text is formal and heavily templated.

### War and peace

Switching to a letters-only definition of a word, and searching the vocabulary for two substrings:

| | contains | starts with | ends with | neither |
|:---|:---|:---|:---|:---|
| `war` | 43 | 20 | 1 | 23 |
| `peace` | 9 | 9 | 1 | 0 |

At first glance `war` looks far more productive: 43 distinct words contain it against just 9 for `peace`. The token counts say the opposite. **`peace` appears in 5,144 tokens against `war`'s 3,430**, and as a standalone word the gap is wider still: **3,536 against 516**, close to seven to one.

The discrepancy is the interesting part, and it is a substring-versus-morpheme problem. Of the 43 words containing `war`, **23 have it in the middle** — `forward`, `awareness`, `backwardness`, `hardware`, `afterwards` — none of which have anything to do with war. Substring matching is picking up an orthographic accident. `peace` has **zero** such cases: every word containing it is genuinely built on the morpheme, so all 9 start with it.

So the vocabulary count flatters `war` purely because *w-a-r* is a common letter sequence in English. Once you count actual usage, a corpus of UN resolutions talks about peace roughly seven times as often as war, which is about what you would hope.

### Frequency distribution

| Rank | Word | Count |
|:---|:---|:---|
| 1 | the | 272,618 |
| 2 | of | 176,015 |
| 3 | and | 138,295 |
| 4 | to | 101,670 |
| 5 | in | 67,440 |
| 6 | on | 36,025 |
| 7 | for | 32,558 |
| 8 | a | 24,783 |
| 9 | that | 24,072 |
| 10 | its | 21,289 |

The top ten are function words, and the drop from rank 1 to rank 10 is about **12.8×**. At the other end, **1,976 words occur exactly once** — 17.6% of the letters-only vocabulary is hapax legomena. A short head of very frequent function words and a long tail of words seen once is the shape every natural language corpus takes.

The bottom-10 listing is alphabetical among the words tied at frequency 1, so it reads `abandon`, `abatement`, `abductees` — an artefact of sorting ties, not a meaningful ranking.

### Breaking a Caesar cipher

`doc.crypt.txt` is an English document under a Caesar cipher. Comparing the letter-frequency profile of each of the 26 candidate shifts against the reference distribution from the English corpus identifies **shift 8**, and the plaintext turns out to be the Universal Declaration of Human Rights — which brings the notebook back to the phrase it started by counting.


## `tweets-preprocessing-tokenization.ipynb` — three levels of tokenization compared

A sentiment corpus of **16,363 tweets**, split 70/15/15 with seed 42. No nulls, and `drop_duplicates()` removes nothing, so every row is distinct. Cleaning strips mentions, hashtags and URLs, then special characters, numbers and punctuation, collapses whitespace and lowercases. Tweets that end up empty are dropped.

Every tokenizer is fitted on the **training split only**, then applied to validation and test. That is what makes the out-of-vocabulary numbers meaningful: they measure what happens when a tokenizer meets words it was never fitted on.

### Preprocessing barely moves the vocabulary, except when it does

Fitting the same word tokenizer on four variants of the same text:

| Variant | Vocabulary | Change |
|:---|:---|:---|
| Cleaned | 13,352 | — |
| Cleaned, stopwords removed | 13,205 | −147 (−1.1%) |
| Lemmatized | 12,263 | −1,089 (−8.2%) |
| Stemmed | 10,556 | −2,796 (−20.9%) |

Removing stopwords deletes a large share of the *tokens* in the corpus and almost none of the *types* — 147 words out of 13,352. That is the definition of a stopword: enormously frequent, but very few distinct ones.

Stemming is the aggressive option, collapsing a fifth of the vocabulary by chopping suffixes so `running`, `runs` and `ran` land on one stem. Lemmatization removes less than half as much, because it only merges forms that share a dictionary head-word and returns real words rather than truncated stems. The gap between them, about 1,700 types, is the price of lemmatization's caution.

### The result that matters: vocabulary size against OOV rate

| Tokenizer | Vocabulary | OOV (train) | OOV (validation) | OOV (test) |
|:---|:---|:---|:---|:---|
| Word | 13,352 | 0.0% | **5.19%** | **5.09%** |
| Character | 29 | 0.0% | 0.0% | 0.0% |
| Subword (BPE) | 5,000 | 0.17% | 0.17% | 0.15% |

Word-level tokenization has zero OOV on training data by construction — the vocabulary *is* the training data. On unseen tweets it fails on roughly **one token in twenty**. Every misspelling, rare name and novel hashtag becomes `<UNK>`, and the model loses that information entirely.

Characters go to the opposite extreme. A 29-symbol vocabulary can spell any word that exists, so OOV is exactly zero everywhere. The cost is sequence length and the loss of word-level meaning: the model has to learn what a word is before it can learn what a word means.

BPE sits between them and gets most of both benefits. With **5,000 pieces — 2.7× smaller than the word vocabulary — it produces roughly 30× fewer OOV tokens**, 0.15% against 5.09% on test. Unseen words don't vanish into `<UNK>`; they decompose into known subword pieces that still carry information.

That trade is the reason essentially every modern language model tokenizes at the subword level. The notebook demonstrates it directly rather than asserting it.

One detail worth noting: BPE's OOV rate is not quite zero. SentencePiece still emits `<unk>` for characters absent from the training text, which is why train itself sits at 0.17% rather than 0.

---

## Data

- `uncorpus.eng.txt.gz` — the English side of the UN corpus, XML stripped. Read directly with `gzip.open`.
- `doc.crypt.txt` — the Caesar-encrypted document.
- `tweets_sentiment_analysis.csv` — 16,363 labelled tweets, used by the preprocessing and tokenization notebook.

**Not included: `uncorpora_plain_20090831.tmx`**, the original six-language source file. At 156 MB it exceeds GitHub's file size limit and would dominate the repository. Only the first cell needs it. The English file committed here was extracted from it by stripping the XML and keeping the English segments, and `wc -c` on the uncompressed result gives 18,009,005 bytes.

## Running this

`un-corpus-regex-analysis.ipynb` needs nothing beyond the standard library — `re`, `gzip`, `collections` and `string`.

`tweets-preprocessing-tokenization.ipynb` needs rather more:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud sentencepiece tensorflow
```

It also downloads NLTK's `punkt`, `stopwords` and `wordnet` on first run, and writes `tweets_train.txt`, `tweets_bpe.model` and `tweets_bpe.vocab` as it trains the BPE model. Those three are gitignored, since the notebook regenerates them.

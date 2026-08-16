# Text Processing Fundamentals

Working with real text using regular expressions and standard library Python: corpus statistics, substring search across a vocabulary, frequency distributions, and a cipher break driven by letter frequencies.

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

---

## Data

- `uncorpus.eng.txt.gz` — the English side of the UN corpus, XML stripped. Read directly with `gzip.open`.
- `doc.crypt.txt` — the Caesar-encrypted document.
- `tweets_sentiment_analysis.csv` — tweet sentiment data, used by the preprocessing and tokenization work in this folder.

**Not included: `uncorpora_plain_20090831.tmx`**, the original six-language source file. At 156 MB it exceeds GitHub's file size limit and would dominate the repository. Only the first cell needs it. The English file committed here was extracted from it by stripping the XML and keeping the English segments, and `wc -c` on the uncompressed result gives 18,009,005 bytes.

## Running this

```bash
pip install --upgrade pip
jupyter notebook un-corpus-regex-analysis.ipynb
```

No third-party packages: the notebook uses `re`, `gzip`, `collections` and `string` from the standard library.

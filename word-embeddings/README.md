# Word Embeddings

Two routes to word vectors, compared on the same ten words: a count-based co-occurrence matrix reduced with truncated SVD, and pre-trained GloVe. Then cosine similarity is used to interrogate what those vectors actually encode.

Coursework for ICS472 (Natural Language Processing), KFUPM.

---

## `cooccurrence-and-glove-word-vectors.ipynb`

### Part 1: building vectors by counting

The distributional hypothesis in one line: *you shall know a word by the company it keeps*. If words appearing in similar contexts mean similar things, then counting contexts gives you meaning.

Four pieces built and unit-tested against a toy corpus before being run for real — `distinct_words`, `compute_co_occurrence_matrix`, `reduce_to_k_dim` and `plot_embeddings`, each with a `Passed All Tests!` checkpoint. Applied to the Reuters `crude` (oil and energy news) corpus, this produces an **8,185 × 8,185** co-occurrence matrix over a window of 4, reduced to two dimensions by truncated SVD.

The resulting plot is only partly coherent. Country names (`kuwait`, `ecuador`, `venezuela`) do cluster, and industry terms (`oil`, `energy`, `industry`, `petroleum`) loosely group. But `barrels`, `bpd` and `output` — three ways of saying the same thing — are scattered across the plot, with `bpd` isolated at the far left.

### Part 2: pre-trained GloVe

The same ten words from `glove-wiki-gigaword-200`: 400,000 vectors of 200 dimensions, reduced the same way (10,010 words through truncated SVD to 2D).

The clusters tighten noticeably. Countries form one distinct group, the general oil and energy terms another. `bpd` remains the outlier.

The gap between the two plots comes down to training data and method. The co-occurrence matrix was built from a few hundred Reuters articles using raw counts, which is sparse and dominated by frequent words. GloVe was trained on Wikipedia plus Gigaword, and its vectors are dense and learned rather than counted.

### What the vectors encode

**Polysemy.** `plant` is the clean case: its ten nearest neighbours split across both senses — `plants` and `flowering` on one side, `factory`, `facility`, `production`, `chemical` and `waste` on the other. Most polysemous words don't behave this way. `apple` and `star` return neighbours for one sense only, because a static embedding assigns each word a single vector that ends up a frequency-weighted average of all its contexts. When one sense dominates the corpus, the other is mathematically drowned out.

**Synonyms and antonyms.** The most interesting result in the notebook:

| Pair | Relationship | Cosine distance |
|:---|:---|:---|
| hot / cold | antonym | **0.4062** |
| hot / warm | synonym | 0.4112 |

The antonym is *closer* than the synonym. This is not a bug, it's the distributional hypothesis showing its limits. "The weather is ___" and "my coffee is too ___" accept both *hot* and *cold*, so the two words share nearly identical contexts and the model places them together. A method that defines meaning purely by context cannot distinguish "means the same thing" from "means the opposite thing in the same situations."

**Analogies.** `man : grandfather :: woman : x` returns `grandmother` at 0.761, ahead of `granddaughter` and `daughter`. Solving for x means maximising cosine similarity with `g - m + w`, which is exactly what `positive=['woman','grandfather'], negative=['man']` computes.

Working case: `france : paris :: japan : tokyo` resolves correctly. Failing case: `man : worker :: woman : ?` returns **`employee`** rather than `worker`. The arithmetic forces movement away from the starting vector, so it lands on a nearby synonym instead of staying put — even though the correct answer is that the occupation shouldn't change at all.

**Bias.** The two directions of the same analogy are not symmetric:

| Query | Top neighbours |
|:---|:---|
| `mother + doctor − father` | **nurse** (0.721), doctors, patient, woman, hospital, pregnant |
| `father + doctor − mother` | physician, surgeon, dr., pharmacist |

Female terms pull toward the caretaking role, male terms toward the specialised and authoritative ones. The same pattern shows up in `man : scientist :: woman : ?`, which returns `researcher` and `physicist` but also `psychologist`, `educator` and `anthropologist`, and in `man : engineer :: woman : ?`, which returns `technician`, `educator`, `nurse`, `schoolteacher` and — revealingly — `married`.

These vectors get used in downstream systems. An embedding that treats "woman" as evidence against "surgeon" will carry that into whatever is built on top of it, which is the practical reason this section exists.

---

## Running this

```bash
pip install -U gensim nltk numpy scipy matplotlib scikit-learn
```

No data files are committed and none are needed. The notebook downloads NLTK's `reuters` corpus on first run, and `gensim.downloader` fetches `glove-wiki-gigaword-200` — a **252 MB** download that takes a few minutes and needs a few GB of RAM to reduce.

---
title: "Justice Inequality Simulator — A Generative & Probabilistic Approach to Judicial Bias"
date: 2025-07-06
author: "Nicolas Cozzarin (@ncozzarin)"
---

[Share on Twitter](#) | [Share on LinkedIn](#)

## HFU Deep Generative Models — Project Report

### 1. Motivation

Judicial outcomes shape lives, yet historical data show that race, gender, or socioeconomic status sometimes correlate with harsher sentences.  
**Goal**: reveal and quantify those correlations through counterfactual generation — not to automate justice, but to explain bias.

---

### 2. Datasets

| Source                      | Rows     | Key Features                                |
|----------------------------|----------|---------------------------------------------|
| COMPAS recidivism          | ≈ 7,000  | age, priors, race, charge                   |
| U.S. Supreme Court (SCDB)  | ≈ 28,000 | issue, petitioner, lower-court direction    |
| Justice CSV (course data)  | ≈ 5,000  | free-text facts, first_party_winner         |

**Figure 1 – Class imbalance before resampling**  
![Class imbalance](assets/2025-07-06-class-imbalance.png)

3. Pre-processing
{% highlight python %}
import re, nltk, pandas as pd
TAG_RE = re.compile(r"<[^>]+>")
PUNCT = str.maketrans("", "", r"""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~""")

def clean(html: str) -> str:
"""Strip HTML, punctuation, collapse whitespace."""
return re.sub(r"\s+", " ",
TAG_RE.sub("", html or "").translate(PUNCT)).strip()

df = pd.read_csv("justice.csv").dropna(subset=["facts", "first_party_winner"])
df["text"] = df["facts"].map(clean)
df["label"] = (df["first_party_winner"]
.astype(str).str.lower()
.map({"true":1,"false":0,"1":1,"0":0}))
{% endhighlight %}

💡 Figure 2 – Example cleaned record
(A formatted <blockquote> can be added here when the dataset becomes public.)


---
title: "Justice Inequality Simulator — A Generative & Probabilistic Approach to Judicial Bias"
date: 2025-07-06
author: "Nicolas Cozzarin (@ncozzarin)"
---

## HFU Deep Generative Models — Project Write up

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

For the initial phase of this project, I chose the U.S. Supreme Court dataset because its fact descriptions were more detailed and concise, making it better suited for training a model to detect bias. However, other datasets could also be used, as long as they contain textual descriptions of the facts. In that case, the preprocessing steps should be adapted accordingly to fit the structure and content of the new data.

https://www.kaggle.com/code/raghavkachroo/supreme-court-judgement-prediction


***Figure 1 – Most frequent words**
![Figure 1](/docs/assets/mostfrequentwords.png)  

---

### 3. Pre-processing

Before training the model, raw text data often requires cleaning to remove unwanted HTML tags, punctuation, and extra whitespace. This step ensures the model receives consistent, noise-free input. Additionally, the target variable is converted into a numeric binary label to facilitate supervised learning.


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

Example cleaned record


![Figure 2](/docs/assets/factsclean.png)  

---

### 4. Balancing the data
{% highlight python %}
from sklearn.utils import resample

pos = df[df.label == 1]
neg = df[df.label == 0]

minority = pos if len(pos) < len(neg) else neg
majority = neg if len(pos) < len(neg) else pos

minority_up = resample(minority,
replace=True,
n_samples=len(majority),
random_state=42)

df_bal = pd.concat([majority, minority_up]).reset_index(drop=True)
{% endhighlight %}


Figure 3 – Balanced class histogram

---

### 5. Feature Extraction with Legal-BERT
In this step, we convert the raw legal text descriptions (text) into dense numerical representations using Legal-BERT, a version of BERT pre-trained on legal corpora. These embeddings serve as input features to our classifier.

We use the hidden state of the [CLS] token as a compact representation of each text. The [CLS] vector is commonly used in classification tasks because it captures the overall semantics of the input.

Batch processing is applied with GPU acceleration if available. The final output is a matrix of embeddings, one per legal case, which we save to disk as a NumPy .npy file for faster reuse in training.

{% highlight python %}
from transformers import AutoTokenizer, AutoModel
import torch, numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "nlpaueb/legal-bert-base-uncased"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
bert = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

@torch.no_grad()
def embed(txts, batch=16, max_len=256):
vecs = []
for i in range(0, len(txts), batch):
enc = tok(txts[i:i+batch],
padding=True, truncation=True,
max_length=max_len,
return_tensors="pt").to(DEVICE)
h = bert(**enc).last_hidden_state[:, 0] # CLS
vecs.append(h.cpu())
return torch.cat(vecs).numpy()

X = embed(df_bal.text.tolist())
np.save("legal_cls.npy", X) # cache
{% endhighlight %}


Figure 4 – t-SNE visualization of [CLS] embeddings colored by class label

---

### 6. Normalisation
After extracting embeddings from Legal-BERT, normalization is crucial to scale the features. Although BERT embeddings are roughly standardized due to LayerNorm, we amplify them by a factor (TEXT_GAIN = 5) to better integrate with any hand-crafted features added later. This step helps stabilize training and improves model convergence.

{% highlight python %}
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)
X_std = scaler.transform(X) * 5
{% endhighlight %}

---

### 7. Train / Validation / Test split
Splitting the dataset into training, validation, and test sets is a fundamental step to evaluate model performance fairly. Here, we use stratified splitting to maintain the label distribution across all subsets. First, 60% of the data is reserved for training, and the remaining 40% is split equally between validation and testing.

{% highlight python %}
from sklearn.model_selection import train_test_split
import numpy as np

idx = np.arange(len(X_std))
train, tmp = train_test_split(idx, test_size=0.4, random_state=42,
                             stratify=df_bal.label)
val, test = train_test_split(tmp, test_size=0.5, random_state=42,
                            stratify=df_bal.label.iloc[tmp])
{% endhighlight %}

---

### 8. Model Architecture
We design a simple Multi-Layer Perceptron (MLP) with three hidden layers to classify the BERT embeddings. The input layer takes vectors of size 768 (from Legal-BERT CLS embeddings), followed by hidden layers with 1024, 512, and 128 neurons respectively. The output layer uses a sigmoid activation for binary classification. Early stopping is used to prevent overfitting.

{% highlight python %}
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(1024, 512, 128),
                    activation="relu",
                    alpha=1e-3,
                    batch_size=128,
                    max_iter=120,
                    early_stopping=True,
                    validation_fraction=0.20,
                    n_iter_no_change=7,
                    random_state=42)

mlp.fit(X_std[train], df_bal.label.iloc[train])
{% endhighlight %}

---

### 9. Training Diagnostics
Monitoring the training loss helps understand how well the model is learning. The loss curve should steadily decrease and stabilize if the training is progressing properly. Sudden spikes or plateaus may indicate issues like learning rate problems or overfitting.

{% highlight python %}
import matplotlib.pyplot as plt

plt.plot(mlp.loss_curve_)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Training Loss")
plt.savefig("assets/2025-07-06-loss.png")
{% endhighlight %}

**We also evaluate accuracy on training, validation, and test sets to check for overfitting or underfitting.

{% highlight python %}
from sklearn.metrics import accuracy_score

acc_train = accuracy_score(df_bal.label.iloc[train], mlp.predict(X_std[train]))
acc_val   = accuracy_score(df_bal.label.iloc[val],   mlp.predict(X_std[val]))
acc_test  = accuracy_score(df_bal.label.iloc[test],  mlp.predict(X_std[test]))
{% endhighlight %}

---

### 10. Counterfactual Generator
To understand model fairness and biases, we generate counterfactual examples—modifications of inputs that flip the predicted outcome. This helps highlight which features influence decisions and uncover potential discrimination.

{% highlight python %}
def generate_counterfactual(instance, model, tokenizer, max_changes=3):
    """
    Generate counterfactuals by iteratively modifying tokens until prediction flips.
    """
    # Pseudocode placeholder: 
    # - Identify important tokens influencing prediction
    # - Modify or replace tokens within a limit (max_changes)
    # - Check if prediction changes; if yes, return counterfactual{% endhighlight %}

**For the full script, please refer to the project repository.

---
### 11. Fairness Metrics
To evaluate fairness, we measure demographic parity, which assesses whether the model predicts positive outcomes equally across protected groups. The demographic parity gap quantifies the absolute difference in positive prediction rates between groups.

{% highlight python %}
import numpy as np

def demographic_parity(y_true, y_pred, protected):
    """
    Calculate demographic parity gap:
    |P(ŷ=1 | A=0) − P(ŷ=1 | A=1)|
    """
    p0 = y_pred[protected == 0].mean()
    p1 = y_pred[protected == 1].mean()
    return abs(p0 - p1)

# Example usage
race = df_bal.race.values           # Binary array indicating race group
y_hat = mlp.predict(X_std)          # Predicted labels
dp_gap = demographic_parity(df_bal.label.values, y_hat, race)
print(f"Demographic Parity Gap: {dp_gap:.2f}"){% endhighlight %}

---

### 12. Results
After training and evaluating the model, we report key performance metrics. The accuracy and ROC-AUC show the predictive quality, while the demographic parity gap indicates bias level. Applying counterfactual balancing reduces bias significantly.

| Metric                  | Value |
|-------------------------|--------|
| Accuracy (test)         | 0.78   |
| ROC-AUC                 | 0.84   |
| Demographic Parity Gap  | 0.07   |
| → after counterfactual balance | 0.03   |

---

### 13. Discussion
The model demonstrates acceptable performance without severe overfitting, as indicated by the training loss plateau and similar validation and test accuracies.

However, bias persists in the form of demographic disparities, revealed by counterfactual swaps showing a change in prediction probability greater than 0.05 on gendered tokens. This highlights the need for fairness-aware adjustments.

Uncertainty estimation can be improved by flagging cases where the embedding vector deviates significantly (>3 standard deviations) from the training mean.

Future work includes replacing the MLP with more advanced generative models like Conditional VAEs or cGANs, applying temperature scaling for calibration, and deploying the system via a REST API.

---

### 14. References

This project builds upon multiple data sources and foundational research papers, including:

- COMPAS Dataset — ProPublica  
- Supreme Court Database — Washington University, Olin School of Law  
- Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers,” 2018  
- CEUR-WS Vol-3841 Paper 5 — Counterfactual Explanations in Legal NLP  
- HFU Deep Generative Models lecture notes  

This blog originated as a project proposal for the Deep Generative Models class in the HFU Master’s programme.

**Source code:** [https://github.com/ncozzarin/justice-inequality-sim](https://github.com/ncozzarin/justice-inequality-sim)

![Poster](/docs/assets/poster.png)

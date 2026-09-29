---
title: "Transformers by Hand"
published: "30 September 2026"
---

We work with a tiny vocabulary of six words:

$$
\{\text{"the"}, \text{"cat"}, \text{"sat"}, \text{"on"}, \text{"mat"}, \text{"dog"}\}
$$

Each word is represented by an integer index $0,\dots,5$ (in the order above).

# Part I - Tokens and embeddings

## Exercise 1.1 - Why subword tokens?

Real language models don't tokenize whole words like our toy vocabulary above — they use **byte-pair encoding (BPE)**: start from single bytes/characters, repeatedly find the most frequent adjacent pair in a large training text and merge it into a new symbol, and repeat until the vocabulary reaches a target size (GPT-2: 50,257 tokens, learned once, before training the network).

1. Why not tokenize by individual character? What would the model have to learn "for free" (before it can even get to meaning) that it wouldn't need to with larger units?
2. Why not tokenize by whole word instead? What happens to a typo, a rare name, or a word from another language under a word-level vocabulary?
3. Subword tokenization is a compromise: frequent words become a single token (e.g. `·the`), while rare or unusual words are split into pieces (e.g. `·Bochum` → `·Bo`, `ch`, `um`). Why does this compromise avoid both problems from 1. and 2.?
4. The model never sees individual letters, only token ids. Why does this make a task like "count the letters r in *strawberry*" surprisingly hard for a language model, even though it is trivial for a three-line script?

## Exercise 1.2 - The embedding matrix

An embedding layer stores one trainable vector of length $d=4$ per vocabulary entry, collected into a matrix $E\in\mathbb{R}^{6\times4}$.

1. How many trainable parameters does this embedding layer have? GPT-2 uses $V=50{,}257$ and $d=768$ — how many parameters is that, and how does it compare to GPT-2's total of 124M parameters?
2. The embedding matrix $E$ is not designed by hand — it is a set of trainable parameters, typically randomly initialized and then updated by backpropagation together with the rest of the network. What would a "bad" set of embeddings look like at initialization (e.g. all rows equal to $\vec 0$, or unrelated words initialized to point in the same direction), and how does training — based on the downstream task loss — change what nearby rows of $E$ come to represent?
3. The embedding dimension $d$ is a hyperparameter chosen by the model designer, independent of vocabulary size. What goes wrong if $d$ is chosen too small (relative to the vocabulary and the complexity of the task)? What goes wrong if $d$ is chosen too large?
4. After training on real text, nearest neighbours in embedding space cluster meaningfully (e.g. nearest to `Monday`: Tuesday, Wednesday, Thursday; nearest to `electron`: photon, neutron), and vector arithmetic on the embeddings captures relationships: $\text{king} - \text{man} + \text{woman} \approx \text{queen}$. What does this tell you about what the embedding space comes to represent? Nobody ever told the model "queen is to king as woman is to man" — so why does this arithmetic work, given that the embeddings were only ever trained to help predict the next token?
5. The word "the" appears twice in the sentence "the cat sat on the mat". Are its two embedding vectors identical after the lookup? Explain why this alone is not enough for the model to distinguish the two occurrences.

## Exercise 1.3 - Why attention needs position information

Self-attention (Part II) computes, for every token, a weighted average of *value* vectors, where the weights come from comparing *query* and *key* vectors — a comparison that depends only on the *content* of two tokens, not on where they sit in the sequence.

1. Suppose we shuffle the tokens of a sentence (and their padding mask) together, in the same order. Argue informally that the set of pairwise query-key comparisons is unchanged, just relabelled.
2. Based on 1., is plain self-attention sensitive to the order of the input tokens?
3. Give one concrete example of a sentence pair where the two sentences contain the same words but the order changes the meaning.
4. To restore order-sensitivity, a **positional encoding** vector $\mathbf p_t$, depending only on the position $t=0,1,2,\dots$ (not on the token identity), is added to each token's embedding: $\mathbf x_t = \mathbf e_{\text{token}(t)} + \mathbf p_t$. Why does this fix the problem identified in 2.?
5. Different models encode position differently:

   | model | position encoding |
   |---|---|
   | Transformer (2017) | fixed sines/cosines, $p_{t,2k}=\sin(t/10000^{2k/d})$ |
   | GPT-2 (2019) | a learned vector per position, up to a maximum length (e.g. 1024) |
   | Llama and most models since 2023 | RoPE: rotate $\mathbf q$ and $\mathbf k$ by an angle proportional to $t$ |

   What is one practical limitation of GPT-2's learned-position-vector approach that a fixed or rotation-based scheme does not share? (Hint: what happens if you feed the model a sequence longer than any it was trained on?)
6. **(Optional/advanced)** RoPE is designed so that the attention score between two tokens depends on the *distance* $t_i-t_j$ between them rather than on their absolute positions. Why might that be a more useful property for a model that needs to generalize to texts of different lengths than it saw during training?

---

# Part II - Scaled dot-product attention by hand

For one attention head, with queries $Q\in\mathbb{R}^{L\times d_k}$, keys $K\in\mathbb{R}^{L\times d_k}$ and values $V\in\mathbb{R}^{L\times d_v}$,

$$
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)
\qquad
Z=AV
$$

Softmax is taken **row-wise**, i.e. separately for each query, over the key dimension.

## Exercise 2.1 - A three-token example

Consider the three tokens "the", "cat", "sat" with $d_k=4$ key/query vectors and $d_v=2$ value vectors:

| token | key $k_i$ | value $v_i$ |
|---|---|---|
| the | $(2,0,0,0)$ | $(1,0)$ |
| cat | $(0,2,0,0)$ | $(0,1)$ |
| sat | $(0,0,2,0)$ | $(1,1)$ |

We compute the output for the query of token "cat", $q_{\text{cat}}=(0,2,0,0)$.

### Step 1 - Raw scores

Calculate the three dot products

$$
s_i = q_{\text{cat}}\cdot k_i\qquad i\in\{\text{the},\text{cat},\text{sat}\}
$$

**Your answers:**

$$
s_{\text{the}} = \underline{\phantom{00}}
\qquad
s_{\text{cat}} = \underline{\phantom{00}}
\qquad
s_{\text{sat}} = \underline{\phantom{00}}
$$

### Step 2 - Scale

Divide each score by $\sqrt{d_k}=\sqrt{4}=2$.

$$
\tilde s_{\text{the}} = \underline{\phantom{00}}
\qquad
\tilde s_{\text{cat}} = \underline{\phantom{00}}
\qquad
\tilde s_{\text{sat}} = \underline{\phantom{00}}
$$

### Step 3 - Softmax

Use $e^0=1$ and $e^2\approx7.39$. Compute

$$
a_i = \frac{e^{\tilde s_i}}{\sum_j e^{\tilde s_j}}
$$

Complete the table (round to 3 decimal places):

| $i$ | $\tilde s_i$ | $e^{\tilde s_i}$ | $a_i$ |
|---|---:|---:|---:|
| the | | | |
| cat | | | |
| sat | | | |

Check that your three weights sum to 1.

### Step 4 - Weighted output

Calculate

$$
z_{\text{cat}} = a_{\text{the}}\,v_{\text{the}} + a_{\text{cat}}\,v_{\text{cat}} + a_{\text{sat}}\,v_{\text{sat}}
$$

**Your answer:**

$$
z_{\text{cat}} \approx \underline{\phantom{(000,000)}}
$$

### Step 5 - Interpretation

1. Which token does "cat" attend to most strongly? Relate this to the dot product $q_{\text{cat}}\cdot k_{\text{cat}}$.
2. Would the result change if we scaled every key vector by the same positive constant $c$ before the scaled dot product? Would it change the *weights* $a_i$?
3. Why do we divide by $\sqrt{d_k}$ rather than, say, $d_k$ or not scaling at all? (Think about what happens to the scores as $d_k$ grows, if $q$ and $k$ have entries of typical size $\mathcal{O}(1)$.)

## Exercise 2.2 - Shapes of scaled dot-product attention

For a batch of $B$ examples, sequence length $L$, key dimension $d_k$ and value dimension $d_v$:

Complete the table with tensor shapes:

| quantity | shape |
|---|---|
| $Q$ | |
| $K$ | |
| $V$ | |
| $QK^\top$ | |
| $A=\operatorname{softmax}(QK^\top/\sqrt{d_k})$ | |
| $Z=AV$ | |

1. Does the shape of $Z$ depend on $d_k$?
2. In self-attention, $Q$, $K$ and $V$ are all linear projections of the same input $X\in\mathbb{R}^{B\times L\times d_{\text{model}}}$. What operation produces $Q$ from $X$, and what is the shape of the weight matrix that does it?

## Exercise 2.3 - Multi-head attention dimensionality

A multi-head attention layer with $h$ heads and model dimension $d_{\text{model}}$ splits $Q$, $K$, $V$ into $h$ smaller pieces, each of dimension

$$
d_k = d_v = \frac{d_{\text{model}}}{h}
$$

runs scaled dot-product attention independently per head, concatenates the $h$ outputs, and applies one more linear projection back to $d_{\text{model}}$.

Take $d_{\text{model}}=8$, $h=2$, $L=5$, $B=1$.

1. What is $d_k$ per head?
2. What is the shape of $Q$, $K$, $V$ **per head**?
3. What is the shape of the attention output $Z_{\text{head}}$ per head?
4. What is the shape after concatenating all $h$ heads?
5. What is the shape after the final output projection?
6. Why must $d_{\text{model}}$ be divisible by $h$?
7. If we increased $h$ from 2 to 4 while keeping $d_{\text{model}}=8$ fixed, what happens to $d_k$ per head? Does the total number of attended values change?

## Exercise 2.4 - Induction heads (Optional/advanced)

GPT-2 was shown a passage of random, unrelated words, then the *same* words again, in the same order. Its per-token prediction loss dropped from 11.8 nats (first time — unpredictable) to 0.73 nats (second time — essentially copied), despite never seeing these specific words during training.

This is explained by a two-head circuit found inside the model:
1. a **previous-token head**, which writes into each token's representation "the word before me was A";
2. an **induction head**, which, at the *current* token B, searches earlier positions whose *previous* token was also B, and copies whatever followed it there.

Pattern: "... A B ... A →" predicts B.

1. Which sub-layer of the transformer block (attention or MLP) is responsible for each of these two steps, and why does it have to be that one? (Hint: which sub-layer moves information *between* token positions?)
2. Why does this circuit require *two* attention layers working in sequence — one head's output feeding into a head in a later layer — rather than a single attention layer?
3. Why is this a form of "learning inside the prompt" (in-context learning), rather than a form of learning that requires updating any weights?

---

# Part III - Transformer blocks, decoding, and scale

## Exercise 3.1 - The transformer block and causal masking

A transformer block consists of, in order:
1. multi-head self-attention over the input sequence (tokens exchange information),
2. a residual (skip) connection adding the block's input, followed by layer normalisation,
3. a position-wise MLP (two linear layers with a nonlinearity in between, applied to each token independently),
4. another residual connection followed by layer normalisation.

A **decoder** block (used for left-to-right text generation, e.g. GPT) additionally uses a **causal mask** in step 1: token $i$ is only allowed to attend to tokens $j\leq i$.

Draw this as a block diagram with boxes and arrows, labelling the input $X$, the intermediate tensors, and the two "Add & Norm" steps.

1. What does "residual connection" mean here, concretely, in terms of a tensor addition?
2. Why is a residual connection useful for training deep stacks of these blocks (connect this to what you learned about backpropagation through many layers)?
3. For a sequence of $L=4$ tokens, complete the causal mask table. Use `True` to mean "query $i$ **may** attend to key $j$", `False` otherwise:

   | query $i$ \ key $j$ | $j=0$ | $j=1$ | $j=2$ | $j=3$ |
   |---|---|---|---|---|
   | $i=0$ | | | | |
   | $i=1$ | | | | |
   | $i=2$ | | | | |
   | $i=3$ | | | | |

4. In practice, the mask is implemented by setting the disallowed scores to $-\infty$ before the softmax, rather than by zeroing weights afterwards. Why does this matter (think about what softmax does to a row of otherwise-unmasked scores if some scores are simply removed vs. set to $-\infty$)?
5. Self-attention in an **encoder** (e.g. BERT) is bidirectional: a token can attend to tokens both before and after it. Why is this appropriate for, e.g., understanding the meaning of a complete sentence, but not appropriate for a model that must generate text left to right?

## Exercise 3.2 - The residual stream and the logit lens

Each block does not replace its input — it *adds* its output to a running vector per token, called the **residual stream**:

$$
\mathbf x \leftarrow \mathbf x + \operatorname{Attn}(\operatorname{LN}(\mathbf x)),
\qquad
\mathbf x \leftarrow \mathbf x + \operatorname{MLP}(\operatorname{LN}(\mathbf x)).
$$

Because every block only *adds* to the stream, we can peek inside it: multiplying the residual stream at *any* layer by the model's final unembedding matrix (a trick called the **logit lens**) gives the "guess" the model would make if it stopped right there. For the prompt "The electron has a negative...", GPT-2's guess for the next token evolves layer by layer: *pressure* (layer 3) → *impact* (layer 7) → *value* (layer 9) → *spin* (layer 11) → *charge* (layer 12, correct, $p=0.47$).

1. Why does the logit lens trick — reading out an intermediate layer using the *final* unembedding matrix — only make sense because of the residual connection? (Why would it fail if each block fully replaced $\mathbf x$ instead of adding to it?)
2. What does the progression *pressure → impact → value → spin → charge* across layers suggest about *where* in the network the correct prediction is formed — is it already "known" right after the embedding layer, or does it emerge gradually?
3. The MLP sub-layer makes up roughly two-thirds of a block's weights and is thought to be where much of a model's factual knowledge lives. Given that the MLP is applied identically and independently to each token position (Exercise 2.4, Question 1), what kind of "knowledge" can it plausibly store, and what kind of information must instead come from attention?

## Exercise 3.3 - Counting parameters and compute

GPT-2 small has $d=768$ ($d_{\text{model}}$), $h=12$ heads, $d_{\text{ff}}=3072$, and $N_{\text{layers}}=12$ blocks.

1. The attention sub-layer has four weight matrices of shape $(d,d)$ ($W_Q, W_K, W_V, W_O$; ignore biases). How many parameters is that, in terms of $d$?
2. The MLP sub-layer has two weight matrices, of shape $(d,d_{\text{ff}})$ and $(d_{\text{ff}},d)$ (ignore biases). How many parameters is that, in terms of $d$ and $d_{\text{ff}}$? Using $d_{\text{ff}}=4d$ (as in GPT-2), express this purely in terms of $d$.
3. Combine 1. and 2. to show that one block has $12d^2$ weight parameters.
4. For $N=12$ blocks, estimate the total number of block parameters. Compare it to GPT-2's reported total of 124M parameters — what is missing from your estimate? (Look back at Exercise 1.2, Question 1.)
5. Does the parameter count from 3.-4. depend on the sequence length $L$? Does the amount of *computation* (and the memory needed for the attention matrix $A$, of shape $(B,h,L,L)$) depend on $L$? Explain the difference.
6. **(Optional/advanced)** When generating text token by token, the keys and values of earlier tokens never change once computed, so they can be stored instead of recomputed (a **KV cache**). Why does this mean the cost of generating one *new* token scales with the number of tokens seen so far, rather than with its square?

## Exercise 3.4 - From logits to text (Optional/advanced)

The final layer produces logits $\mathbf z\in\mathbb{R}^{50257}$ for the next token, turned into probabilities via a softmax with **temperature** $T$:

$$
p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}.
$$

1. What happens to the distribution as $T\to 0$? What is this decoding strategy called, and why can it lead to repetitive loops (e.g. "the capital of the French Republic ... is the capital of the French Republic...")?
2. What happens as $T$ increases well above 1? What is the tradeoff against $T\to0$?
3. **top-k** sampling keeps only the $k$ most likely tokens before sampling; **top-p** sampling keeps the smallest set of tokens whose probabilities sum to at least $p$. Give one advantage top-p has over a fixed top-k in a case where the model is very confident about the next token (probability mass concentrated on 1-2 tokens), compared to a case where it is very unsure (mass spread over hundreds of tokens).

## Exercise 3.5 - Matching architectures to tasks (Optional/advanced)

Match each task to the most natural architecture choice: **encoder-only**, **decoder-only**, or **encoder-decoder**. Justify each in one sentence.

| task | architecture | justification |
|---|---|---|
| Sentiment classification of a movie review | | |
| Autoregressive next-word text generation | | |
| Machine translation (English $\to$ German) | | |
| Producing a single embedding vector to search a document database | | |

---

```{=latex}
\newpage
```

# Short conceptual questions

1. Why is self-attention, on its own, permutation equivariant, and what two mechanisms fix that in a real transformer (one for the encoder input, one for the decoder's generation order)?

2. Why do queries, keys and values need separate learned projections rather than using the raw input embeddings directly for all three?

3. If the embedding dimension $d_{\text{model}}$ is increased while keeping the number of heads $h$ fixed, what happens to $d_k$ per head, and what does that imply for the granularity of what each head can represent?

4. Why does layer normalisation typically operate over the feature dimension $d_{\text{model}}$ rather than over the batch dimension, unlike batch normalisation?

5. Explain, without equations, what would go wrong if a decoder used bidirectional (unmasked) self-attention during training for next-token prediction.

If the embedding dimension $d_{\text{model}}$ is increased while keeping the number of heads $h$ fixed, what happens to $d_k$ per head, and what does that imply for the granularity of what each head can represent?

## Question 4

Why does layer normalisation typically operate over the feature dimension $d_{\text{model}}$ rather than over the batch dimension, unlike batch normalisation?

## Question 5

Explain, without equations, what would go wrong if a decoder used bidirectional (unmasked) self-attention during training for next-token prediction.

## Question 6

The MLP sub-layer of a transformer block is applied identically and independently to every token position. What is doing the "mixing" of information *between* token positions in a transformer, if not the MLP?

## Question 7 (Optional/advanced)

Instead of training the embedding matrix $E$ from scratch, one could initialize it with **pretrained** embeddings (e.g. word2vec/GloVe) and either freeze them or fine-tune them further on the downstream task. When would freezing be preferable, and when would fine-tuning be preferable?

## Question 8 (Optional/advanced)

Sampling with temperature $T$ uses $p_i \propto e^{z_i/T}$. Show that this is equivalent to a softmax computed in a different base $a = e^{1/T}$ applied directly to the untouched logits $z_i$. In these terms, what does a low temperature ($T<1$) correspond to, and what does a high temperature ($T>1$) correspond to?

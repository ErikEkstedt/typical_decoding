# Typical Decoding


1. Run [streamlit](https://streamlit.io/) interface 

```bash
streamlit run typical_decoding.py
```


# Typical Decoding for Natural Language Generation

* Clara Meister, Tiago Pimentel, Gian Wiher and Ryan Cotterell, 2022 
    * [Paper arxiv](https://arxiv.org/pdf/2202.00666.pdf)
    * [Paper pdf](https://arxiv.org/pdf/2202.00666.pdf)

Let each utterance/message/turn/sentence $\textbf{y}$, consists of symbols/words/phonemes $y_{t}$, where $t$ is the
position of a symbol in time. Then the likelihood of a sequence $\textbf{y}$ can be model, through the chain-rule, as

$$

q(\textbf{y}) = \prod_{t=1}^{T} q(y_t | \textbf{y}_{<t})

$$

## Language Modelling

Model the distribution via maximization of the log-likelhood of a training corpus $\mathcal{C}$.

$$ 
L( \theta , \mathcal{C}) = - \sum_{y \in \mathcal{C}} log \ q( \textbf{y} ), \quad \text{(eq. 2)}
$$

## Information

The information of $\textbf{y}$ is quantified as the negative log-probability

$$
I(\textbf{y}) := -log \ p(\textbf{y})
$$

which over the symbols of the sequence can be expressed as

$$
I(\textbf{y}) = \sum_{t=1}^{T} I(y_t) = - \sum_{t=1}^{T} log \ p(y_t | \textbf{y}_{<t}), \quad \text{(eq. 5)}
$$

## Efficient Information Theoretic Messages

1. Maximize information throughput
    - If we send many packages with low information $\to \textbf{BAD}$ .
2. Minimize miscommunication (lost packages/messages)
    - If we "spend" maximum information on a message and it is lost $\to \textbf{BAD}$ .

A uniform distribution of Information is both robust to miscommunication while not "wasting channel time".

from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st

mpl.use("agg")


# MODEL_NAME = "gpt2"
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
TEMP = 0.9
P = 0.1
N_BEAMS = 3
N = 50
PROMPT = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley"


def update_session_state():
    if "prompt" not in st.session_state:
        st.session_state.prompt = PROMPT

    if "figure" not in st.session_state:
        st.session_state.figure = None

    if "beam_output" not in st.session_state:
        st.session_state.beam_output = ""

    if "typical_output" not in st.session_state:
        st.session_state.typical_output = ""

    if "model" not in st.session_state:
        with st.spinner(text="Loading Model..."):
            model, tokenizer = load_model()
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.success("Model loaded")

    if "model" in st.session_state:
        forward_pass()


def load_model():
    # tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    if MODEL_NAME == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    else:
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def plot_entropy(entropy, tokens):
    # shift tokens and entropy to get the 'expected information content' matching the correct symbols
    T = tokens + [""]
    T = [t.replace("Ġ", "") for t in T]
    E = torch.cat((torch.tensor((0,)), entropy[:-1]))

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.bar(torch.arange(len(E)), E, label="entropy")
    ax.set_ylabel("Expected Information")
    ax.set_xticks(range(len(T)))
    ax.set_xticklabels(T, rotation=65)
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def forward_pass():
    prompt = st.session_state.prompt

    if prompt == "":
        prompt = PROMPT

    if not isinstance(prompt, list):
        prompt = [prompt]

    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        n = input_ids.shape[-1]
        out = model(input_ids)
        logits = out.logits
        q = Categorical(logits=logits)
        entropy = q.entropy()[0].cpu()
        beam_out = model.sample(
            input_ids,
            max_length=n + N,
            do_sample=True,
            nucleus_p=P,
            temperature=TEMP,
            pad_token_id=tokenizer.eos_token_id,
        )[0][n:]
        typical_out = model.sample(
            input_ids,
            max_length=n + N,
            typical_p=P,
            do_sample=True,
            temperature=TEMP,
            pad_token_id=tokenizer.eos_token_id,
        )[0][n:]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    st.session_state.figure = plot_entropy(entropy, tokens)
    st.session_state.beam_output = tokenizer.decode(beam_out)
    st.session_state.typical_output = tokenizer.decode(typical_out)


if __name__ == "__main__":
    update_session_state()

    with st.container():
        st.markdown(
            """
            # Typical Decoding for Natural Language Generation

            * Clara Meister, Tiago Pimentel, Gian Wiher and Ryan Cotterell, 2022 
                * [Paper arxiv](https://arxiv.org/pdf/2202.00666.pdf)
                * [Paper pdf](https://arxiv.org/pdf/2202.00666.pdf)

            Let each utterance/message/turn/sentence $\\textbf{y}$, consists of symbols/words/phonemes $y_{t}$, where $t$ is the
            position of a symbol in time. Then the likelihood of a sequence $\\textbf{y}$ can be model, through the chain-rule, as

            $$
            q(\\textbf{y}) = \prod_{t=1}^{T} q(y_t | \\textbf{y}_{<t})
            $$
            """
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(
                """
                    ## Language Modelling

                    Model the distribution via maximization of the log-likelhood of a training corpus $\\mathcal{C}$.

                    $$ 
                    L( \\theta , \mathcal{C}) = - \sum_{y \in \mathcal{C}} log \ q( \\textbf{y} ), \quad \\text{(eq. 2)}
                    $$
                    """
            )
            st.markdown(
                """
                    ## Information

                    The information of $\\textbf{y}$ is quantified as the negative log-probability

                    $$
                    I(\\textbf{y}) := -log \ p(\\textbf{y})
                    $$

                    which over the symbols of the sequence can be expressed as

                    $$
                    I(\\textbf{y}) = \sum_{t=1}^{T} I(y_t) = - \sum_{t=1}^{T} log \ p(y_t | \\textbf{y}_{<t}), \quad \\text{(eq. 5)}
                    $$
                    """
            )
        with c2:
            st.markdown(
                """
                    ## Efficient Information Theoretic Messages

                    1. Maximize information throughput
                        - If we send many packages with low information $\\to \\textbf{BAD}$ .
                    2. Minimize miscommunication (lost packages/messages)
                        - If we "spend" maximum information on a message and it is lost $\\to \\textbf{BAD}$ .

                    A uniform distribution of Information is both robust to miscommunication while not "wasting channel time".
                    """
            )

    with st.container():
        st.subheader("Expected Information Content")
        st.button("update", on_click=forward_pass)
        st.pyplot(st.session_state.figure)

        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader("Prompt")
            st.text_input(st.session_state.prompt, key="prompt", on_change=forward_pass)
        with c2:
            st.json({"model": MODEL_NAME, "P": P, "temperature": TEMP})

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Continuation Typical Sampling (p=0.1)")
            st.markdown(st.session_state.typical_output)

        with c2:
            st.subheader("Continuation Nucleus Sampling (p=0.1)")
            st.markdown(st.session_state.beam_output)

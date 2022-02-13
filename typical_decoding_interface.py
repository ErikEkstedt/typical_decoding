from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st

from typical_decoding import load_model, plot_info, get_typical_information

mpl.use("agg")


# MODEL_NAME = "gpt2"
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
TEMP = 1.0
P = 0.1
OTHER_P = 0.9
N_BEAMS = 3
N = 20
# PROMPT = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley"
PROMPT = "This movie was the best that I have ever seen!"


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
            model, tokenizer = load_model(MODEL_NAME)
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.success("Model loaded")

    if "model" in st.session_state:
        forward_pass()


def forward_pass():
    prompt = st.session_state.prompt

    if prompt == "":
        prompt = PROMPT

    if not isinstance(prompt, list):
        prompt = [prompt]

    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    info = get_typical_information(prompt, model, tokenizer, to_device="cpu")
    with torch.no_grad():
        n = info["input_ids"].shape[-1]
        # TYPICAL
        typical_ids = model.generate(
            info["input_ids"].to(model.device),
            max_length=n + N,
            do_sample=True,
            typical_p=P,
            top_k=0,
            temperature=TEMP,
            pad_token_id=tokenizer.eos_token_id,
        )
        typical_info = get_typical_information(
            prompt, model, tokenizer, input_ids=typical_ids, to_device="cpu"
        )

        # Other
        other_ids = model.generate(
            info["input_ids"].to(model.device),
            max_length=n + N,
            early_stopping=True,
            do_sample=True,
            top_p=OTHER_P,
            top_k=0,
            # temperature=TEMP,
            pad_token_id=tokenizer.eos_token_id,
        )
        other_info = get_typical_information(
            prompt, model, tokenizer, input_ids=other_ids, to_device="cpu"
        )

    st.session_state.figure, _ = plot_info(
        entropy=info["entropy"][0],
        nll=info["nll"][0],
        epsilon=info["epsilon"][0],
        tokens=info["tokens"][0],
        ylim=[0, 14],
    )

    st.session_state.other_output = tokenizer.decode(other_ids[0][n:])
    st.session_state.other_figure, _ = plot_info(
        entropy=other_info["entropy"][0][n:],
        nll=other_info["nll"][0][n:],
        epsilon=other_info["epsilon"][0][n:],
        tokens=other_info["tokens"][0][n:],
        ylim=[0, 14],
    )

    st.session_state.typical_output = tokenizer.decode(typical_ids[0][n:])
    st.session_state.typical_figure, _ = plot_info(
        entropy=typical_info["entropy"][0][n:],
        nll=typical_info["nll"][0][n:],
        epsilon=typical_info["epsilon"][0][n:],
        tokens=typical_info["tokens"][0][n:],
        ylim=[0, 14],
    )


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
            st.subheader(f"Continuation Typical Sampling (p={P})")
            st.markdown(st.session_state.typical_output)
            st.pyplot(st.session_state.typical_figure)

        with c2:
            st.subheader(f"Continuation Other (p={OTHER_P})")
            st.markdown(st.session_state.other_output)
            st.pyplot(st.session_state.other_figure)

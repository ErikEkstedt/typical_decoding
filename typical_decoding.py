import math
import torch
import torch.nn.functional as F
import einops
from torch.distributions import Categorical
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt


def bits_to_nats(bits):
    return bits / torch.tensor([math.e]).log2()


def nats_to_bits(nats):
    return nats / torch.tensor([2.0]).log()


def load_model(model_name="gpt2"):
    if model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def plot_info(
    avg_expected_information,
    information_correct,
    epsilon,
    tokens,
    shifted=True,
    ylim=None,
    plot=False,
):
    fig, ax = plt.subplots(2, 1, figsize=(9, 6))

    if shifted:
        avg_expected_information = torch.tensor(
            [0] + avg_expected_information[:-1].tolist()
        )
        information_correct = torch.tensor([0] + information_correct[:-1].tolist())
        epsilon = torch.tensor([0] + epsilon[:-1].tolist())
        tokens = tokens + [""]

    n = len(avg_expected_information)
    x = torch.arange(n)
    ax[0].bar(
        x - 0.2,
        avg_expected_information,
        label="avg_expected_information (bits)",
        width=0.4,
    )
    ax[0].bar(
        x + 0.2,
        information_correct,
        label="nll: information (correct labels) (bits)",
        width=0.4,
    )
    ax[0].hlines(
        y=avg_expected_information.mean(),
        xmin=-0.3,
        xmax=n + 0.3,
        linestyle="dashed",
        color="b",
        linewidth=1.0,
        label=f"avg expected information {round(avg_expected_information.mean().item(),1)}",
    )
    ax[0].hlines(
        y=information_correct.mean(),
        xmin=-0.3,
        xmax=n + 0.3,
        linestyle="dashed",
        color="orange",
        linewidth=1.0,
        label=f"avg information (correct labels) {round(information_correct.mean().item(),1)}",
    )
    ax[0].legend(loc="upper left")
    ax[0].set_ylabel("avg expected info")
    ax[0].set_xticks(range(len(tokens)))
    ax[0].set_xticklabels(tokens, rotation=65)
    if ylim is not None:
        ax[0].set_ylim(ylim)

    ax[1].bar(x, epsilon, label="H(p) - log p(y_t)")
    ax[1].hlines(y=0, xmin=-0.3, xmax=n + 0.3, color="k", linestyle="dashed")
    ax[1].hlines(
        y=epsilon.mean(),
        xmin=-0.3,
        xmax=n + 0.3,
        linestyle="dashed",
        color="blue",
        linewidth=1.0,
        label=f"avg eps = {round(epsilon.mean().item(),2)}",
    )
    ax[1].set_ylabel("epsilon")
    ax[1].legend(loc="upper left")
    if ylim is not None:
        ax[1].set_ylim((-ylim[1] / 2, ylim[1] / 2))

    ax[1].set_xticks(range(len(tokens)))
    ax[1].set_xticklabels(tokens, rotation=65)

    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig, ax


@torch.no_grad()
def get_typical_information(prompt, model, tokenizer, input_ids=None, to_device=None):
    if input_ids is None:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    out = model(input_ids)
    logits = out.logits

    # Calculate Average Expected Information (Entropy) in bits

    # Normalize model output as probabilites
    probs = logits.softmax(dim=-1)[:, :-1].cpu()

    # Calculate the information/entropy content for associated with each symbol
    information = -probs.log2()

    # Entropy
    # Calculate the average expected information
    # Weighted sum by (normalized/sum-to-one) probabilites -> Average Weighted Entropy
    avg_expected_information = (probs * information).sum(-1)

    # Select the actual symbols in the message
    # should be a cleaner way?
    targets = input_ids[:, 1:]
    seq = torch.arange(targets.shape[-1]).repeat(targets.shape[0], 1)
    batch = torch.arange(targets.shape[0]).unsqueeze(-1)
    information_correct = information[batch, seq, targets]

    # Negative Log Likelihood as provided by F.cross_entropy (nats)
    nll = bits_to_nats(information_correct)

    # epsilon: avg_expected_information - information gained with correct token
    epsilon = avg_expected_information - information_correct

    # extract tokens
    tokens = []
    for b in range(input_ids.shape[0]):
        tmp_tokens = tokenizer.convert_ids_to_tokens(input_ids[b])
        tmp_tokens = [t.replace("Ġ", "") for t in tmp_tokens]
        tokens.append(tmp_tokens)

    ret = {
        "input_ids": input_ids,
        "logits": logits,
        "probs": probs,
        "information": information,
        "nll": nll,
        "avg_expected_information": avg_expected_information,
        "information_correct": information_correct,
        "epsilon": epsilon,
        "tokens": tokens,
    }

    if to_device is not None:
        for k, v in ret.items():
            if k not in ["tokens", "q"]:
                ret[k] = v.to(to_device)

    return ret


if __name__ == "__main__":

    model, tokenizer = load_model("EleutherAI/gpt-neo-1.3B")

    prompt = "Once upon a time there lived a princess"
    info = get_typical_information(prompt, model, tokenizer)

    b = 0
    fig, ax = plot_info(
        info["avg_expected_information"][b],
        info["information_correct"][b],
        epsilon=info["epsilon"][b],
        tokens=info["tokens"][b][:-1],
    )
    plt.show()

    b = 0
    input_ids = info["input_ids"][0].cpu()
    logits = info["logits"][0].cpu()
    entropy = info["entropy"][0].cpu()
    nll = info["nll"][0].cpu()
    tokens = info["tokens"][0]
    probs = logits.softmax(dim=-1)
    epsilon = info["epsilon"][0].cpu()
    print("logits: ", tuple(logits.shape))
    print("probs: ", tuple(probs.shape))
    print("input_ids: ", tuple(input_ids.shape))
    print("entropy: ", tuple(entropy.shape))
    print("nll: ", tuple(nll.shape))
    print("epsilon: ", tuple(epsilon.shape))
    n = logits.shape[0]

    # extract tokens
    tokens = []
    for b in range(input_ids.shape[0]):
        tmp_tokens = tokenizer.convert_ids_to_tokens(input_ids[b])
        tmp_tokens = [t.replace("Ġ", "") for t in tmp_tokens]
        tokens.append(tmp_tokens)

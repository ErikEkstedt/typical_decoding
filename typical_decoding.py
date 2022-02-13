import torch
import torch.nn.functional as F
import einops
from torch.distributions import Categorical
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt


def load_model(model_name="gpt2"):
    if model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def plot_info(entropy, nll, epsilon, tokens, ylim=None, plot=False):
    fig, ax = plt.subplots(2, 1, figsize=(9, 6))

    n = len(entropy)
    x = torch.arange(n)
    ax[0].hlines(
        y=entropy.mean(),
        xmin=-0.3,
        xmax=n + 0.3,
        linestyle="dashed",
        color="b",
        linewidth=1.0,
        label=f"avg entropy {round(entropy.mean().item(),1)}",
    )
    ax[0].hlines(
        y=nll.mean(),
        xmin=-0.3,
        xmax=n + 0.3,
        linestyle="dashed",
        color="orange",
        linewidth=1.0,
        label=f"avg nll {round(nll.mean().item(),1)}",
    )
    ax[0].bar(x - 0.2, entropy, label="avg entropy(Y)", width=0.4)
    ax[0].bar(x + 0.2, nll, label="nll(y_t)", width=0.4)
    ax[0].legend(loc="upper left")
    ax[0].set_ylabel("avg entropy")
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
    ax[1].hlines(
        y=epsilon.abs().mean(),
        xmin=-0.3,
        xmax=n + 0.3,
        linestyle="dashed",
        color="orange",
        linewidth=1.0,
        label=f"avg |eps| = {round(epsilon.abs().mean().item(),2)}",
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

    # extract tokens
    tokens = []
    for b in range(input_ids.shape[0]):
        tmp_tokens = tokenizer.convert_ids_to_tokens(input_ids[b])
        tmp_tokens = [t.replace("Ġ", "") for t in tmp_tokens]
        tokens.append(tmp_tokens)

    # get distribution
    q = Categorical(logits=logits)
    entropy = q.entropy()
    entropy = torch.cat(
        (torch.zeros((entropy.shape[0], 1), device=entropy.device), entropy[:, :-1]),
        dim=1,
    )

    n = logits.shape[1] - 1
    nll = F.cross_entropy(
        einops.rearrange(logits[:, :-1], "b n d -> (b n) d"),
        einops.rearrange(input_ids[:, 1:], "b n -> (b n)"),
        reduction="none",
    )
    nll = einops.rearrange(nll, "(b n) -> b n", n=n)
    nll = torch.cat((torch.zeros((nll.shape[0], 1), device=nll.device), nll), dim=1)
    epsilon = entropy - nll
    ret = {
        "input_ids": input_ids,
        "logits": logits,
        "nll": nll,
        "epsilon": epsilon,
        "tokens": tokens,
        "q": q,
        "entropy": entropy,
    }

    if to_device is not None:
        for k, v in ret.items():
            if k not in ["tokens", "q"]:
                ret[k] = v.to(to_device)
    return ret


if __name__ == "__main__":

    model, tokenizer = load_model("EleutherAI/gpt-neo-1.3B")

    prompt = "Once upon a time there lived a princess"

    info = typical_information(prompt, model, tokenizer)

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

    # plot all info

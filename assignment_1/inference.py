import torch
import argparse
from transformer import Transformer
from tokenizer import Tokenizer

def top_p_sampling(probs: torch.Tensor, p: float) -> int:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative_probs > p

    if torch.any(cutoff):
        cutoff_idx = torch.argmax(cutoff.to(torch.int)) + 1
        top_probs = sorted_probs[:cutoff_idx]
        top_indices = sorted_indices[:cutoff_idx]
    else:
        top_probs = sorted_probs
        top_indices = sorted_indices

    top_probs = top_probs / top_probs.sum()
    sampled_index = torch.multinomial(top_probs, 1).item()
    return top_indices[sampled_index].item()

@torch.inference_mode()
def generate(
    model: Transformer,
    prompt: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    end_token: int
):
    model.eval()
    x = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_tokens):
        x_condensed = x[:, -model.rope.precomputed_sin.shape[0]:]
        logits = model(x_condensed)
        last_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(last_logits, dim=-1).squeeze()

        next_token = top_p_sampling(probs, p=top_p)
        x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)

        if next_token == end_token:
            break

    return x.squeeze().tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--merges_file", type=str, required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(
        args.vocab_file,
        args.merges_file,
        special_tokens=["<|endoftext|>"]
    )
    end_token = next(k for k, v in tokenizer.vocab.items() if v == b"<|endoftext|>")

    if args.prompt.strip() == "":
        prompt_ids = [end_token]
    else:
        prompt_ids = tokenizer.encode(args.prompt)

    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(args.device)

    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state"])

    output_ids = generate(
        model=model,
        prompt=prompt_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        end_token=end_token,
    )

    decoded = tokenizer.decode(output_ids)
    print("Generated text:\n", decoded)

if __name__ == "__main__":
    main()

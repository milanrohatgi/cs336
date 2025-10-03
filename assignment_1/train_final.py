import torch
import numpy as np
import argparse
import wandb
from transformer_final import *
import time
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32   = True
torch.backends.cudnn.benchmark = True

@torch.no_grad()
def get_val_loss(model, x_val, batch_size, context_length, device, num_batches=10):
    model.eval()
    losses = []
    for _ in range(num_batches):
        inputs, targets = get_batch(x_val, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        losses.append(loss.item())
    return sum(losses) / len(losses)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--min_lr", type=float, default=5e-5)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    wandb.init(project="cs336", name=args.run_name, config=vars(args))

    x = np.load(args.train_path, mmap_mode="r") 
    x_val = np.load(args.val_path, mmap_mode="r")

    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(args.device)

    model = torch.compile(model)

    muon_params  = []
    adamw_params = []

    for name, p in model.named_parameters():
        if p.ndim >= 2 and "embedding" not in name and "out_proj" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)
   

    muon_opt  = Muon(muon_params, lr=0.05, momentum=0.95, ns_steps=5)
    adamw_opt = AdamW(adamw_params, lr=args.learning_rate)
    scaler = GradScaler()

    start_iter = 0
    if args.load:
        start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)

    start_time = time.time()
    for t in tqdm(range(start_iter, args.max_iters)):
        model.train()
        inputs, targets = get_batch(x, args.batch_size, args.context_length, args.device)

        adamw_opt.zero_grad()
        muon_opt.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(adamw_opt)        
        gradient_clipping(model.parameters(), 5.0)

        # Cosine LR schedule
        lr = learning_rate_schedule(
            t,
            a_max=args.learning_rate,
            a_min=args.min_lr,
            T_w=args.warmup_iters,
            T_c=args.max_iters
        )
        for param_group in adamw_opt.param_groups:
            param_group["lr"] = lr

        scaler.step(adamw_opt)
        scaler.unscale_(muon_opt)
        muon_opt.step()

        scaler.update()

        if t % 100 == 0 or t % 1000 == 0:
            elapsed_time = time.time() - start_time
            log_dict = {
                "train_loss": loss.item(),
                "lr": lr,
                "step": t,
                "elapsed_time": elapsed_time,
            }

            if t % 1000 == 0:
                val_loss = get_val_loss(model, x_val, args.batch_size, args.context_length, args.device)
                print(f"Step {t} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed_time:.2f}s | LR: {lr:.2e}")
                log_dict["val_loss"] = val_loss

            wandb.log(log_dict, step=t)

    print("Running full validation...")
    final_val_loss = get_val_loss(model, x_val, args.batch_size, args.context_length, args.device, num_batches=100)
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    save_checkpoint(model, adamw_opt, muon_opt, t, args.checkpoint_path)

if __name__ == "__main__":
    main()
#!/bin/bash

CHECKPOINT_PATH="/data/c-mrohatgi/checkpoints/owt_base.pt"
VOCAB_FILE="/data/c-mrohatgi/owt_bpe_vocab.json"    
MERGES_FILE="/data/c-mrohatgi/owt_bpe_merges.json"    
PROMPT=""
VOCAB_SIZE=32000
CONTEXT_LENGTH=256
NUM_LAYERS=4
D_MODEL=512
NUM_HEADS=16
D_FF=1344
ROPE_THETA=10000.0
MAX_TOKENS=512
TEMPERATURE=1.0
TOP_P=0.9
DEVICE="cuda"

uv run python inference.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --prompt "$PROMPT" \
  --vocab_size "$VOCAB_SIZE" \
  --context_length "$CONTEXT_LENGTH" \
  --num_layers "$NUM_LAYERS" \
  --d_model "$D_MODEL" \
  --num_heads "$NUM_HEADS" \
  --d_ff "$D_FF" \
  --rope_theta "$ROPE_THETA" \
  --max_tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --device "$DEVICE" \
  --vocab_file "$VOCAB_FILE" \
  --merges_file "$MERGES_FILE"

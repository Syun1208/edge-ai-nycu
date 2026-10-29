import argparse
import contextlib
import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import DEFAULT_MODEL, compression_rate, eval_ppl_wikitext2, load_model 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=None, choices=[1, 2, 3, 4], help="omit for the base model")
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--wbits", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.case is not None and args.rate is None:
        ap.error("--rate is required with --case")

    with contextlib.redirect_stdout(sys.stderr): 
        torch.manual_seed(args.seed)
        model, tok = load_model(args.model, args.dtype)
       
        ppl = eval_ppl_wikitext2(model, tok)
        rate = compression_rate(model)

    print(f"case{args.case}:" if args.case is not None else "base:")
    print(f"ppl: {ppl:.4f}")
    print(f"comp_rate: {rate:.2%}")
    print("------------------------", flush=True)

if __name__ == "__main__":
    main()

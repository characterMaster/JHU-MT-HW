
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ibm1_final_minibatch.py
-----------------------
IBM Model 1 with streaming mini-batch EM (Scheme A), designed to be a drop-in
replacement for ibm1_final.py's training flow while producing a persistent
alignment file that Gradescope expects: "alignment".

Key points:
- Single process, I EM iterations over the entire dataset
- Streams sentence pairs in mini-batches; accumulates expected counts per-iter
- Outputs Pharaoh format (fIndex-eIndex) alignments; 0-based indices
- Supports NULL alignment internally; links to NULL are not printed
- Backward-compatible CLI flags: -d (prefix), -n (num_sent), -i (iters), -t (threshold)
- New: -b (batch size, default 1000), --save_alignment_path (default required filename)
- Optional: --save_model / --init_from for saving/loading translation probs

Usage example:
  python ibm1_final_minibatch.py -d data/hansards -n 10000 -i 10 -b 1000 -t 0.1
  python score-alignments < alignment (cmd)
  Get-Content .\alignment | python .\score-alignments (powershell)

"""

import argparse
import json
from collections import defaultdict
import math
from typing import Iterator, List, Tuple, Dict, Iterable

NULL = "##NULL##"

def read_parallel(prefix: str, limit: int | None = None) -> Iterator[Tuple[List[str], List[str]]]:
    e_path = f"{prefix}.e"
    f_path = f"{prefix}.f"
    with open(e_path, "r", encoding="utf-8") as fe, open(f_path, "r", encoding="utf-8") as ff:
        count = 0
        while True:
            e_line = fe.readline()
            f_line = ff.readline()
            if not e_line or not f_line:
                break
            e_toks = e_line.strip().split()
            f_toks = f_line.strip().split()
            yield (e_toks, f_toks)
            count += 1
            if limit is not None and count >= limit:
                break

def batched(it: Iterable, batch_size: int) -> Iterator[List]:
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def build_cooccur(prefix: str, limit: int | None, batch_size: int):
    from collections import defaultdict
    co = defaultdict(set)
    for batch in batched(read_parallel(prefix, limit), batch_size):
        for e_toks, f_toks in batch:
            e_set = set(e_toks + [NULL])
            for f in set(f_toks):
                co[f].update(e_set)
    return co

def init_t_from_cooccur(co) -> Dict[str, Dict[str, float]]:
    t = {}
    for f, e_set in co.items():
        Z = float(len(e_set)) if e_set else 1.0
        t[f] = {e: 1.0 / Z for e in e_set}
    return t

def load_t(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        t = json.load(f)
    for fwd in list(t.keys()):
        t[fwd] = {e: float(p) for e, p in t[fwd].items()}
    return t

def save_t(t: Dict[str, Dict[str, float]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(t, f, ensure_ascii=False)

def em_train(prefix: str, limit: int | None, iters: int, batch_size: int,
             t: Dict[str, Dict[str, float]] | None = None) -> Dict[str, Dict[str, float]]:
    if t is None:
        co = build_cooccur(prefix, limit, batch_size)
        t = {}
        for f, e_set in co.items():
            for e in e_set:
                if e not in t:
                    t.setdefault(f, {})[e] = 1.0 / len(e_set) #t[f][e]

    for it in range(1, iters + 1):
        count_fe = defaultdict(lambda: defaultdict(float))  # count[f][e]
        total_e  = defaultdict(float)                       

        for batch in batched(read_parallel(prefix, limit), batch_size):
            for e_toks, f_toks in batch:
                e_list = e_toks + [NULL]
                for f in f_toks:
                    Z = sum(t.get(f, {}).get(e, 0.0) for e in e_list)
                    if Z == 0.0:
                        invZ = 1.0 / len(e_list)
                        for e in e_list:
                            count_fe[f][e] += invZ
                            total_e[e]     += invZ
                        continue
                    invZ = 1.0 / Z
                    for e in e_list:
                        p = t.get(f, {}).get(e, 0.0)
                        if p > 0.0:
                            delta = p * invZ
                            count_fe[f][e] += delta
                            total_e[e]     += delta

        new_t = {}
        for f, e_counts in count_fe.items():
            for e, c in e_counts.items():
                if total_e[e] > 0.0:
                    new_t.setdefault(f, {})[e] = c / total_e[e]   # t[f][e] normalize to e
        t = new_t
        print(f"Iteration {it} done (params={sum(len(v) for v in t.values())})")
    return t


def viterbi_align_sentence(e_toks, f_toks, t, thr):
    links = []
    for i, f in enumerate(f_toks):
        tf = t.get(f, {})
        best_j = -1
        best_p = thr                    
        for j, e in enumerate(e_toks):
            p = tf.get(e, 0.0)
            # diagonal preference
            distance_penalty = math.exp(-abs(i - j * len(f_toks) / max(1, len(e_toks))))
            p *= distance_penalty
            if p > best_p:
                best_p = p
                best_j = j
        if best_j >= 0:
            links.append((i, best_j))
    return links


def write_alignments(prefix: str, limit: int | None, batch_size: int,
                     t: Dict[str, Dict[str, float]], thr: float, out_path: str):
    with open(out_path, "w", encoding="utf-8") as out:
        for batch in batched(read_parallel(prefix, limit), batch_size):
            for e_toks, f_toks in batch:
                links = viterbi_align_sentence(e_toks, f_toks, t, thr)
                out.write(" ".join(f"{i}-{j}" for i, j in links) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data_prefix", type=str, default="data/hansards",
                    help="Prefix to parallel files: <prefix>.e and <prefix>.f")
    ap.add_argument("-n", "--num_sent", type=int, default=None,
                    help="Number of sentence pairs to use")
    ap.add_argument("-i", "--iters", type=int, default=5,
                    help="EM iterations")
    ap.add_argument("-b", "--batch_size", type=int, default=1000,
                    help="Mini-batch size (sentences)")
    ap.add_argument("-t", "--threshold", type=float, default=0.0,
                    help="Posterior threshold when emitting links")
    ap.add_argument("--save_alignment_path", type=str,
                    default="alignment",
                    help="Output alignment file path (Pharaoh format).")
    ap.add_argument("--save_model", type=str, default=None,
                    help="Optional: save learned t(e|f) JSON (sparse)")
    ap.add_argument("--init_from", type=str, default=None,
                    help="Optional: initialize t(e|f) from a saved JSON")

    args = ap.parse_args()

    if args.init_from:
        print(f"Loading model from {args.init_from} ...")
        t = load_t(args.init_from)
    else:
        t = None

    print("Training with streaming mini-batch EM...")
    t = em_train(args.data_prefix, args.num_sent, args.iters, args.batch_size, t=t)

    if args.save_model:
        print(f"Saving model to {args.save_model} ...")
        save_t(t, args.save_model)

    print(f"Writing Pharaoh alignments to {args.save_alignment_path} ...")
    write_alignments(args.data_prefix, args.num_sent, args.batch_size, t, args.threshold, args.save_alignment_path)
    print("Done. You can now evaluate with:  python score-alignments < alignment")

if __name__ == "__main__":
    main()

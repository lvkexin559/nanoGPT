"""
inspect_cr.py - 鸡兔同笼 (chickens_rabbits) 数据集和模型的速查工具。

把日常重复跑的 4 条诊断命令整合成子命令：
  peek-train   解码 train.bin 前 N 个 token (默认 100)
  peek-ids     展示 train.bin 前 N 个 token 的 (id, char) 对照 (默认 20)
  peek-ood     解码 val_ood.bin 前 N 个 token (默认 200)，验证是否真的 OOD
  eval         加载 ckpt，对 val_iid + val_ood 做 exact-match 评估
  all          跑前 3 个 peek 子命令 (不含 eval，因 eval 要 GPU 时间)

运行示例 (建议先 cd 到 nanoGPT 项目根目录):
  python inspect_cr.py peek-train
  python inspect_cr.py peek-ids -n 30
  python inspect_cr.py peek-ood -n 300
  python inspect_cr.py eval --ckpt out-cr/ckpt.pt --n-iid 200 --n-ood 200
  python inspect_cr.py eval --ckpt out-cr-cot/ckpt.pt   # 评估 CoT 版本
  python inspect_cr.py all

所有路径默认以本脚本所在目录为基准，所以 cwd 在哪儿都能跑。
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from typing import Callable

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "chickens_rabbits")
DEFAULT_META = os.path.join(DATA_DIR, "meta.pkl")
DEFAULT_TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
DEFAULT_VAL_OOD_BIN = os.path.join(DATA_DIR, "val_ood.bin")
DEFAULT_CKPT = os.path.join(SCRIPT_DIR, "out-cr", "ckpt.pt")


def _require_file(path: str, hint: str) -> None:
    """Fail fast with an actionable message when an input file is missing."""
    if not os.path.exists(path):
        raise SystemExit(
            f"[inspect_cr] missing file: {path}\n"
            f"             {hint}"
        )


def _load_meta(meta_path: str) -> tuple[dict, dict]:
    """Return (meta_dict, itos). Centralized so each subcommand handles the
    same way and shows the same error if meta.pkl is gone."""
    _require_file(
        meta_path,
        "did you run `python data/chickens_rabbits/prepare.py` yet?",
    )
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    if "itos" not in meta:
        raise SystemExit(
            f"[inspect_cr] meta.pkl at {meta_path} has no 'itos' field; "
            f"keys present: {sorted(meta.keys())}"
        )
    return meta, meta["itos"]


def cmd_peek_train(args: argparse.Namespace) -> None:
    """命令 ①: 看 train.bin 前 N 个 token 的文本形式。"""
    _require_file(args.bin, "did you run prepare.py?")
    meta, itos = _load_meta(args.meta)

    arr = np.fromfile(args.bin, dtype=np.uint16)
    if len(arr) == 0:
        raise SystemExit(f"[inspect_cr] {args.bin} is empty")

    n = min(args.n, len(arr))
    text = "".join(itos[i] for i in arr[:n].tolist())

    print(f"=== train.bin 前 {n} 个 token 解码 ===")
    print(text)
    print()
    print(f"总 token 数: {len(arr):,}")
    print(f"文件大小估算: {len(arr) * 2 / 1024 / 1024:.2f} MB (uint16 = 2 字节/token)")
    if "format" in meta:
        print(f"prepare.py format: {meta['format']}  (A=直答, B=CoT)")
    if "h_max_train" in meta:
        print(f"训练 H 范围: [2, {meta['h_max_train']}]")


def cmd_peek_ids(args: argparse.Namespace) -> None:
    """命令 ②: 看 token id 和字符的对应。"""
    _require_file(args.bin, "did you run prepare.py?")
    _meta, itos = _load_meta(args.meta)

    arr = np.fromfile(args.bin, dtype=np.uint16)
    if len(arr) == 0:
        raise SystemExit(f"[inspect_cr] {args.bin} is empty")

    n = min(args.n, len(arr))
    print(f"前 {n} 个 token 的 (id, char) 对照 (来源: {os.path.basename(args.bin)}):")
    for i, tok in enumerate(arr[:n].tolist()):
        if tok not in itos:
            print(f"  pos {i:2d}  id={tok:2d}  char=<UNKNOWN; not in itos>")
            continue
        ch = itos[tok]
        print(f"  pos {i:2d}  id={tok:2d}  char={ch!r}")


def cmd_peek_ood(args: argparse.Namespace) -> None:
    """命令 ③: 看 val_ood.bin 是不是真的"超出训练范围"。"""
    _require_file(args.bin, "did you run prepare.py?")
    meta, itos = _load_meta(args.meta)

    arr = np.fromfile(args.bin, dtype=np.uint16)
    if len(arr) == 0:
        raise SystemExit(f"[inspect_cr] {args.bin} is empty")

    n = min(args.n, len(arr))
    text = "".join(itos[i] for i in arr[:n].tolist())

    h_max_train = meta.get("h_max_train", 20)
    print(f"=== val_ood.bin 前 {n} 个 token 解码 "
          f"(H 应该都 > {h_max_train}) ===")
    print(text)

    leak_count = 0
    sample_count = 0
    for line in text.split("\n"):
        if not line.startswith("H="):
            continue
        sample_count += 1
        try:
            head = line.split(" F=")[0]
            H = int(head[2:])
            if H <= h_max_train:
                leak_count += 1
        except (ValueError, IndexError) as e:
            print(f"[warn] failed to parse line {line!r}: {e}", file=sys.stderr)

    print()
    print(f"[check] 在前 {n} 个 token 中解析到 {sample_count} 个样本头, "
          f"其中 H <= {h_max_train} 的有 {leak_count} 个")
    if leak_count > 0 and sample_count > 0:
        print(f"[warn] val_ood 里出现了不该有的小 H, "
              f"prepare.py 的 OOD 隔离可能被破坏", file=sys.stderr)


def _build_encoder_decoder(meta: dict) -> tuple[Callable[[str], list], Callable[[list], str]]:
    """Wrap stoi/itos into encode/decode closures. Same convention as
    prepare.py so behavior is consistent."""
    stoi = meta["stoi"]
    itos = meta["itos"]

    def encode(s: str) -> list:
        out = []
        for ch in s:
            if ch not in stoi:
                raise ValueError(f"char {ch!r} not in vocab")
            out.append(stoi[ch])
        return out

    def decode(ids) -> str:
        return "".join(itos[i] for i in ids)

    return encode, decode


def _gen_sample(h_min: int, h_max: int, rng: random.Random) -> tuple[int, int, int, int]:
    H = rng.randint(h_min, h_max)
    c = rng.randint(0, H)
    r = H - c
    return H, 2 * c + 4 * r, c, r


def _parse_answer_fmt_a(generated_tail: str) -> tuple[int, int] | None:
    """fmt_A 输出: 'c=<int> r=<int>\\n...' -> (c, r) or None on failure."""
    first_line = generated_tail.split("\n")[0].strip()
    try:
        c_part, r_part = first_line.split(" r=")
        return int(c_part), int(r_part)
    except (ValueError, IndexError):
        return None


def _parse_answer_fmt_b(generated_tail: str) -> tuple[int, int] | None:
    """fmt_B 输出: '2H=<int> D=<int> r=<int> c=<int>\\n...' -> (c, r) or None.
    prepare.py 里 fmt_B 的输出顺序是 r 先 c 后。"""
    first_line = generated_tail.split("\n")[0].strip()
    try:
        parts = first_line.split()
        kv = {}
        for token in parts:
            if "=" not in token:
                continue
            k, v = token.split("=", 1)
            kv[k] = int(v)
        if "c" in kv and "r" in kv:
            return kv["c"], kv["r"]
        return None
    except (ValueError, IndexError):
        return None


def cmd_eval(args: argparse.Namespace) -> None:
    """命令 ④: 验证模型学到了什么 (quick eval, exact match)。"""
    import torch
    from model import GPTConfig, GPT

    _require_file(args.ckpt, "train first or pass --ckpt to a real .pt file")
    meta, itos = _load_meta(args.meta)
    encode, decode = _build_encoder_decoder(meta)

    fmt = meta.get("format", "A") if args.format == "auto" else args.format
    if fmt == "A":
        prompt_template = "H={H} F={F}\nc="
        parser = _parse_answer_fmt_a
    elif fmt == "B":
        prompt_template = "H={H} F={F}\n2H="
        parser = _parse_answer_fmt_b
    else:
        raise SystemExit(f"[inspect_cr] unknown format {fmt!r}; expected 'A' or 'B'")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda not available, falling back to cpu", file=sys.stderr)
        device = "cpu"

    print(f"[load] ckpt = {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    gptconf = GPTConfig(**ckpt["model_args"])
    model = GPT(gptconf)
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[10:]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval().to(device)

    h_max_train = meta.get("h_max_train", 20)
    h_min_ood = meta.get("h_min_ood", h_max_train + 1)
    h_max_ood = meta.get("h_max_ood", 50)
    print(f"[load] format={fmt}  device={device}  "
          f"h_train=[2,{h_max_train}]  h_ood=[{h_min_ood},{h_max_ood}]")

    @torch.no_grad()
    def eval_split(name: str, h_min: int, h_max: int, n: int, seed: int) -> None:
        rng = random.Random(seed)
        em = 0
        parse_errors = 0
        first_errors: list[str] = []
        for _ in range(n):
            H, F, c_gt, r_gt = _gen_sample(h_min, h_max, rng)
            prompt = prompt_template.format(H=H, F=F)
            x = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            y = model.generate(x, max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature, top_k=args.top_k)
            gen = decode(y[0].tolist())[len(prompt):]
            parsed = parser(gen)
            if parsed is None:
                parse_errors += 1
                if len(first_errors) < args.show_errors:
                    first_errors.append(f"  H={H} F={F}  raw={gen.split(chr(10))[0]!r}")
                continue
            c_pred, r_pred = parsed
            if c_pred == c_gt and r_pred == r_gt:
                em += 1

        print(f"{name:8} n={n}  exact_match={100 * em / n:5.1f}%  "
              f"parse_err={parse_errors}/{n}")
        if first_errors:
            print(f"  first {len(first_errors)} parse failures:")
            for line in first_errors:
                print(line)

    print("\nquick eval:")
    eval_split("val_iid", 2, h_max_train, args.n_iid, args.seed_iid)
    eval_split("val_ood", h_min_ood, h_max_ood, args.n_ood, args.seed_ood)


def cmd_all(args: argparse.Namespace) -> None:
    """跑前 3 个 peek 子命令 (eval 单独跑，因为要 GPU)。"""
    print("\n" + "=" * 60)
    print(" [1/3] peek-train")
    print("=" * 60)
    cmd_peek_train(args)
    print("\n" + "=" * 60)
    print(" [2/3] peek-ids")
    print("=" * 60)
    cmd_peek_ids(args)
    print("\n" + "=" * 60)
    print(" [3/3] peek-ood")
    print("=" * 60)
    cmd_peek_ood(args)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inspect_cr",
        description="鸡兔同笼数据集 / 模型速查工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--meta", default=DEFAULT_META, help="meta.pkl 路径")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="CMD")

    p1 = sub.add_parser("peek-train", help="解码 train.bin 前 N 个 token (命令 ①)",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p1.add_argument("--bin", default=DEFAULT_TRAIN_BIN, help="train.bin 路径")
    p1.add_argument("-n", type=int, default=100, help="解码 token 数")
    p1.set_defaults(func=cmd_peek_train)

    p2 = sub.add_parser("peek-ids", help="展示前 N 个 token 的 (id, char) 对照 (命令 ②)",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p2.add_argument("--bin", default=DEFAULT_TRAIN_BIN, help="bin 文件路径")
    p2.add_argument("-n", type=int, default=20, help="展示 token 数")
    p2.set_defaults(func=cmd_peek_ids)

    p3 = sub.add_parser("peek-ood", help="解码 val_ood.bin 验证是否真 OOD (命令 ③)",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p3.add_argument("--bin", default=DEFAULT_VAL_OOD_BIN, help="val_ood.bin 路径")
    p3.add_argument("-n", type=int, default=200, help="解码 token 数")
    p3.set_defaults(func=cmd_peek_ood)

    p4 = sub.add_parser("eval", help="quick eval: val_iid + val_ood exact-match (命令 ④)",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p4.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint .pt 路径")
    p4.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="推理设备")
    p4.add_argument("--format", default="auto", choices=["auto", "A", "B"],
                    help="prompt 格式: auto=从 meta.pkl 读; A=直答; B=CoT. "
                         "评估老的 out-cr/ckpt.pt 时若 meta 已重生成为 B, 需手动指定 A")
    p4.add_argument("--n-iid", type=int, default=200, help="val_iid 评估样本数")
    p4.add_argument("--n-ood", type=int, default=200, help="val_ood 评估样本数")
    p4.add_argument("--seed-iid", type=int, default=1001)
    p4.add_argument("--seed-ood", type=int, default=1002)
    p4.add_argument("--max-new-tokens", type=int, default=12,
                    help="单次生成 token 上限 (fmt_B 可能要 20+)")
    p4.add_argument("--temperature", type=float, default=0.01, help="贪心采样建议 <= 0.01")
    p4.add_argument("--top-k", type=int, default=1, help="贪心采样建议 1")
    p4.add_argument("--show-errors", type=int, default=3,
                    help="每个 split 显示前 K 条 parse 失败的样例")
    p4.set_defaults(func=cmd_eval)

    p5 = sub.add_parser("all", help="跑 peek-train + peek-ids + peek-ood (不含 eval)",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p5.add_argument("--bin", default=DEFAULT_TRAIN_BIN,
                    help="peek-train/peek-ids 用的 bin 文件")
    p5.add_argument("-n", type=int, default=100,
                    help="(为方便聚合) peek-train 用; peek-ids/peek-ood 用内部默认值")
    p5.set_defaults(func=cmd_all)

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.cmd == "all":
        train_args = argparse.Namespace(meta=args.meta, bin=DEFAULT_TRAIN_BIN, n=args.n)
        ids_args = argparse.Namespace(meta=args.meta, bin=DEFAULT_TRAIN_BIN, n=20)
        ood_args = argparse.Namespace(meta=args.meta, bin=DEFAULT_VAL_OOD_BIN, n=200)
        cmd_peek_train(train_args)
        print()
        cmd_peek_ids(ids_args)
        print()
        cmd_peek_ood(ood_args)
        return
    args.func(args)


if __name__ == "__main__":
    main()

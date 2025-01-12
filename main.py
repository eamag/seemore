from modules.lightning_model import LitVisionLanguageModel
from modules.datamodule import VLMDataModule
from argparse import ArgumentParser
import os
import lightning as L
from pathlib import Path


def preprocess_txt():
    filename = Path("input.txt")
    with open(str(filename), "r", encoding="utf-8") as f:
        text = f.read()
    # Character encoding and decoding functions
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    stoi["<pad>"] = 65
    itos = {i: ch for i, ch in enumerate(chars)}
    itos[65] = "<pad>"

    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return "".join([itos[i] for i in l])

    vocab_size = len(stoi.keys())
    return stoi, encode, decode, vocab_size


parser = ArgumentParser()
parser.add_argument("--image_embed_dim", type=int, default=512)
parser.add_argument("--n_layer", type=int, default=8)
parser.add_argument("--img_size", type=int, default=96)
parser.add_argument("--patch_size", type=int, default=16)
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--num_blks", type=int, default=3)
parser.add_argument("--emb_dropout", type=float, default=0.1)
parser.add_argument("--blk_dropout", type=float, default=0.1)
parser.add_argument("--learning_rate", type=float, default=1e-3)


if __name__ == "__main__":
    stoi, encode, decode, vocab_size = preprocess_txt()
    args = parser.parse_args()

    model = LitVisionLanguageModel(
        vocab_size=vocab_size,
        image_embed_dim=args.image_embed_dim,
        n_layer=args.n_layer,
        img_size=args.img_size,
        patch_size=args.patch_size,
        n_head=args.n_head,
        num_blks=args.num_blks,
        emb_dropout=args.emb_dropout,
        blk_dropout=args.blk_dropout,
        learning_rate=args.learning_rate,
    )
    dm = VLMDataModule(encode, stoi)
    trainer = L.Trainer()
    trainer.fit(model, dm)

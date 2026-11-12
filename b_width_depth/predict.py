"""Write a flip-TTA submission from a saved checkpoint (best.pt / latest.pt / final.pt).

Use it for runs trained with --skip-submission. Prediction is train.generate_submission
(average of original and flipped logits).

    python predict.py --checkpoint outputs/B3_large_fold0_<time>/best.pt \
        --widths 64 128 256 --blocks-per-stage 4 4 3 --output submissions/b3.csv
"""
import argparse

import torch

from data_utils import read_csv
from models.resnet import WideResNet
from train import generate_submission


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--widths", type=int, nargs=3, required=True)
    p.add_argument("--blocks-per-stage", type=int, nargs=3, required=True)
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    model = WideResNet(num_classes=10, widths=tuple(args.widths), blocks_per_stage=tuple(args.blocks_per_stage)).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    print(f"[Checkpoint] epoch={state.get('epoch')} weights={state.get('weights_source', 'raw')} "
          f"params={model.count_params():,}")
    generate_submission(model, read_csv(f"{args.data_dir}/test.csv"), args.data_dir, args.output,
                        device, batch_size=256, tta=True)


if __name__ == "__main__":
    main()

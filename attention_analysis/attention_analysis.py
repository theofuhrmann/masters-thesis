import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from dotenv import load_dotenv
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
load_dotenv()
vovit_path = os.path.abspath(os.getenv("VOVIT_PATH"))
sys.path.insert(0, vovit_path)

from vovit import VoViT_b, VoViT_f, VoViT_fb  # type: ignore # noqa: E402
from vovit.display.dataloaders_new import (  # noqa: E402 # type: ignore
    AudioType,
    ModelType,
    SaragaAudiovisualDataset,
)

AUDIO_RATE = 16384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DURATION = 4.0
DATASET_PATH = os.getenv("DATASET_PATH")
DATASET_METADATA_PATH = os.path.join(
    DATASET_PATH, "dataset_metadata_test.json"
)


def find_attentions(model, sample, device):
    """
    Finds the attention scores for audio and video features for a given sample.
    """
    audio = sample["audio"].to(device)
    landmarks = sample["landmarks"].to(device)
    body_ld = None
    if "body_ld" in sample and isinstance(sample["body_ld"], torch.Tensor):
        body_ld = sample["body_ld"].to(device)

    with torch.no_grad():
        with autocast():
            output = model(audio, landmarks, body_ld)

    attention_weights = output.get("attention_weights")

    if attention_weights is None:
        return None, None

    # attention_weights shape: (batch_size, audio_seq_len, video_seq_len)
    # with batch_size = 1

    # Attention score for each video feature timestep (how much audio attends to it)
    video_attention = attention_weights.mean(dim=1).squeeze(0)

    # Attention score for each audio feature timestep (how much it attends to video)
    audio_attention = attention_weights.mean(dim=2).squeeze(0)

    return audio_attention.cpu().numpy(), video_attention.cpu().numpy()


def main(args):
    model_type_map = {
        "vocal": ModelType.FACE,
        "violin": ModelType.BODY,
    }
    model_type = model_type_map[args.model]

    dataset = SaragaAudiovisualDataset(
        data_path=DATASET_PATH,
        audiorate=AUDIO_RATE,
        chunk_duration=DURATION,
        model_type=model_type,
        audio_type=AudioType.VOCAL,
        metadata_path=DATASET_METADATA_PATH,
    )

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True
    )

    print(f"Loading {args.model} model to {DEVICE}...")
    if model_type == ModelType.FACE:
        model = VoViT_f(pretrained=True, debug={}).to(DEVICE)
    elif model_type == ModelType.BODY:
        model = VoViT_b(pretrained=True, debug={}).to(DEVICE)
    else:
        model = VoViT_fb(pretrained=True, debug={}).to(DEVICE)
    model.eval()

    results = defaultdict(lambda: defaultdict(list))
    print(f"Find attention scores for all {len(dataset)} samples...")
    for sample in tqdm(dataloader):
        artist_name = sample["artist"][0]
        song_name = sample["song"][0]
        audio_attention, video_attention = find_attentions(
            model, sample, DEVICE
        )

        if audio_attention is not None and video_attention is not None:
            results[artist_name][song_name].append(
                {
                    "audio_attention": audio_attention.tolist(),
                    "video_attention": video_attention.tolist(),
                }
            )

    output_path = f"attention_scores_{args.model}.json"
    print(f"Saving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VoViT Full Dataset Attention Analysis"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vocal",
        choices=["vocal", "violin"],
        help="Type of VoViT model to use",
    )

    args = parser.parse_args()
    main(args)

import torch
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
vovit_path = os.path.abspath(os.getenv("VOVIT_PATH"))
sys.path.insert(0, vovit_path)

from vovit import VoViT_b, VoViT_f, VoViT_fb  # type: ignore
from vovit.display.dataloaders_new import ModelType  # type: ignore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est = est.float().flatten()
    ref = ref.float().flatten()
    ref_zm = ref - ref.mean()
    est_zm = est - est.mean()
    s_target = (torch.dot(est_zm, ref_zm) / (ref_zm.pow(2).sum() + eps)) * ref_zm
    e_noise = est_zm - s_target
    ratio = (s_target.pow(2).sum() + eps) / (e_noise.pow(2).sum() + eps)
    return 10 * torch.log10(ratio + eps)

def load_model(model_str: str, device: str = DEVICE):
    model_type_map = {"face": ModelType.FACE, "body": ModelType.BODY, "face_body": ModelType.BODY_FACE}
    model_type = model_type_map[model_str]
    if model_type == ModelType.FACE:
        model = VoViT_f(pretrained=True, debug={}).to(device)
    elif model_type == ModelType.BODY:
        model = VoViT_b(pretrained=True, debug={}).to(device)
    else:
        model = VoViT_fb(pretrained=True, debug={}).to(device)
    model.eval()
    return model, model_type

def _extract_modalities(sample: Dict[str, Any], device: str = DEVICE):
    mix = sample["mix"].float().to(device)
    face = (
        torch.nan_to_num(sample["face"].float(), nan=0.0).to(device)
        if ("face" in sample and sample["face"] is not None)
        else None
    )
    body = (
        torch.nan_to_num(sample["body"].float(), nan=0.0).to(device)
        if ("body" in sample and sample["body"] is not None)
        else None
    )
    return mix, face, body

def forward_model(model, sample: Dict[str, Any], return_none_if_missing: bool = True):
    mix, face, body = _extract_modalities(sample)
    if isinstance(model, VoViT_f):
        if face is None and return_none_if_missing:
            return None
        out = model.forward(mix, face)
    elif isinstance(model, VoViT_b):
        if body is None and return_none_if_missing:
            return None
        out = model.forward(mix, body)
    else:  # VoViT_fb
        if (face is None or body is None) and return_none_if_missing:
            return None
        out = model.forward(mix, face, body)
    return out["estimated_wav"].detach().cpu()

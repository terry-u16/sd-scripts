import argparse
import math
import os
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from library import anima_utils, sai_model_spec, train_util
from library.lora_utils import load_safetensors_with_lora_and_fp8
from library.safetensors_utils import WeightTransformHooks
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def str_to_dtype(precision: str | None):
    if precision == "float":
        return torch.float
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def load_state_dict(file_name: str, dtype: torch.dtype | None):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = train_util.load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    if dtype is not None:
        for key in list(sd.keys()):
            if isinstance(sd[key], torch.Tensor) and sd[key].dtype.is_floating_point:
                sd[key] = sd[key].to(dtype)

    return sd, metadata


def save_to_file(file_name: str, model: dict[str, torch.Tensor], metadata: dict[str, str]):
    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(model, file_name, metadata=metadata)
    else:
        torch.save(model, file_name)


def ensure_destination_dir(save_to: str):
    assert save_to is not None, "save_to must be specified / save_toを指定してください"
    dest_dir = os.path.dirname(save_to)
    if dest_dir and not os.path.exists(dest_dir):
        logger.info(f"creating directory: {dest_dir}")
        os.makedirs(dest_dir)


def normalize_lora_key(key: str) -> str | None:
    if key.startswith("lora_unet_"):
        return key

    # Common Anima/ComfyUI style:
    # diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight
    # -> lora_unet_blocks_0_self_attn_q_proj.lora_down.weight
    if key.startswith("diffusion_model.") and ".lora_" in key:
        module_name, suffix = key[len("diffusion_model.") :].rsplit(".lora_", 1)
        if suffix == "A.weight":
            return f"lora_unet_{module_name.replace('.', '_')}.lora_down.weight"
        if suffix == "B.weight":
            return f"lora_unet_{module_name.replace('.', '_')}.lora_up.weight"
        if suffix == "alpha":
            return f"lora_unet_{module_name.replace('.', '_')}.alpha"

    return None


def normalize_lora_state_dict(file_name: str, lora_sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    normalized_sd = {}
    unused_keys = []

    for key, value in lora_sd.items():
        normalized_key = normalize_lora_key(key)
        if normalized_key is None:
            unused_keys.append(key)
            continue
        normalized_sd[normalized_key] = value

    missing_pairs = []
    module_names = set()
    for key in normalized_sd.keys():
        if ".lora_down.weight" in key:
            module_names.add(key[: key.rfind(".lora_down.weight")])
        elif ".lora_up.weight" in key:
            module_names.add(key[: key.rfind(".lora_up.weight")])

    for module_name in sorted(module_names):
        down_key = module_name + ".lora_down.weight"
        up_key = module_name + ".lora_up.weight"
        if down_key not in normalized_sd or up_key not in normalized_sd:
            missing_pairs.append(module_name)

    if missing_pairs:
        raise ValueError(
            f"LoRA weights in {file_name} have incomplete up/down pairs: {missing_pairs[:5]}"
            f"{'...' if len(missing_pairs) > 5 else ''}"
        )

    return normalized_sd, unused_keys


def filter_dit_lora_state_dict(file_name: str, lora_sd: dict[str, torch.Tensor], allow_partial: bool) -> dict[str, torch.Tensor]:
    dit_lora_sd, unused_keys = normalize_lora_state_dict(file_name, lora_sd)

    if unused_keys:
        message = (
            f"{file_name} contains {len(unused_keys)} non-DiT LoRA keys. "
            "Only lora_unet_* or diffusion_model.*.lora_A/B.weight keys can be merged into an Anima DiT checkpoint."
            " / Anima DiTへマージできるのはlora_unet_*またはdiffusion_model.*.lora_A/B.weightのみです。"
        )
        if allow_partial:
            logger.warning(message)
            logger.warning("Ignoring non-DiT LoRA keys because --allow_partial is specified.")
        else:
            raise ValueError(message + " Use --allow_partial to ignore them.")

    if not dit_lora_sd:
        raise ValueError(f"No lora_unet_* keys found in {file_name} / lora_unet_*のキーが見つかりません")

    return dit_lora_sd


def merge_to_dit_model(args, merge_dtype: torch.dtype | None, save_dtype: torch.dtype | None):
    assert args.dit is not None, "dit must be specified / ditを指定してください"
    assert args.models is not None and len(args.models) > 0, "models must be specified / modelsを指定してください"
    assert len(args.models) == len(
        args.ratios
    ), "number of models must be equal to number of ratios / モデルの数と重みの数は合わせてください"

    if os.path.splitext(args.save_to)[1] != ".safetensors":
        raise ValueError("Anima DiT checkpoints must be saved as .safetensors / Anima DiTは.safetensorsで保存してください")

    lora_weights_list = []
    for model in args.models:
        logger.info(f"loading LoRA model: {model}")
        lora_sd, _ = load_state_dict(model, merge_dtype)
        lora_weights_list.append(filter_dit_lora_state_dict(model, lora_sd, args.allow_partial))

    calc_device = torch.device(args.working_device)
    logger.info(f"loading Anima DiT model: {args.dit}")
    logger.info(f"merging LoRA weights into DiT. ratios: {args.ratios}")

    rename_hooks = WeightTransformHooks(rename_hook=anima_utils.strip_anima_state_dict_prefix)
    dit_state_dict = load_safetensors_with_lora_and_fp8(
        model_files=args.dit,
        lora_weights_list=lora_weights_list,
        lora_multipliers=args.ratios,
        fp8_optimization=False,
        calc_device=calc_device,
        move_to_device=False,
        dit_weight_dtype=merge_dtype,
        disable_numpy_memmap=args.disable_mmap,
        weight_transform_hooks=rename_hooks,
    )

    if args.no_metadata:
        sai_metadata = None
    else:
        merged_from = sai_model_spec.build_merged_from([args.dit] + args.models)
        title = os.path.splitext(os.path.basename(args.save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(
            None,
            False,
            False,
            False,
            False,
            False,
            time.time(),
            title=title,
            merged_from=merged_from,
            model_config={"anima": "preview"},
            is_stable_diffusion_ckpt=True,
        )

    logger.info(f"saving Anima DiT model to: {args.save_to}")
    anima_utils.save_anima_model(args.save_to, dit_state_dict, sai_metadata, save_dtype)


def merge_lora_models(models, ratios, merge_dtype, concat=False, shuffle=False):
    base_alphas = {}
    base_dims = {}
    merged_sd = {}
    base_model = None

    for model, ratio in zip(models, ratios):
        logger.info(f"loading: {model}")
        lora_sd, lora_metadata = load_state_dict(model, merge_dtype)
        lora_sd, unused_keys = normalize_lora_state_dict(model, lora_sd)
        if unused_keys:
            logger.warning(f"Ignoring {len(unused_keys)} unsupported LoRA keys in {model}.")

        if any("hada_" in key or "lokr_" in key for key in lora_sd.keys()):
            raise ValueError("LoHa/LoKr model-to-model merging is not supported by this script.")

        if lora_metadata is not None and base_model is None:
            base_model = lora_metadata.get(train_util.SS_METADATA_KEY_BASE_MODEL_VERSION, None)

        alphas = {}
        dims = {}
        for key in lora_sd.keys():
            if "alpha" in key:
                lora_module_name = key[: key.rfind(".alpha")]
                alpha = float(lora_sd[key].detach().float().cpu().numpy())
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha
            elif "lora_down" in key:
                lora_module_name = key[: key.rfind(".lora_down")]
                dim = lora_sd[key].size()[0]
                dims[lora_module_name] = dim
                if lora_module_name not in base_dims:
                    base_dims[lora_module_name] = dim

        if not dims:
            raise ValueError(f"No LoRA weights found in {model} / LoRAの重みが見つかりません")

        for lora_module_name in dims.keys():
            if lora_module_name not in alphas:
                alpha = dims[lora_module_name]
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha

        logger.info(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")
        logger.info("merging...")
        for key in tqdm(lora_sd.keys()):
            if "alpha" in key:
                continue

            if "lora_up" in key and concat:
                concat_dim = 1
            elif "lora_down" in key and concat:
                concat_dim = 0
            else:
                concat_dim = None

            if ".lora_" not in key:
                continue

            lora_module_name = key[: key.rfind(".lora_")]
            base_alpha = base_alphas[lora_module_name]
            alpha = alphas[lora_module_name]
            scale = math.sqrt(alpha / base_alpha) * ratio
            scale = abs(scale) if "lora_up" in key else scale

            if key in merged_sd:
                assert (
                    merged_sd[key].size() == lora_sd[key].size() or concat_dim is not None
                ), "weights shape mismatch. Different dims can only be merged with --concat. / 重みのサイズが合いません。次元数が異なる場合は--concatを指定してください"
                if concat_dim is not None:
                    merged_sd[key] = torch.cat([merged_sd[key], lora_sd[key] * scale], dim=concat_dim)
                else:
                    merged_sd[key] = merged_sd[key] + lora_sd[key] * scale
            else:
                merged_sd[key] = lora_sd[key] * scale

    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)
        if shuffle:
            key_down = lora_module_name + ".lora_down.weight"
            key_up = lora_module_name + ".lora_up.weight"
            dim = merged_sd[key_down].shape[0]
            perm = torch.randperm(dim)
            merged_sd[key_down] = merged_sd[key_down][perm]
            merged_sd[key_up] = merged_sd[key_up][:, perm]

    logger.info("merged LoRA models")
    logger.info(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    dims_list = list(set(base_dims.values()))
    alphas_list = list(set(base_alphas.values()))
    dims = f"{dims_list[0]}" if len(dims_list) == 1 else "Dynamic"
    alphas = f"{alphas_list[0]}" if len(alphas_list) == 1 else "Dynamic"
    metadata = train_util.build_minimum_network_metadata(None, base_model, "networks.lora_anima", dims, alphas, None)

    return merged_sd, metadata


def merge(args):
    if args.models is None:
        args.models = []
    if args.ratios is None:
        args.ratios = []
    assert len(args.models) == len(
        args.ratios
    ), "number of models must be equal to number of ratios / モデルの数と重みの数は合わせてください"

    ensure_destination_dir(args.save_to)

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    if args.dit is not None:
        merge_to_dit_model(args, merge_dtype, save_dtype)
        return

    state_dict, metadata = merge_lora_models(args.models, args.ratios, merge_dtype, args.concat, args.shuffle)

    if save_dtype is not None:
        for key in list(state_dict.keys()):
            value = state_dict[key]
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point and value.dtype != save_dtype:
                state_dict[key] = value.to(save_dtype)

    logger.info("calculating hashes and creating metadata...")
    model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash

    if not args.no_metadata:
        merged_from = sai_model_spec.build_merged_from(args.models)
        title = os.path.splitext(os.path.basename(args.save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(
            state_dict,
            False,
            False,
            False,
            True,
            False,
            time.time(),
            title=title,
            merged_from=merged_from,
            model_config={"anima": "preview"},
        )
        metadata.update(sai_metadata)

    logger.info(f"saving LoRA model to: {args.save_to}")
    save_to_file(args.save_to, state_dict, metadata)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving, same to merging if omitted / 保存時に精度を変更して保存する、省略時はマージ時の精度と同じ",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        choices=["float", "fp16", "bf16"],
        help="precision in merging (float is recommended) / マージの計算時の精度（floatを推奨）",
    )
    parser.add_argument(
        "--dit",
        type=str,
        default=None,
        help="Anima DiT model to load: safetensors file. If omitted, LoRA models are merged together. / 読み込むAnima DiTモデル。省略時はLoRAモデル同士をマージする",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="destination file name: safetensors file / 保存先のファイル名、safetensors",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="LoRA models to merge: safetensors file / マージするLoRAモデル、safetensorsファイル",
    )
    parser.add_argument("--ratios", type=float, nargs="*", help="ratios for each model / それぞれのLoRAモデルの比率")
    parser.add_argument(
        "--working_device",
        type=str,
        default="cpu",
        help="device to work on while merging into DiT / DiTへのマージ計算に使用するデバイス",
    )
    parser.add_argument(
        "--disable_mmap",
        action="store_true",
        help="disable numpy memmap while loading the DiT checkpoint / DiT読み込み時にnumpy memmapを無効化する",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="ignore non-DiT LoRA keys such as lora_te_* when merging into --dit / --ditへのマージ時にlora_te_*などDiT以外のLoRAキーを無視する",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata (minimum ss_metadata for LoRA is saved) / "
        + "sai modelspecのメタデータを保存しない（LoRA同士のマージでは最低限のss_metadataは保存される）",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="concat lora instead of merge (The dim/rank of the output LoRA is the sum of the input dims) / "
        + "マージの代わりに結合する（LoRAのdim/rankは入力dimの合計になる）",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle LoRA weights / LoRAの重みをシャッフルする",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    merge(args)

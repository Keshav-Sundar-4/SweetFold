#!/usr/bin/env python3

import copy
import os
import sys

import torch
import omegaconf.dictconfig


# ---------------------------------------------------------------------
# User-configurable paths
# ---------------------------------------------------------------------
#
# Before running this script, replace these paths with the correct absolute
# paths on your system.
#
# base_ckpt_path:
#   Original/base checkpoint used before glycan training.
#
# trained_ckpt_path:
#   Checkpoint produced by training. This is usually:
#     <output path from full.yaml>/checkpoints/last.ckpt
#
# out_ckpt_path_inference:
#   Final checkpoint for inference or future pretrained loading.
#
# out_ckpt_path_resume:
#   Final checkpoint for resuming training.
# ---------------------------------------------------------------------

base_ckpt_path = "/path/to/sweetfold_env/weights/boltz1_conf_converted.ckpt"
trained_ckpt_path = "/path/to/training_outputs/checkpoints/last.ckpt"

out_ckpt_path_inference = "/path/to/sweetfold_env/weights/boltz1_glycan.ckpt"
out_ckpt_path_resume = "/path/to/sweetfold_env/weights/boltz1_glycan_resume.ckpt"


# Newer PyTorch versions have torch.serialization.add_safe_globals.
# Older PyTorch versions do not, so this must be guarded.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])


def load_checkpoint_compatible(path):
    """
    Load a checkpoint in a way that works across PyTorch versions.

    Older PyTorch versions may not support weights_only=False.
    Newer PyTorch versions may require or benefit from weights_only=False
    when loading Lightning checkpoints that contain non-tensor metadata.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def clone_tensor_if_possible(value):
    """
    Clone tensors when possible; otherwise return the object as-is.

    Most state_dict entries are tensors, but this helper avoids crashing if a
    non-tensor value appears.
    """
    if hasattr(value, "clone"):
        return value.clone()
    return value


def create_inference_checkpoint(base_ckpt, trained_ckpt, output_path):
    """
    Create an inference/pretrained-loading checkpoint.

    This updates model weights only. It is not intended for resuming training,
    because optimizer and scheduler states may not match.
    """
    print("\n" + "-" * 80)
    print(f"METHOD 1: Creating INFERENCE checkpoint -> {os.path.basename(output_path)}")
    print("-" * 80)

    base_copy = copy.deepcopy(base_ckpt)

    base_sd = base_copy.get("state_dict", base_copy)
    trained_sd = trained_ckpt.get("state_dict", trained_ckpt)

    updated_keys = 0
    for key, tensor in trained_sd.items():
        if key in base_sd:
            updated_keys += 1
        base_sd[key] = clone_tensor_if_possible(tensor)

    print(f"Merged {len(trained_sd)} parameter tensors from the trained checkpoint.")
    print(f"Overwrote {updated_keys} existing keys in the base state_dict.")

    if "state_dict" in base_copy:
        base_copy["state_dict"] = base_sd
    else:
        base_copy = base_sd

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(base_copy, output_path)
    print("Saved inference checkpoint successfully.")


def create_resume_checkpoint(base_ckpt, trained_ckpt, output_path):
    """
    Create a resumable checkpoint.

    This uses the trained checkpoint as the foundation, preserving model weights,
    optimizer state, scheduler state, epoch, and global_step from the training run.
    Missing frozen/base weights are copied from the base checkpoint.
    """
    print("\n" + "-" * 80)
    print(f"METHOD 2: Creating RESUMABLE checkpoint -> {os.path.basename(output_path)}")
    print("-" * 80)

    new_checkpoint = copy.deepcopy(trained_ckpt)
    print("[Step 1] Initialized from the trained checkpoint to preserve optimizer/scheduler state.")

    base_sd = base_ckpt.get("state_dict", base_ckpt)
    trained_sd = trained_ckpt.get("state_dict", trained_ckpt)

    final_sd = dict(trained_sd)
    added_from_base = 0

    for key, tensor in base_sd.items():
        if key not in final_sd:
            final_sd[key] = clone_tensor_if_possible(tensor)
            added_from_base += 1

    new_checkpoint["state_dict"] = final_sd

    print(f"[Step 2] Merged model weights. Final state_dict contains {len(final_sd)} tensors.")
    if added_from_base > 0:
        print(f"Transferred {added_from_base} missing parameter(s) from the base checkpoint.")

    print("[Step 3] Verifying final checkpoint for resume-critical keys...")
    all_keys_present = True

    for key in ["epoch", "global_step", "state_dict", "optimizer_states", "lr_schedulers"]:
        if key in new_checkpoint and new_checkpoint[key] is not None:
            print(f"  {key}: Found")
        else:
            print(f"  {key}: MISSING. This checkpoint may not be resumable.")
            all_keys_present = False

    if not all_keys_present:
        print(
            "\nWARNING: One or more critical keys for resuming were missing. Proceed with caution.",
            file=sys.stderr,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(new_checkpoint, output_path)
    print("Saved resumable checkpoint successfully.")


if __name__ == "__main__":
    print("=" * 80)
    print("Starting checkpoint creation process.")
    print("=" * 80)

    print("Base checkpoint:")
    print(f"  {base_ckpt_path}")

    print("Trained checkpoint:")
    print(f"  {trained_ckpt_path}")

    print("Inference checkpoint output:")
    print(f"  {out_ckpt_path_inference}")

    print("Resume checkpoint output:")
    print(f"  {out_ckpt_path_resume}")

    if not os.path.exists(base_ckpt_path):
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt_path}")

    if not os.path.exists(trained_ckpt_path):
        raise FileNotFoundError(f"Trained checkpoint not found: {trained_ckpt_path}")

    base_checkpoint = load_checkpoint_compatible(base_ckpt_path)
    trained_checkpoint = load_checkpoint_compatible(trained_ckpt_path)

    create_inference_checkpoint(
        base_checkpoint,
        trained_checkpoint,
        out_ckpt_path_inference,
    )

    create_resume_checkpoint(
        base_checkpoint,
        trained_checkpoint,
        out_ckpt_path_resume,
    )

    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETE")
    print(f"Inference checkpoint saved to: {out_ckpt_path_inference}")
    print(f"Resume checkpoint saved to:    {out_ckpt_path_resume}")
    print("=" * 80)

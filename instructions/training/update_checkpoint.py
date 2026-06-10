#!/usr/bin/env python3

import copy
import os
import sys

import torch
import omegaconf.dictconfig


# Newer PyTorch versions have torch.serialization.add_safe_globals.
# Your torch==2.2.2+cu118 does not, so this must be guarded.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])


def load_checkpoint_compatible(path):
    """
    Load a checkpoint in a way that works across torch versions.

    torch 2.2.2 does not have:
      - torch.serialization.add_safe_globals
      - torch.serialization.safe_globals

    Newer torch versions may require/benefit from weights_only=False when loading
    Lightning checkpoints that contain non-tensor metadata.
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
    Creates an inference/pretrained-loading checkpoint.

    This updates model weights only. It is not suitable for resuming training
    because optimizer/scheduler states may not match.
    """
    print("\n" + "-" * 80)
    print(f"METHOD 1: Creating INFERENCE-ONLY checkpoint -> {os.path.basename(output_path)}")
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
    print(f"   > Overwrote {updated_keys} existing keys in the base state_dict.")

    if "state_dict" in base_copy:
        base_copy["state_dict"] = base_sd
    else:
        base_copy = base_sd

    torch.save(base_copy, output_path)
    print("Saved new INFERENCE checkpoint successfully.")


def create_resume_checkpoint(base_ckpt, trained_ckpt, output_path):
    """
    Creates a resumable checkpoint.

    This uses the trained checkpoint as the foundation, preserving model weights,
    optimizer state, scheduler state, epoch, and global_step from the training run.
    Missing frozen/base weights are copied from the base checkpoint.
    """
    print("\n" + "-" * 80)
    print(f"METHOD 2: Creating RESUMABLE checkpoint -> {os.path.basename(output_path)}")
    print("-" * 80)

    new_checkpoint = copy.deepcopy(trained_ckpt)
    print("[Step 1] Initialized from the TRAINED checkpoint to preserve optimizer/scheduler state.")

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
        print(f"         > Transferred {added_from_base} missing parameter(s) from the base checkpoint.")

    print("[Step 3] Verifying final checkpoint for resume-critical keys...")
    all_keys_present = True

    for key in ["epoch", "global_step", "state_dict", "optimizer_states", "lr_schedulers"]:
        if key in new_checkpoint and new_checkpoint[key] is not None:
            print(f"  {key}: Found")
        else:
            print(f"  {key}: MISSING! This checkpoint may not be resumable.")
            all_keys_present = False

    if not all_keys_present:
        print(
            "\nWARNING: One or more critical keys for resuming were missing. Proceed with caution.",
            file=sys.stderr,
        )

    torch.save(new_checkpoint, output_path)
    print("Saved new RESUMABLE checkpoint successfully.")


if __name__ == "__main__":
    base_ckpt_path = "/work/keshavsundar/env/sweetfold/weights/boltz1_conf_converted.ckpt"
    trained_ckpt_path = "/work/keshavsundar/work_sundar/glycan_test/checkpoints/last.ckpt"

    out_ckpt_path_inference = "/work/keshavsundar/env/sweetfold/weights/boltz1_glycan.ckpt"
    out_ckpt_path_resume = "/work/keshavsundar/env/sweetfold/weights/boltz1_glycan_resume.ckpt"

    print("=" * 80)
    print("Starting checkpoint creation process for both Inference and Resume...")
    print("=" * 80)

    output_dir = os.path.dirname(out_ckpt_path_inference)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading base checkpoint from:\n  {base_ckpt_path}")
    base_checkpoint = load_checkpoint_compatible(base_ckpt_path)

    print(f"Loading trained checkpoint from:\n  {trained_ckpt_path}")
    trained_checkpoint = load_checkpoint_compatible(trained_ckpt_path)

    create_inference_checkpoint(base_checkpoint, trained_checkpoint, out_ckpt_path_inference)

    create_resume_checkpoint(base_checkpoint, trained_checkpoint, out_ckpt_path_resume)

    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETE")
    print(f"Inference Checkpoint saved to: {out_ckpt_path_inference}")
    print(f"Resume Checkpoint saved to:    {out_ckpt_path_resume}")
    print("=" * 80)
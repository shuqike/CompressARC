import argparse
import ast
import csv
import html
import json
import math
from pathlib import Path

import numpy as np


DIM_NAMES = ["example", "color", "direction", "x", "y"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect one ARC task through preprocessing, latent init, forward trace, and short training."
    )
    parser.add_argument("--split", default="training", choices=["training", "evaluation", "test"])
    parser.add_argument("--task", default="272f95fa")
    parser.add_argument("--steps", type=int, default=5, help="Number of optimizer steps to run after init inspection.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="inspect_outputs")
    parser.add_argument("--top-k", type=int, default=8, help="How many high-KL latent tensors to visualize.")
    return parser.parse_args()


def choose_device(torch, requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False.")
    return requested


def configure_torch(torch, device, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)


def dims_key(dims):
    return "[" + ",".join(str(int(x)) for x in dims) + "]"


def dims_title(dims):
    active = [name for name, exists in zip(DIM_NAMES, dims) if exists]
    return dims_key(dims) + " " + "(" + ",".join(active) + ")"


def original_color_grid(task, restricted_grid, shape=None):
    arr = np.array(restricted_grid)
    if shape is not None:
        arr = arr[: shape[0], : shape[1]]
    mapper = np.array(task.colors)
    return mapper[arr]


def format_grid(grid):
    arr = np.array(grid)
    return "\n".join(" ".join(str(int(v)) for v in row) for row in arr)


def tensor_stats(tensor):
    import torch

    detached = tensor.detach()
    shape = list(detached.shape)
    device = str(detached.device)
    requires_grad = bool(getattr(tensor, "requires_grad", False))
    if detached.numel() == 0:
        return {
            "shape": shape,
            "device": device,
            "requires_grad": requires_grad,
            "numel": 0,
        }
    values = detached.float().cpu()
    return {
        "shape": shape,
        "device": device,
        "requires_grad": requires_grad,
        "numel": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "l2": float(torch.linalg.vector_norm(values).item()),
    }


def compact_stat_line(stats):
    return (
        f"shape={stats['shape']} mean={stats.get('mean', float('nan')):.4g} "
        f"std={stats.get('std', float('nan')):.4g} "
        f"min={stats.get('min', float('nan')):.4g} max={stats.get('max', float('nan')):.4g}"
    )


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def import_project_modules(torch, device, seed):
    # arc_compressor sets the default device to cuda at import time. Import first,
    # then set the requested device again so this inspector can run on CPU too.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import arc_compressor
    import layers
    import preprocessing
    import solution_selection
    import train
    import visualization

    configure_torch(torch, device, seed)
    return {
        "arc_compressor": arc_compressor,
        "layers": layers,
        "preprocessing": preprocessing,
        "solution_selection": solution_selection,
        "train": train,
        "visualization": visualization,
        "plt": plt,
    }


def inspect_task(task, out_dir):
    print("\n== Task / preprocessing ==")
    print(f"task: {task.task_name}")
    print(f"n_train={task.n_train} n_test={task.n_test} n_examples={task.n_examples}")
    print(f"n_x={task.n_x} n_y={task.n_y} restricted_colors={task.colors}")
    print(f"in_out_same_size={task.in_out_same_size}")
    print(f"all_in_same_size={task.all_in_same_size}")
    print(f"all_out_same_size={task.all_out_same_size}")
    for i, shape_pair in enumerate(task.shapes):
        split_name = "train" if i < task.n_train else "test"
        print(f"example {i} ({split_name}) input_shape={shape_pair[0]} output_shape={shape_pair[1]}")

    print("\nRaw grids from the challenge file:")
    for split_name in ["train", "test"]:
        for i, example in enumerate(task.unprocessed_problem[split_name]):
            global_i = i if split_name == "train" else task.n_train + i
            print(f"\n[{split_name} example {i} / global {global_i}] input")
            print(format_grid(example["input"]))
            if "output" in example:
                print("output")
                print(format_grid(example["output"]))
            elif task.solution is not None:
                print("held-out solution (not used by training)")
                solution_grid = original_color_grid(
                    task,
                    task.solution[i].detach().cpu().numpy(),
                    task.shapes[global_i][1],
                )
                print(format_grid(solution_grid))
            else:
                print("output: ?")

    mask_summary = {
        "problem_tensor": tensor_stats(task.problem),
        "masks": tensor_stats(task.masks),
        "shapes": task.shapes,
        "colors": task.colors,
        "n_train": task.n_train,
        "n_test": task.n_test,
    }
    write_json(out_dir / "task_summary.json", mask_summary)
    print(f"\nWrote preprocessing summary: {out_dir / 'task_summary.json'}")


def grid_to_rgb(grid, color_list):
    grid = np.array(grid)
    return color_list[grid]


def draw_grid(ax, grid, color_list, title):
    ax.set_title(title, fontsize=8)
    if grid is None:
        ax.text(0.5, 0.5, "?", ha="center", va="center", fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return
    grid = np.array(grid, dtype=int)
    ax.imshow(grid_to_rgb(grid, color_list), interpolation="none")
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color=(0.25, 0.25, 0.25), linewidth=0.4)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def plot_task_grids(task, visualization, plt, out_dir):
    color_list = visualization.color_list
    rows = task.n_examples
    cols = 3 if task.solution is not None else 2
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.5 * rows), squeeze=False)
    for global_i in range(task.n_examples):
        if global_i < task.n_train:
            split_name = "train"
            local_i = global_i
        else:
            split_name = "test"
            local_i = global_i - task.n_train
        example = task.unprocessed_problem[split_name][local_i]
        draw_grid(axes[global_i][0], example["input"], color_list, f"{split_name} {local_i} input")
        target = example.get("output")
        draw_grid(axes[global_i][1], target, color_list, f"{split_name} {local_i} output")
        if task.solution is not None:
            heldout = None
            if split_name == "test":
                heldout = original_color_grid(
                    task,
                    task.solution[local_i].detach().cpu().numpy(),
                    task.shapes[global_i][1],
                )
            draw_grid(axes[global_i][2], heldout, color_list, f"{split_name} {local_i} solution")
    fig.tight_layout()
    path = out_dir / "task_grids.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Wrote task grid plot: {path}")


def collect_latent_rows(task, model):
    rows = []
    state = {}
    for dims in task.multitensor_system:
        mean, local_capacity = model.multiposteriors[dims]
        target_capacity = model.target_capacities[dims]
        decode_weight, decode_bias = model.decode_weights[dims]
        key = dims_key(dims)
        rows.append(
            {
                "dims": key,
                "title": dims_title(dims),
                "posterior_mean": tensor_stats(mean),
                "local_capacity_adjustment": tensor_stats(local_capacity),
                "target_capacity": tensor_stats(target_capacity),
                "decode_weight": tensor_stats(decode_weight),
                "decode_bias": tensor_stats(decode_bias),
            }
        )
        state[key] = {
            "mean": mean.detach().cpu().clone(),
            "local_capacity": local_capacity.detach().cpu().clone(),
            "target_capacity": target_capacity.detach().cpu().clone(),
        }
    return rows, state


def flatten_latent_rows(rows):
    flat = []
    for row in rows:
        for name in ["posterior_mean", "local_capacity_adjustment", "target_capacity", "decode_weight", "decode_bias"]:
            stats = row[name]
            flat.append(
                {
                    "dims": row["dims"],
                    "title": row["title"],
                    "component": name,
                    "shape": stats.get("shape"),
                    "numel": stats.get("numel"),
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                    "min": stats.get("min"),
                    "max": stats.get("max"),
                    "l2": stats.get("l2"),
                    "requires_grad": stats.get("requires_grad"),
                    "device": stats.get("device"),
                }
            )
    return flat


def compare_latent_state(task, model, initial_state):
    rows = []
    for dims in task.multitensor_system:
        key = dims_key(dims)
        mean, local_capacity = model.multiposteriors[dims]
        target_capacity = model.target_capacities[dims]
        current = {
            "mean": mean.detach().cpu(),
            "local_capacity": local_capacity.detach().cpu(),
            "target_capacity": target_capacity.detach().cpu(),
        }
        row = {"dims": key, "title": dims_title(dims)}
        for name, tensor in current.items():
            delta = tensor - initial_state[key][name]
            row[f"{name}_delta_l2"] = float(np.linalg.norm(delta.numpy().reshape(-1)))
            row[f"{name}_delta_max_abs"] = float(np.max(np.abs(delta.numpy())))
        rows.append(row)
    return rows


def decode_latent_summary(task, model, layers, out_dir, top_k):
    with model_no_grad():
        decoded, kl_amounts, kl_names = layers.decode_latents(
            model.target_capacities, model.decode_weights, model.multiposteriors
        )

    kl_rows = []
    for kl, name in zip(kl_amounts, kl_names):
        dims = ast.literal_eval(name)
        kl_rows.append(
            {
                "dims": dims_key(dims),
                "title": dims_title(dims),
                "kl_sum": float(kl.detach().sum().cpu().item()),
                "kl_mean": float(kl.detach().float().mean().cpu().item()),
                "kl_shape": list(kl.shape),
                "decoded": tensor_stats(decoded[dims]),
            }
        )
    kl_rows.sort(key=lambda row: row["kl_sum"], reverse=True)
    write_json(out_dir / "decoded_latents_initial.json", kl_rows)
    write_csv(
        out_dir / "decoded_latents_initial.csv",
        [
            {
                "dims": row["dims"],
                "title": row["title"],
                "kl_sum": row["kl_sum"],
                "kl_mean": row["kl_mean"],
                "kl_shape": row["kl_shape"],
                "decoded_shape": row["decoded"]["shape"],
                "decoded_mean": row["decoded"].get("mean"),
                "decoded_std": row["decoded"].get("std"),
                "decoded_l2": row["decoded"].get("l2"),
            }
            for row in kl_rows
        ],
        [
            "dims",
            "title",
            "kl_sum",
            "kl_mean",
            "kl_shape",
            "decoded_shape",
            "decoded_mean",
            "decoded_std",
            "decoded_l2",
        ],
    )
    print("\n== Initial decoded latent KL, top components ==")
    for row in kl_rows[: min(8, len(kl_rows))]:
        print(f"{row['title']}: KL_sum={row['kl_sum']:.4g}; decoded {compact_stat_line(row['decoded'])}")

    selected = [ast.literal_eval(row["dims"]) for row in kl_rows[:top_k]]
    ensure_core_dims(selected)
    return decoded, kl_rows, selected


class model_no_grad:
    def __enter__(self):
        import torch

        self.context = torch.no_grad()
        return self.context.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self.context.__exit__(exc_type, exc, tb)


def ensure_core_dims(dims_list):
    for dims in ([1, 1, 0, 1, 1], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1]):
        if dims not in dims_list:
            dims_list.append(dims)


def projection_for_plot(tensor, dims):
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim > 0:
        arr = np.sqrt(np.mean(arr * arr, axis=-1))
    active = [idx for idx, exists in enumerate(dims) if exists]

    axis = 0
    kept = []
    for dim_idx in active:
        if dim_idx in (3, 4):
            kept.append(dim_idx)
            axis += 1
        else:
            arr = np.mean(arr, axis=axis)

    if arr.ndim == 0:
        return arr.reshape(1, 1), "collapsed", "collapsed"
    if arr.ndim == 1:
        if kept and kept[0] == 3:
            return arr.reshape(-1, 1), "x", "collapsed"
        if kept and kept[0] == 4:
            return arr.reshape(1, -1), "collapsed", "y"
        return arr.reshape(1, -1), "collapsed", "collapsed"
    while arr.ndim > 2:
        arr = np.mean(arr, axis=0)
    y_label = "x" if 3 in kept else "collapsed"
    x_label = "y" if 4 in kept else "collapsed"
    return arr, y_label, x_label


def plot_tensor_getter(plt, dims_list, getter, path, title):
    if not dims_list:
        return
    n = len(dims_list)
    fig, axes = plt.subplots(n, 1, figsize=(5.5, 2.0 * n), squeeze=False)
    for ax, dims in zip(axes[:, 0], dims_list):
        arr, y_label, x_label = projection_for_plot(getter(dims), dims)
        im = ax.imshow(arr, aspect="auto", cmap="viridis")
        ax.set_title(dims_title(dims), fontsize=8)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Wrote tensor plot: {path}")


def forward_with_trace(model, layers):
    trace = []
    x, kl_amounts, kl_names = layers.decode_latents(
        model.target_capacities, model.decode_weights, model.multiposteriors
    )
    trace.append(("decode_latents", x))

    for layer_num in range(model.n_layers):
        x = layers.share_up(x, model.share_up_weights[layer_num])
        trace.append((f"layer{layer_num}.share_up", x))
        x = layers.softmax(x, model.softmax_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)
        trace.append((f"layer{layer_num}.softmax", x))
        x = layers.cummax(
            x,
            model.cummax_weights[layer_num],
            model.multitensor_system.task.masks,
            pre_norm=False,
            post_norm=True,
            use_bias=False,
        )
        trace.append((f"layer{layer_num}.cummax", x))
        x = layers.shift(
            x,
            model.shift_weights[layer_num],
            model.multitensor_system.task.masks,
            pre_norm=False,
            post_norm=True,
            use_bias=False,
        )
        trace.append((f"layer{layer_num}.shift", x))
        x = layers.direction_share(x, model.direction_share_weights[layer_num], pre_norm=True, use_bias=False)
        trace.append((f"layer{layer_num}.direction_share", x))
        x = layers.nonlinear(x, model.nonlinear_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)
        trace.append((f"layer{layer_num}.nonlinear", x))
        x = layers.share_down(x, model.share_down_weights[layer_num])
        trace.append((f"layer{layer_num}.share_down", x))
        x = layers.normalize(x)
        trace.append((f"layer{layer_num}.normalize", x))

    output = layers.affine(x[[1, 1, 0, 1, 1]], model.head_weights, use_bias=False) + 100 * model.head_weights[1]
    x_mask = layers.affine(x[[1, 0, 0, 1, 0]], model.mask_weights, use_bias=True)
    y_mask = layers.affine(x[[1, 0, 0, 0, 1]], model.mask_weights, use_bias=True)
    x_mask, y_mask = layers.postprocess_mask(model.multitensor_system.task, x_mask, y_mask)
    return output, x_mask, y_mask, kl_amounts, kl_names, trace


def trace_stats(task, trace):
    rows = []
    for stage_index, (stage, multitensor) in enumerate(trace):
        for dims in task.multitensor_system:
            stats = tensor_stats(multitensor[dims])
            rows.append(
                {
                    "stage_index": stage_index,
                    "stage": stage,
                    "dims": dims_key(dims),
                    "title": dims_title(dims),
                    "shape": stats.get("shape"),
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                    "min": stats.get("min"),
                    "max": stats.get("max"),
                    "l2": stats.get("l2"),
                    "numel": stats.get("numel"),
                }
            )
    return rows


def write_trace_snapshot(task, model, layers, plt, selected_dims, out_dir, label):
    with model_no_grad():
        output, x_mask, y_mask, kl_amounts, kl_names, trace = forward_with_trace(model, layers)
    rows = trace_stats(task, trace)
    write_json(out_dir / f"forward_trace_{label}.json", rows)
    write_csv(
        out_dir / f"forward_trace_{label}.csv",
        rows,
        ["stage_index", "stage", "dims", "title", "shape", "mean", "std", "min", "max", "l2", "numel"],
    )

    print(f"\n== Forward trace ({label}) ==")
    print(f"stages={len(trace)} tensors_per_stage={len(list(task.multitensor_system))}")
    for stage_index in [0, 1, len(trace) // 2, len(trace) - 1]:
        stage = trace[stage_index][0]
        stage_rows = [row for row in rows if row["stage_index"] == stage_index]
        mean_l2 = np.mean([row["l2"] for row in stage_rows if row["l2"] is not None])
        max_std = np.max([row["std"] for row in stage_rows if row["std"] is not None])
        print(f"stage {stage_index:02d} {stage}: mean_l2={mean_l2:.4g} max_std={max_std:.4g}")
    print(f"Wrote forward trace CSV/JSON for {label}")

    plot_trace_norms(plt, rows, selected_dims[: min(5, len(selected_dims))], out_dir / f"forward_trace_norms_{label}.png")
    return output, x_mask, y_mask, rows


def plot_trace_norms(plt, rows, selected_dims, path):
    selected_keys = [dims_key(dims) for dims in selected_dims]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    stage_names = []
    for key in selected_keys:
        key_rows = [row for row in rows if row["dims"] == key]
        key_rows.sort(key=lambda row: row["stage_index"])
        if not stage_names:
            stage_names = [row["stage"] for row in key_rows]
        ax.plot([row["stage_index"] for row in key_rows], [row["l2"] for row in key_rows], marker="o", label=key)
    ax.set_yscale("log")
    ax.set_xlabel("forward stage")
    ax.set_ylabel("tensor L2 norm")
    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names, rotation=80, ha="right", fontsize=7)
    ax.grid(True, which="both", linewidth=0.4)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Wrote trace norm plot: {path}")


def mask_select_logprobs(mask, length):
    import torch

    logprobs = []
    for offset in range(mask.shape[0] - length + 1):
        logprob = -torch.sum(mask[:offset])
        logprob = logprob + torch.sum(mask[offset : offset + length])
        logprob = logprob - torch.sum(mask[offset + length :])
        logprobs.append(logprob)
    logprobs = torch.stack(logprobs, dim=0)
    log_partition = torch.logsumexp(logprobs, dim=0)
    return log_partition, logprobs


def compute_loss_snapshot(task, model, train_step):
    import torch

    logits, x_mask, y_mask, kl_amounts, kl_names = model.forward()
    logits = torch.cat([torch.zeros_like(logits[:, :1, :, :]), logits], dim=1)

    total_kl = sum(torch.sum(kl_amount) for kl_amount in kl_amounts)
    reconstruction_error = torch.tensor(0.0, device=logits.device)

    for example_num in range(task.n_examples):
        for in_out_mode in range(2):
            if example_num >= task.n_train and in_out_mode == 1:
                continue

            fixed_size = (
                task.in_out_same_size
                or (task.all_out_same_size and in_out_mode == 1)
                or (task.all_in_same_size and in_out_mode == 0)
            )
            grid_size_uncertain = not fixed_size
            coefficient = 0.01 ** max(0, 1 - train_step / 100) if grid_size_uncertain else 1
            logits_slice = logits[example_num, :, :, :, in_out_mode]
            problem_slice = task.problem[example_num, :, :, in_out_mode]
            output_shape = task.shapes[example_num][in_out_mode]
            x_log_partition, x_logprobs = mask_select_logprobs(
                coefficient * x_mask[example_num, :, in_out_mode], output_shape[0]
            )
            y_log_partition, y_logprobs = mask_select_logprobs(
                coefficient * y_mask[example_num, :, in_out_mode], output_shape[1]
            )

            if grid_size_uncertain:
                x_log_partitions = [
                    mask_select_logprobs(coefficient * x_mask[example_num, :, in_out_mode], length)[0]
                    for length in range(1, x_mask.shape[1] + 1)
                ]
                y_log_partitions = [
                    mask_select_logprobs(coefficient * y_mask[example_num, :, in_out_mode], length)[0]
                    for length in range(1, y_mask.shape[1] + 1)
                ]
                x_log_partition = torch.logsumexp(torch.stack(x_log_partitions, dim=0), dim=0)
                y_log_partition = torch.logsumexp(torch.stack(y_log_partitions, dim=0), dim=0)

            logprobs = [[] for _ in range(x_logprobs.shape[0])]
            for x_offset in range(x_logprobs.shape[0]):
                for y_offset in range(y_logprobs.shape[0]):
                    logprob = (
                        x_logprobs[x_offset]
                        - x_log_partition
                        + y_logprobs[y_offset]
                        - y_log_partition
                    )
                    logits_crop = logits_slice[
                        :, x_offset : x_offset + output_shape[0], y_offset : y_offset + output_shape[1]
                    ]
                    target_crop = problem_slice[: output_shape[0], : output_shape[1]]
                    logprob = logprob - torch.nn.functional.cross_entropy(
                        logits_crop[None, ...], target_crop[None, ...], reduction="sum"
                    )
                    logprobs[x_offset].append(logprob)
            logprobs = torch.stack([torch.stack(items, dim=0) for items in logprobs], dim=0)
            coefficient = 0.1 ** max(0, 1 - train_step / 100) if grid_size_uncertain else 1
            logprob = torch.logsumexp(coefficient * logprobs, dim=(0, 1)) / coefficient
            reconstruction_error = reconstruction_error - logprob

    loss = total_kl + 10 * reconstruction_error
    return {
        "logits": logits,
        "x_mask": x_mask,
        "y_mask": y_mask,
        "kl_amounts": kl_amounts,
        "kl_names": kl_names,
        "total_kl": total_kl,
        "reconstruction_error": reconstruction_error,
        "loss": loss,
    }


def best_slice_point(mask, length):
    import torch

    if length is None:
        search_lengths = list(range(1, mask.shape[0] + 1))
    else:
        search_lengths = [length]
    max_logprob = None
    best_start = None
    best_end = None
    for candidate_length in search_lengths:
        logprobs = torch.stack(
            [
                -torch.sum(mask[:offset])
                + torch.sum(mask[offset : offset + candidate_length])
                - torch.sum(mask[offset + candidate_length :])
                for offset in range(mask.shape[0] - candidate_length + 1)
            ]
        )
        current = torch.max(logprobs)
        if max_logprob is None or current > max_logprob:
            max_logprob = current
            best_start = int(torch.argmax(logprobs).item())
            best_end = best_start + candidate_length
    return best_start, best_end


def predicted_grid_for(task, logits, x_mask, y_mask, example_num, mode_num):
    prediction = torch_argmax(logits[example_num, :, :, :, mode_num], dim=0)
    fixed_length = None
    if mode_num == 0:
        fixed_length = task.shapes[example_num][0]
    elif example_num < task.n_train or task.in_out_same_size or task.all_out_same_size:
        fixed_length = task.shapes[example_num][mode_num]
    if fixed_length is None:
        x_len = None
        y_len = None
    else:
        x_len, y_len = fixed_length
    x_start, x_end = best_slice_point(x_mask[example_num, :, mode_num], x_len)
    y_start, y_end = best_slice_point(y_mask[example_num, :, mode_num], y_len)
    restricted = prediction[x_start:x_end, y_start:y_end].detach().cpu().numpy()
    return original_color_grid(task, restricted)


def torch_argmax(tensor, dim):
    import torch

    return torch.argmax(tensor, dim=dim)


def plot_prediction_snapshot(task, visualization, plt, snapshot, out_dir, label):
    logits = snapshot["logits"]
    x_mask = snapshot["x_mask"]
    y_mask = snapshot["y_mask"]
    color_list = visualization.color_list

    cols = 4
    rows = task.n_examples
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.4 * rows), squeeze=False)
    for example_num in range(task.n_examples):
        if example_num < task.n_train:
            split_name = "train"
            local_i = example_num
        else:
            split_name = "test"
            local_i = example_num - task.n_train
        example = task.unprocessed_problem[split_name][local_i]
        draw_grid(axes[example_num][0], example["input"], color_list, f"{split_name} {local_i} input")
        draw_grid(axes[example_num][1], example.get("output"), color_list, f"{split_name} {local_i} target")
        pred_input = predicted_grid_for(task, logits, x_mask, y_mask, example_num, 0)
        pred_output = predicted_grid_for(task, logits, x_mask, y_mask, example_num, 1)
        draw_grid(axes[example_num][2], pred_input, color_list, "pred input")
        draw_grid(axes[example_num][3], pred_output, color_list, "pred output")
    fig.suptitle(
        f"{task.task_name} {label}: KL={float(snapshot['total_kl'].detach().cpu()):.4g}, "
        f"recon={float(snapshot['reconstruction_error'].detach().cpu()):.4g}, "
        f"loss={float(snapshot['loss'].detach().cpu()):.4g}",
        fontsize=10,
    )
    fig.tight_layout()
    path = out_dir / f"prediction_{label}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Wrote prediction snapshot: {path}")

    write_mask_plot(task, plt, x_mask, y_mask, out_dir / f"masks_{label}.png", label)


def write_mask_plot(task, plt, x_mask, y_mask, path, label):
    x_vals = x_mask.detach().cpu().numpy()
    y_vals = y_mask.detach().cpu().numpy()
    rows = task.n_examples
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 1.2 * rows), squeeze=False)
    for example_num in range(task.n_examples):
        for mode_num, mode_name in enumerate(["input", "output"]):
            ax = axes[example_num][2 * mode_num]
            ax.imshow(x_vals[example_num, :, mode_num][None, :], aspect="auto", cmap="coolwarm")
            ax.set_title(f"ex{example_num} {mode_name} x_mask", fontsize=7)
            ax.set_yticks([])
            ax = axes[example_num][2 * mode_num + 1]
            ax.imshow(y_vals[example_num, :, mode_num][None, :], aspect="auto", cmap="coolwarm")
            ax.set_title(f"ex{example_num} {mode_name} y_mask", fontsize=7)
            ax.set_yticks([])
    fig.suptitle(f"mask logits {label}", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Wrote mask snapshot: {path}")


def write_loss_snapshot(snapshot, out_dir, label):
    rows = []
    for kl, name in zip(snapshot["kl_amounts"], snapshot["kl_names"]):
        rows.append(
            {
                "dims": name,
                "kl_sum": float(kl.detach().sum().cpu().item()),
                "kl_mean": float(kl.detach().float().mean().cpu().item()),
                "kl_shape": list(kl.shape),
            }
        )
    rows.sort(key=lambda row: row["kl_sum"], reverse=True)
    obj = {
        "total_kl": float(snapshot["total_kl"].detach().cpu().item()),
        "reconstruction_error": float(snapshot["reconstruction_error"].detach().cpu().item()),
        "loss": float(snapshot["loss"].detach().cpu().item()),
        "kl_components": rows,
    }
    write_json(out_dir / f"loss_{label}.json", obj)
    print(
        f"{label}: total_KL={obj['total_kl']:.4g} "
        f"reconstruction={obj['reconstruction_error']:.4g} loss={obj['loss']:.4g}"
    )


def shape_text(shape):
    return "[" + " x ".join(str(item) for item in shape) + "]"


def md_table(headers, rows):
    escaped_headers = [str(header) for header in headers]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for value in row:
            value = str(value).replace("\n", "<br>")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def axis_summary(task, dims):
    lengths = task.multitensor_system.dim_lengths
    parts = []
    for name, exists, length in zip(DIM_NAMES, dims, lengths):
        if exists:
            parts.append(f"{name}={length}")
    return ", ".join(parts) if parts else "scalar"


def source_reference_for_dims(dims):
    if dims[2] == 0:
        channel = "16"
        source = "ARCCompressor.channel_dim_fn(): direction bit is 0"
    else:
        channel = "8"
        source = "ARCCompressor.channel_dim_fn(): direction bit is 1"
    return channel, source


def make_latent_shape_table(task, model, kl_rows):
    kl_by_dims = {row["dims"]: row for row in kl_rows}
    rows = []
    for dims in task.multitensor_system:
        channel_dim = model.channel_dim_fn(dims)
        key = dims_key(dims)
        mean, local_capacity = model.multiposteriors[dims]
        target_capacity = model.target_capacities[dims]
        decode_weight, decode_bias = model.decode_weights[dims]
        decoded_shape = task.multitensor_system.shape(dims, channel_dim)
        kl_sum = kl_by_dims.get(key, {}).get("kl_sum")
        rows.append(
            {
                "dims": key,
                "axes": axis_summary(task, dims),
                "channel": channel_dim,
                "posterior_mean": shape_text(list(mean.shape)),
                "local_capacity": shape_text(list(local_capacity.shape)),
                "target_capacity": shape_text(list(target_capacity.shape)),
                "decode_weight": shape_text(list(decode_weight.shape)),
                "decode_bias": shape_text(list(decode_bias.shape)),
                "decoded_hidden": shape_text(decoded_shape),
                "init_kl_sum": f"{kl_sum:.4g}" if kl_sum is not None else "",
            }
        )
    return rows


def make_forward_flow_rows(model):
    rows = [["00", "decode_latents", "posterior z -> hidden MultiTensor", "per dims: [active axes, C(dims)]"]]
    stage_index = 1
    for layer_num in range(model.n_layers):
        block = [
            ("share_up", "cross-MultiTensor upward communication", "shape preserved"),
            ("softmax", "softmax features over non-channel axes, then residual projection", "shape preserved; internal feature dim is 2*(2^N - 1)"),
            ("cummax", "directional cumulative max", "shape preserved; active for [1,1,1,1,1] and [1,0,1,1,1]"),
            ("shift", "directional shift", "shape preserved; active for [1,1,1,1,1] and [1,0,1,1,1]"),
            ("direction_share", "8-direction communication", "shape preserved; active only when direction bit is 1"),
            ("nonlinear", "SiLU residual MLP", "shape preserved"),
            ("share_down", "cross-MultiTensor downward communication", "shape preserved"),
            ("normalize", "mean/variance normalization over non-channel axes", "shape preserved"),
        ]
        for op_name, description, shape_effect in block:
            rows.append([f"{stage_index:02d}", f"layer{layer_num}.{op_name}", description, shape_effect])
            stage_index += 1
    rows.extend(
        [
            ["head", "color logits", "x[[1,1,0,1,1]] -> affine head", "[example, color, x, y, 2] before black color is added"],
            ["head", "x_mask", "x[[1,0,0,1,0]] -> mask affine", "[example, x, 2]"],
            ["head", "y_mask", "x[[1,0,0,0,1]] -> mask affine", "[example, y, 2]"],
        ]
    )
    return rows


def make_weight_family_rows(model):
    return [
        [
            "latent posterior mean",
            "all dims",
            "[active axes, 4]",
            "trainable mean before channel_layer normalization",
        ],
        [
            "latent local capacity adjustment",
            "all dims",
            "[active axes, 4]",
            "initialized zeros, trainable",
        ],
        ["target capacity", "all dims", "[4]", "initialized zeros, trainable"],
        ["decode linear", "all dims", "[4, C] + [C]", "C=16 without direction, C=8 with direction"],
        ["share_up residual", "4 layers, all dims", "[C, 16] + [16], then [16, C] + [C]", "bias exists but layer call uses use_bias=False"],
        ["share_down residual", "4 layers, all dims", "[C, 8] + [8], then [8, C] + [C]", "bias exists but layer call uses use_bias=False"],
        [
            "softmax residual",
            "4 layers, all dims",
            "[C, 2] + [2], then [2*(2^N - 1), C] + [C]",
            "N=sum(color,direction,x,y bits)",
        ],
        ["cummax residual", "4 layers, all dims", "[C, 4] + [4], then [4, C] + [C]", "operation itself only affects directional x/y tensors"],
        ["shift residual", "4 layers, all dims", "[C, 4] + [4], then [4, C] + [C]", "operation itself only affects directional x/y tensors"],
        ["direction_share", "4 layers, direction dims", "8 x 8 maps of [C, C] + [C]", "applied only when direction bit is 1"],
        ["nonlinear residual", "4 layers, all dims", "[C, 16] + [16], then [16, C] + [C]", "SiLU in the middle"],
        ["color head", "dims [1,1,0,1,1]", "[16, 2] + [2]", "2 is input/output mode channel"],
        ["mask head", "dims [1,0,0,1,0] and [1,0,0,0,1]", "[16, 2] + [2]", "shared x/y mask weight"],
    ]


def make_selected_trace_rows(trace_rows, selected_dims):
    selected_keys = [dims_key(dims) for dims in selected_dims[:5]]
    stage_indices = sorted({row["stage_index"] for row in trace_rows})
    rows = []
    for stage_index in stage_indices:
        stage_rows = [row for row in trace_rows if row["stage_index"] == stage_index]
        stage = stage_rows[0]["stage"]
        row = [f"{stage_index:02d}", stage]
        for key in selected_keys:
            match = next(item for item in stage_rows if item["dims"] == key)
            row.append(f"{shape_text(ast.literal_eval(str(match['shape'])))} / L2 {match['l2']:.3g}")
        rows.append(row)
    headers = ["#", "stage"] + selected_keys
    return headers, rows


def write_shape_report_md(out_dir, task, model, latent_shape_rows, kl_rows, trace_rows, selected_dims, snapshot):
    nonblack_colors = task.n_colors
    sections = []
    sections.append(f"# Shape report: {task.task_name}")
    sections.append(
        "\n".join(
            [
                "## Task shape capsule",
                "",
                md_table(
                    ["item", "shape/value", "meaning"],
                    [
                        ["n_examples", task.n_examples, f"{task.n_train} train + {task.n_test} test"],
                        ["restricted colors", task.colors, "black is included in task.colors; model color axis excludes black before the final concat"],
                        ["n_colors", nonblack_colors, "non-black color count used inside ARCCompressor"],
                        ["n_x / n_y", f"{task.n_x} / {task.n_y}", "max padded spatial size for this task"],
                        ["task.problem", shape_text(list(task.problem.shape)), "[example, x, y, input_output] after color argmax"],
                        ["task.masks", shape_text(list(task.masks.shape)), "[example, x, y, input_output] valid-pixel mask"],
                        ["model output before black concat", shape_text([task.n_examples, task.n_colors, task.n_x, task.n_y, 2]), "[example, nonblack_color, x, y, input_output]"],
                        ["training logits after black concat", shape_text([task.n_examples, task.n_colors + 1, task.n_x, task.n_y, 2]), "adds zero-logit black channel"],
                        ["x_mask", shape_text([task.n_examples, task.n_x, 2]), "[example, x, input_output]"],
                        ["y_mask", shape_text([task.n_examples, task.n_y, 2]), "[example, y, input_output]"],
                    ],
                ),
            ]
        )
    )
    sections.append(
        "\n".join(
            [
                "## Axis convention",
                "",
                "`dims = [example, color, direction, x, y]`. A 1 means that axis is present in that multitensor; a 0 means it is collapsed.",
                "",
                md_table(
                    ["axis", "length in this task", "note"],
                    [
                        ["example", task.n_examples, "train examples first, then test examples"],
                        ["color", task.n_colors, "non-black restricted colors only"],
                        ["direction", 8, "8 compass directions"],
                        ["x", task.n_x, "padded height"],
                        ["y", task.n_y, "padded width"],
                        ["channel", "C(dims)", "16 when direction bit is 0; 8 when direction bit is 1"],
                    ],
                ),
            ]
        )
    )
    sections.append(
        "\n".join(
            [
                "## Neural architecture flow",
                "",
                "The residual hidden state is a MultiTensor. Every valid `dims` key carries a tensor shaped `[active axes, C(dims)]` after `decode_latents`; the repeated blocks preserve that shape.",
                "",
                md_table(["#", "operation", "what it does", "shape effect"], make_forward_flow_rows(model)),
            ]
        )
    )
    sections.append(
        "\n".join(
            [
                "## Weight families",
                "",
                md_table(["family", "scope", "shape formula", "note"], make_weight_family_rows(model)),
            ]
        )
    )
    latent_rows_for_md = [
        [
            row["dims"],
            row["axes"],
            row["channel"],
            row["posterior_mean"],
            row["target_capacity"],
            row["decode_weight"],
            row["decoded_hidden"],
            row["init_kl_sum"],
        ]
        for row in latent_shape_rows
    ]
    sections.append(
        "\n".join(
            [
                "## Latent and hidden state shapes",
                "",
                md_table(
                    ["dims", "active axes", "C", "posterior mean", "target cap", "decode W", "decoded hidden", "init KL"],
                    latent_rows_for_md,
                ),
            ]
        )
    )
    trace_headers, trace_table_rows = make_selected_trace_rows(trace_rows, selected_dims)
    sections.append(
        "\n".join(
            [
                "## Selected forward hidden-state trace",
                "",
                "Each cell is `shape / L2 norm`. Shapes stay constant through the residual block; the norm shows how the values move.",
                "",
                md_table(trace_headers, trace_table_rows),
            ]
        )
    )
    sections.append(
        "\n".join(
            [
                "## Initial loss snapshot",
                "",
                md_table(
                    ["quantity", "value"],
                    [
                        ["total KL", f"{float(snapshot['total_kl'].detach().cpu()):.6g}"],
                        ["reconstruction error", f"{float(snapshot['reconstruction_error'].detach().cpu()):.6g}"],
                        ["loss", f"{float(snapshot['loss'].detach().cpu()):.6g}"],
                    ],
                ),
            ]
        )
    )
    sections.append(
        "\n".join(
            [
                "## Visual references",
                "",
                "![task grids](task_grids.png)",
                "",
                "![initial trace norms](forward_trace_norms_initial.png)",
                "",
                "![initial posterior mean](posterior_mean_initial.png)",
                "",
                "![initial decoded latents](decoded_latents_initial.png)",
            ]
        )
    )
    path = out_dir / "shape_report.md"
    path.write_text("\n\n".join(sections), encoding="utf-8")
    return path


def html_table(headers, rows, class_name=""):
    header_html = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    row_html = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        row_html.append(f"<tr>{cells}</tr>")
    return f"<table class=\"{class_name}\"><thead><tr>{header_html}</tr></thead><tbody>{''.join(row_html)}</tbody></table>"


def write_shape_report_html(out_dir, task, model, latent_shape_rows, trace_rows, selected_dims, snapshot):
    latent_rows = [
        [
            row["dims"],
            row["axes"],
            row["channel"],
            row["posterior_mean"],
            row["local_capacity"],
            row["target_capacity"],
            row["decode_weight"],
            row["decode_bias"],
            row["decoded_hidden"],
            row["init_kl_sum"],
        ]
        for row in latent_shape_rows
    ]
    trace_headers, trace_rows_selected = make_selected_trace_rows(trace_rows, selected_dims)
    task_rows = [
        ["n_examples", task.n_examples, f"{task.n_train} train + {task.n_test} test"],
        ["restricted colors", task.colors, "model uses non-black colors internally"],
        ["n_x / n_y", f"{task.n_x} / {task.n_y}", "max padded spatial extent"],
        ["task.problem", shape_text(list(task.problem.shape)), "[example, x, y, input_output]"],
        ["task.masks", shape_text(list(task.masks.shape)), "[example, x, y, input_output]"],
        ["output logits before black", shape_text([task.n_examples, task.n_colors, task.n_x, task.n_y, 2]), "[example, nonblack_color, x, y, input_output]"],
        ["training logits after black", shape_text([task.n_examples, task.n_colors + 1, task.n_x, task.n_y, 2]), "black zero-logit channel added in train.take_step"],
        ["x_mask / y_mask", f"{shape_text([task.n_examples, task.n_x, 2])} / {shape_text([task.n_examples, task.n_y, 2])}", "mask logits for crop selection"],
    ]
    css = """
body { margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; color: #17202a; background: #f6f7f9; }
.page { max-width: 1220px; margin: 0 auto; padding: 28px; }
.hero { background: #111827; color: white; padding: 28px 32px; border-radius: 8px; margin-bottom: 22px; }
.hero h1 { margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }
.hero p { margin: 0; color: #cbd5e1; }
.cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
.card { background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 14px 16px; }
.metric { color: #667085; font-size: 12px; text-transform: uppercase; }
.value { margin-top: 6px; font-size: 22px; font-weight: 700; }
section { background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 18px; margin: 16px 0; }
h2 { margin: 0 0 12px; font-size: 20px; }
p { line-height: 1.5; }
code, .shape { background: #eef2f7; border: 1px solid #d8dee8; padding: 1px 5px; border-radius: 4px; font-family: Consolas, monospace; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { position: sticky; top: 0; background: #f1f5f9; color: #334155; text-align: left; border-bottom: 1px solid #cbd5e1; padding: 8px; white-space: nowrap; }
td { border-bottom: 1px solid #e5e7eb; padding: 7px 8px; vertical-align: top; }
tr:nth-child(even) td { background: #fafafa; }
.flow td:first-child, .latent td:first-child { font-family: Consolas, monospace; white-space: nowrap; }
.images { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.images img { width: 100%; border: 1px solid #d8dee8; border-radius: 8px; background: white; }
.caption { font-size: 12px; color: #667085; margin-top: 6px; }
@media (max-width: 900px) { .cards, .images { grid-template-columns: 1fr; } .page { padding: 14px; } }
"""
    body = f"""
<div class="page">
  <div class="hero">
    <h1>Shape report: {html.escape(task.task_name)}</h1>
    <p>dims = [example, color, direction, x, y]. Hidden state is a per-task MultiTensor; each valid dims key has shape [active axes, C(dims)].</p>
  </div>
  <div class="cards">
    <div class="card"><div class="metric">examples</div><div class="value">{task.n_examples}</div></div>
    <div class="card"><div class="metric">non-black colors</div><div class="value">{task.n_colors}</div></div>
    <div class="card"><div class="metric">spatial pad</div><div class="value">{task.n_x} x {task.n_y}</div></div>
    <div class="card"><div class="metric">valid multitensors</div><div class="value">{len(list(task.multitensor_system))}</div></div>
  </div>
  <section>
    <h2>Task Shape Capsule</h2>
    <div class="table-wrap">{html_table(["item", "shape/value", "meaning"], task_rows)}</div>
  </section>
  <section>
    <h2>Architecture Flow</h2>
    <p>The 4 repeated blocks preserve every hidden tensor's shape. The interesting shape changes happen at latent decode, inside softmax's temporary feature expansion, and at the heads.</p>
    <div class="table-wrap">{html_table(["#", "operation", "what it does", "shape effect"], make_forward_flow_rows(model), "flow")}</div>
  </section>
  <section>
    <h2>Weight Families</h2>
    <div class="table-wrap">{html_table(["family", "scope", "shape formula", "note"], make_weight_family_rows(model))}</div>
  </section>
  <section>
    <h2>Latent And Hidden State Shapes</h2>
    <p>C(dims) is 16 when the direction bit is 0 and 8 when the direction bit is 1. The latent width is always 4.</p>
    <div class="table-wrap">{html_table(["dims", "active axes", "C", "posterior mean", "local cap", "target cap", "decode W", "decode b", "decoded hidden", "init KL"], latent_rows, "latent")}</div>
  </section>
  <section>
    <h2>Selected Forward Hidden-State Trace</h2>
    <p>Each cell is shape plus L2 norm. This is useful for checking where values grow or get normalized.</p>
    <div class="table-wrap">{html_table(trace_headers, trace_rows_selected)}</div>
  </section>
  <section>
    <h2>Initial Loss Snapshot</h2>
    <div class="table-wrap">{html_table(["quantity", "value"], [
        ["total KL", f"{float(snapshot['total_kl'].detach().cpu()):.6g}"],
        ["reconstruction error", f"{float(snapshot['reconstruction_error'].detach().cpu()):.6g}"],
        ["loss", f"{float(snapshot['loss'].detach().cpu()):.6g}"],
    ])}</div>
  </section>
  <section>
    <h2>Visual References</h2>
    <div class="images">
      <div><img src="task_grids.png"><div class="caption">Task grids and held-out solution for orientation.</div></div>
      <div><img src="forward_trace_norms_initial.png"><div class="caption">Norms across the copied forward pass.</div></div>
      <div><img src="posterior_mean_initial.png"><div class="caption">Initial posterior mean RMS projections.</div></div>
      <div><img src="decoded_latents_initial.png"><div class="caption">Initial decoded latent RMS projections.</div></div>
    </div>
  </section>
</div>
"""
    path = out_dir / "shape_report.html"
    path.write_text(f"<!doctype html><html><head><meta charset=\"utf-8\"><title>Shape report {html.escape(task.task_name)}</title><style>{css}</style></head><body>{body}</body></html>", encoding="utf-8")
    return path


def write_shape_reports(out_dir, task, model, kl_rows, trace_rows, selected_dims, snapshot):
    latent_shape_rows = make_latent_shape_table(task, model, kl_rows)
    md_path = write_shape_report_md(out_dir, task, model, latent_shape_rows, kl_rows, trace_rows, selected_dims, snapshot)
    html_path = write_shape_report_html(out_dir, task, model, latent_shape_rows, trace_rows, selected_dims, snapshot)
    print(f"Wrote shape report Markdown: {md_path}")
    print(f"Wrote shape report HTML: {html_path}")
    return md_path, html_path


def write_readme(out_dir, task_name, split, steps, device):
    lines = [
        f"# Inspect output for {split}/{task_name}",
        "",
        f"Device: {device}",
        f"Training steps: {steps}",
        "",
        "Useful files:",
        "- task_summary.json: preprocessing shapes, color mapping, mask/problem tensor stats.",
        "- latent_initialization.json/csv: posterior mean, local capacity, target capacity, decode weight stats.",
        "- decoded_latents_initial.json/csv: output of layers.decode_latents() and KL by multitensor dims.",
        "- posterior_mean_initial.png and decoded_latents_initial.png: RMS projections of high-KL tensors.",
        "- forward_trace_initial.csv/json: every recorded forward stage for every multitensor dims.",
        "- forward_trace_norms_initial.png: selected tensor norms through the copied forward pass.",
        "- prediction_step_*.png and masks_step_*.png: model predictions and mask logits at snapshots.",
        "- latent_change_after_training.csv: how latent parameters moved after the requested optimizer steps.",
        "- shape_report.md and shape_report.html: formatted shape report for latent, architecture, hidden states, and heads.",
    ]
    path = out_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()

    import torch

    device = choose_device(torch, args.device)
    modules = import_project_modules(torch, device, args.seed)
    preprocessing = modules["preprocessing"]
    arc_compressor = modules["arc_compressor"]
    layers = modules["layers"]
    train = modules["train"]
    solution_selection = modules["solution_selection"]
    visualization = modules["visualization"]
    plt = modules["plt"]

    out_dir = Path(args.out_dir) / f"{args.split}_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_readme(out_dir, args.task, args.split, args.steps, device)

    print(f"Using device={device}")
    task = preprocessing.preprocess_tasks(args.split, [args.task])[0]
    inspect_task(task, out_dir)
    plot_task_grids(task, visualization, plt, out_dir)

    model = arc_compressor.ARCCompressor(task)
    optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
    logger = solution_selection.Logger(task)

    latent_rows, initial_state = collect_latent_rows(task, model)
    write_json(out_dir / "latent_initialization.json", latent_rows)
    write_csv(
        out_dir / "latent_initialization.csv",
        flatten_latent_rows(latent_rows),
        [
            "dims",
            "title",
            "component",
            "shape",
            "numel",
            "mean",
            "std",
            "min",
            "max",
            "l2",
            "requires_grad",
            "device",
        ],
    )
    print("\n== Latent initialization examples ==")
    for row in latent_rows[:5]:
        print(f"{row['title']}")
        print(f"  posterior_mean: {compact_stat_line(row['posterior_mean'])}")
        print(f"  local_capacity_adjustment: {compact_stat_line(row['local_capacity_adjustment'])}")
        print(f"  target_capacity: {compact_stat_line(row['target_capacity'])}")
        print(f"  decode_weight: {compact_stat_line(row['decode_weight'])}")
    print(f"Wrote latent initialization CSV/JSON in {out_dir}")

    decoded, kl_rows, selected_dims = decode_latent_summary(task, model, layers, out_dir, args.top_k)
    plot_tensor_getter(
        plt,
        selected_dims,
        lambda dims: model.multiposteriors[dims][0],
        out_dir / "posterior_mean_initial.png",
        "Initial posterior mean RMS projections",
    )
    plot_tensor_getter(
        plt,
        selected_dims,
        lambda dims: decoded[dims],
        out_dir / "decoded_latents_initial.png",
        "Initial decoded latent RMS projections",
    )

    _, _, _, initial_trace_rows = write_trace_snapshot(task, model, layers, plt, selected_dims, out_dir, "initial")

    with model_no_grad():
        snapshot = compute_loss_snapshot(task, model, 0)
    write_loss_snapshot(snapshot, out_dir, "step_0000")
    write_shape_reports(out_dir, task, model, kl_rows, initial_trace_rows, selected_dims, snapshot)
    plot_prediction_snapshot(task, visualization, plt, snapshot, out_dir, "step_0000")

    if args.steps > 0:
        print(f"\n== Training for {args.steps} inspect steps ==")
    for step in range(args.steps):
        train.take_step(task, model, optimizer, step, logger)
        if step + 1 in {1, args.steps}:
            with model_no_grad():
                snapshot = compute_loss_snapshot(task, model, step + 1)
            label = f"step_{step + 1:04d}"
            write_loss_snapshot(snapshot, out_dir, label)
            plot_prediction_snapshot(task, visualization, plt, snapshot, out_dir, label)

    if args.steps > 0:
        write_trace_snapshot(task, model, layers, plt, selected_dims, out_dir, f"step_{args.steps:04d}")

    change_rows = compare_latent_state(task, model, initial_state)
    write_csv(
        out_dir / "latent_change_after_training.csv",
        change_rows,
        [
            "dims",
            "title",
            "mean_delta_l2",
            "mean_delta_max_abs",
            "local_capacity_delta_l2",
            "local_capacity_delta_max_abs",
            "target_capacity_delta_l2",
            "target_capacity_delta_max_abs",
        ],
    )
    print(f"\nWrote latent-change summary: {out_dir / 'latent_change_after_training.csv'}")
    print(f"Done. Open {out_dir / 'README.md'} for the artifact map.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.layers as layers
from models.losses import IGNORE_LABEL_ID


def install_torch_tropical_fallback() -> None:
    """Use exact PyTorch max/min-plus kernels when tropical-gemm is unavailable."""

    def maxplus_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.contiguous().to(torch.float32)
        b = b.contiguous().to(torch.float32)
        return (a.unsqueeze(2) + b.unsqueeze(0)).amax(dim=1)

    def maxplus_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.contiguous().to(torch.float32)
        b = b.contiguous().to(torch.float32)
        return (a.unsqueeze(3) + b.unsqueeze(1)).amax(dim=2)

    def minplus_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.contiguous().to(torch.float32)
        b = b.contiguous().to(torch.float32)
        return (a.unsqueeze(3) + b.unsqueeze(1)).amin(dim=2)

    layers._tg_maxplus_mm = maxplus_mm
    layers._tg_maxplus_bmm = maxplus_bmm
    layers._tg_minplus_bmm = minplus_bmm


install_torch_tropical_fallback()

from models.TAPR.tapr import TAPR, TAPRLossHead
from models.tarm.tarm import TARM
from models.urm.urm import URM


@dataclass
class ArrayDataset:
    name: str
    train_inputs: np.ndarray
    train_labels: np.ndarray
    test_inputs: np.ndarray
    test_labels: np.ndarray
    ood_inputs: Optional[np.ndarray]
    ood_labels: Optional[np.ndarray]
    vocab_size: int
    seq_len: int
    train_puzzle_identifiers: Optional[np.ndarray] = None
    test_puzzle_identifiers: Optional[np.ndarray] = None
    ood_puzzle_identifiers: Optional[np.ndarray] = None
    num_puzzle_identifiers: int = 1
    puzzle_emb_ndim: int = 0
    ignore_label_id: Optional[int] = None
    task_residual_type: str = "none"
    grid_size: int = 0
    clrs_nodes: int = 0
    clrs_max_weight: int = 9
    clrs_dist_cap: int = 32


@dataclass
class Variant:
    name: str
    model_cls: type
    config: Dict[str, object]
    tapr_loss: bool = False
    loss_type: str = "softmax_cross_entropy"


def _sudoku_csv_to_arrays(url: str, limit: int) -> Tuple[np.ndarray, np.ndarray]:
    inputs, labels = [], []
    with urllib.request.urlopen(url, timeout=120) as response:
        lines = (line.decode("utf-8") for line in response)
        reader = csv.reader(lines)
        next(reader)
        for row_idx, row in enumerate(reader):
            if row_idx >= limit:
                break
            _, question, answer, _ = row
            question = question.replace(".", "0")
            inputs.append(np.frombuffer(question.encode("ascii"), dtype=np.uint8).astype(np.int64) - ord("0") + 1)
            labels.append(np.frombuffer(answer.encode("ascii"), dtype=np.uint8).astype(np.int64) - ord("0") + 1)
    return np.asarray(inputs, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def _load_or_download_sudoku_arrays(
    data_path: Path,
    cache_path: Path,
    *,
    train_limit: int,
    test_limit: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_inputs_path = data_path / "train" / "all__inputs.npy"
    train_labels_path = data_path / "train" / "all__labels.npy"
    test_inputs_path = data_path / "test" / "all__inputs.npy"
    test_labels_path = data_path / "test" / "all__labels.npy"

    if all(p.exists() for p in (train_inputs_path, train_labels_path, test_inputs_path, test_labels_path)):
        return (
            np.load(train_inputs_path, mmap_mode="r"),
            np.load(train_labels_path, mmap_mode="r"),
            np.load(test_inputs_path, mmap_mode="r"),
            np.load(test_labels_path, mmap_mode="r"),
        )

    cache_path.mkdir(parents=True, exist_ok=True)
    cached_train_inputs = cache_path / f"train_{train_limit}__inputs.npy"
    cached_train_labels = cache_path / f"train_{train_limit}__labels.npy"
    cached_test_inputs = cache_path / f"test_{test_limit}__inputs.npy"
    cached_test_labels = cache_path / f"test_{test_limit}__labels.npy"

    if not all(p.exists() for p in (cached_train_inputs, cached_train_labels, cached_test_inputs, cached_test_labels)):
        print(
            "Sudoku NPY arrays are missing locally; downloading official sapientinc/sudoku-extreme CSV subsets.",
            flush=True,
        )
        train_inputs, train_labels = _sudoku_csv_to_arrays(
            "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/train.csv",
            train_limit,
        )
        test_inputs, test_labels = _sudoku_csv_to_arrays(
            "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/test.csv",
            test_limit,
        )
        np.save(cached_train_inputs, train_inputs)
        np.save(cached_train_labels, train_labels)
        np.save(cached_test_inputs, test_inputs)
        np.save(cached_test_labels, test_labels)

    return (
        np.load(cached_train_inputs, mmap_mode="r"),
        np.load(cached_train_labels, mmap_mode="r"),
        np.load(cached_test_inputs, mmap_mode="r"),
        np.load(cached_test_labels, mmap_mode="r"),
    )


def load_sudoku(data_path: Path, cache_path: Path, *, train_limit: int, test_limit: int) -> ArrayDataset:
    train_inputs, train_labels, test_inputs, test_labels = _load_or_download_sudoku_arrays(
        data_path,
        cache_path,
        train_limit=train_limit,
        test_limit=test_limit,
    )
    return ArrayDataset(
        name="sudoku",
        train_inputs=train_inputs,
        train_labels=train_labels,
        test_inputs=test_inputs,
        test_labels=test_labels,
        ood_inputs=None,
        ood_labels=None,
        vocab_size=11,
        seq_len=81,
        task_residual_type="sudoku_constraints",
        grid_size=9,
    )


def _load_arc_split(data_path: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    split_path = data_path / split
    metadata = json.loads((split_path / "dataset.json").read_text(encoding="utf-8"))
    inputs = np.load(split_path / "all__inputs.npy", mmap_mode="r")
    labels = np.load(split_path / "all__labels.npy", mmap_mode="r")
    puzzle_identifiers = np.load(split_path / "all__puzzle_identifiers.npy", mmap_mode="r")
    puzzle_indices = np.load(split_path / "all__puzzle_indices.npy", mmap_mode="r")
    example_puzzle_indices = np.searchsorted(puzzle_indices, np.arange(inputs.shape[0]), side="right") - 1
    example_identifiers = np.asarray(puzzle_identifiers[example_puzzle_indices], dtype=np.int64)
    return inputs, labels, example_identifiers, int(metadata["num_puzzle_identifiers"])


def load_arc(data_path: Path, *, name: str, hidden_size: int) -> ArrayDataset:
    train_inputs, train_labels, train_ids, train_num_ids = _load_arc_split(data_path, "train")
    test_inputs, test_labels, test_ids, test_num_ids = _load_arc_split(data_path, "test")
    metadata = json.loads((data_path / "train" / "dataset.json").read_text(encoding="utf-8"))
    num_puzzle_identifiers = max(train_num_ids, test_num_ids)
    return ArrayDataset(
        name=name,
        train_inputs=train_inputs,
        train_labels=train_labels,
        test_inputs=test_inputs,
        test_labels=test_labels,
        ood_inputs=None,
        ood_labels=None,
        vocab_size=int(metadata["vocab_size"]),
        seq_len=int(metadata["seq_len"]),
        train_puzzle_identifiers=train_ids,
        test_puzzle_identifiers=test_ids,
        num_puzzle_identifiers=num_puzzle_identifiers,
        puzzle_emb_ndim=hidden_size,
        ignore_label_id=int(metadata.get("ignore_label_id", 0)),
        task_residual_type="none",
    )


def _neighbors(cell: Tuple[int, int], size: int) -> Iterable[Tuple[int, int]]:
    r, c = cell
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            yield nr, nc


def _generate_maze_example(
    rng: np.random.Generator,
    size: int,
    min_path_len: int = 0,
    max_attempts: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    # Perfect maze over odd lattice cells. Tokens: 1 '#', 2 ' ', 3 'S', 4 'G', 5 'o'.
    if size % 2 == 0:
        raise ValueError("maze size must be odd.")

    for _ in range(max_attempts):
        grid = np.ones((size, size), dtype=np.int64)
        start = (1, 1)
        grid[start] = 2
        stack = [start]
        visited = {start}
        cell_coords = [(r, c) for r in range(1, size, 2) for c in range(1, size, 2)]

        while stack:
            current = stack[-1]
            candidates = []
            for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                nr, nc = current[0] + dr, current[1] + dc
                if 1 <= nr < size - 1 and 1 <= nc < size - 1 and (nr, nc) not in visited:
                    candidates.append((nr, nc, dr, dc))
            if not candidates:
                stack.pop()
                continue
            nr, nc, dr, dc = candidates[int(rng.integers(0, len(candidates)))]
            grid[current[0] + dr // 2, current[1] + dc // 2] = 2
            grid[nr, nc] = 2
            visited.add((nr, nc))
            stack.append((nr, nc))

        # Choose the farthest odd cell from start as goal.
        queue = [start]
        parent = {start: None}
        for cell in queue:
            for nb in _neighbors(cell, size):
                if grid[nb] != 1 and nb not in parent:
                    parent[nb] = cell
                    queue.append(nb)
        odd_reachable = [cell for cell in cell_coords if cell in parent]
        goal = max(odd_reachable, key=lambda x: abs(x[0] - start[0]) + abs(x[1] - start[1]))

        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path_len = len(path)
        if path_len < min_path_len:
            continue

        inputs = grid.copy()
        labels = grid.copy()
        inputs[start] = 3
        inputs[goal] = 4
        labels[start] = 3
        labels[goal] = 4
        for cell in path[1:-1]:
            labels[cell] = 5
        return inputs.reshape(-1), labels.reshape(-1), path_len
    raise RuntimeError(f"Could not generate a maze with path length >= {min_path_len} after {max_attempts} attempts.")


def make_maze_dataset(
    *,
    seed: int,
    size: int,
    train_examples: int,
    test_examples: int,
    ood_examples: int,
) -> ArrayDataset:
    rng = np.random.default_rng(seed)
    train_inputs, train_labels = [], []
    test_inputs, test_labels = [], []
    ood_inputs, ood_labels = [], []
    path_lengths = []

    for _ in range(train_examples):
        x, y, path_len = _generate_maze_example(rng, size=size)
        train_inputs.append(x)
        train_labels.append(y)
        path_lengths.append(path_len)
    for _ in range(test_examples):
        x, y, path_len = _generate_maze_example(rng, size=size)
        test_inputs.append(x)
        test_labels.append(y)
        path_lengths.append(path_len)

    # OOD here means same sequence length but longer solution paths than training/test average.
    min_ood_path = int(np.quantile(path_lengths, 0.75)) if path_lengths else 0
    for _ in range(ood_examples):
        try:
            x, y, _ = _generate_maze_example(rng, size=size, min_path_len=min_ood_path)
        except RuntimeError:
            x, y, _ = _generate_maze_example(rng, size=size)
        ood_inputs.append(x)
        ood_labels.append(y)

    return ArrayDataset(
        name=f"maze{size}",
        train_inputs=np.asarray(train_inputs, dtype=np.int64),
        train_labels=np.asarray(train_labels, dtype=np.int64),
        test_inputs=np.asarray(test_inputs, dtype=np.int64),
        test_labels=np.asarray(test_labels, dtype=np.int64),
        ood_inputs=np.asarray(ood_inputs, dtype=np.int64),
        ood_labels=np.asarray(ood_labels, dtype=np.int64),
        vocab_size=6,
        seq_len=size * size,
        task_residual_type="maze_certificate",
        grid_size=size,
    )


def _dijkstra_distances(weights: np.ndarray, source: int) -> np.ndarray:
    n = weights.shape[0]
    inf = 10**9
    distances = np.full(n, inf, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    distances[source] = 0

    for _ in range(n):
        candidates = np.where(~visited, distances, inf + 1)
        node = int(candidates.argmin())
        if candidates[node] >= inf:
            break
        visited[node] = True
        outgoing = weights[node]
        for neighbor in np.nonzero(outgoing > 0)[0]:
            candidate = distances[node] + int(outgoing[neighbor])
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
    return distances


def _generate_shortest_path_example(
    rng: np.random.Generator,
    *,
    nodes: int,
    edge_prob: float,
    max_weight: int,
    dist_cap: int,
    official_like: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    node_token = 1
    source_token = 2
    no_edge_token = 3
    self_token = 4
    weight_base = 5
    dist_base = weight_base + max_weight
    inf_distance = 10**9
    inf_token = dist_base + dist_cap + 1

    source = int(rng.integers(0, nodes))
    weights = np.zeros((nodes, nodes), dtype=np.int64)

    if official_like:
        # Mirrors the official CLRS Dijkstra sampler more closely: undirected
        # Erdos-Renyi graph, weighted edges, random source, no reachability fix.
        mat = rng.binomial(1, edge_prob, size=(nodes, nodes))
        mat *= mat.T
        np.fill_diagonal(mat, 0)
        raw = rng.uniform(0.0, 1.0, size=(nodes, nodes))
        raw *= raw.T
        continuous = np.sqrt(raw + 1e-3)
        edge_i, edge_j = np.nonzero(mat > 0)
        if edge_i.size:
            quantized = np.clip(np.ceil(continuous[edge_i, edge_j] * max_weight), 1, max_weight).astype(np.int64)
            weights[edge_i, edge_j] = quantized
    else:
        # Guarantee source reachability with a random directed spanning path,
        # then add random shortcut edges. This keeps the earlier benchmark from
        # degenerating into mostly INF detection.
        order = [source] + [int(x) for x in rng.permutation([i for i in range(nodes) if i != source])]
        for left, right in zip(order[:-1], order[1:]):
            weights[left, right] = int(rng.integers(1, max_weight + 1))

        random_edges = rng.random((nodes, nodes)) < edge_prob
        for i in range(nodes):
            random_edges[i, i] = False
        add_i, add_j = np.nonzero(random_edges & (weights == 0))
        if add_i.size:
            weights[add_i, add_j] = rng.integers(1, max_weight + 1, size=add_i.size, dtype=np.int64)

    distances = _dijkstra_distances(weights, source)
    seq_len = nodes + nodes * nodes
    inputs = np.empty(seq_len, dtype=np.int64)
    labels = np.full(seq_len, -100, dtype=np.int64)

    inputs[:nodes] = node_token
    inputs[source] = source_token
    matrix_tokens = np.full((nodes, nodes), no_edge_token, dtype=np.int64)
    for i in range(nodes):
        matrix_tokens[i, i] = self_token
    edge_i, edge_j = np.nonzero(weights > 0)
    matrix_tokens[edge_i, edge_j] = weight_base + weights[edge_i, edge_j] - 1
    inputs[nodes:] = matrix_tokens.reshape(-1)

    for node, distance in enumerate(distances):
        if distance >= inf_distance:
            labels[node] = inf_token
        else:
            labels[node] = dist_base + min(int(distance), dist_cap)
    return inputs, labels


def make_clrs_shortest_path_dataset(
    *,
    seed: int,
    nodes: int,
    edge_prob: float,
    ood_edge_prob: float,
    max_weight: int,
    dist_cap: int,
    train_examples: int,
    test_examples: int,
    ood_examples: int,
    official_like: bool = False,
) -> ArrayDataset:
    rng = np.random.default_rng(seed)
    train_inputs, train_labels = [], []
    test_inputs, test_labels = [], []
    ood_inputs, ood_labels = [], []

    for _ in range(train_examples):
        x, y = _generate_shortest_path_example(
            rng,
            nodes=nodes,
            edge_prob=edge_prob,
            max_weight=max_weight,
            dist_cap=dist_cap,
            official_like=official_like,
        )
        train_inputs.append(x)
        train_labels.append(y)
    for _ in range(test_examples):
        x, y = _generate_shortest_path_example(
            rng,
            nodes=nodes,
            edge_prob=edge_prob,
            max_weight=max_weight,
            dist_cap=dist_cap,
            official_like=official_like,
        )
        test_inputs.append(x)
        test_labels.append(y)
    for _ in range(ood_examples):
        x, y = _generate_shortest_path_example(
            rng,
            nodes=nodes,
            edge_prob=ood_edge_prob,
            max_weight=max_weight,
            dist_cap=dist_cap,
            official_like=official_like,
        )
        ood_inputs.append(x)
        ood_labels.append(y)

    return ArrayDataset(
        name=f"clrs_sp{nodes}{'_official_like' if official_like else ''}",
        train_inputs=np.asarray(train_inputs, dtype=np.int64),
        train_labels=np.asarray(train_labels, dtype=np.int64),
        test_inputs=np.asarray(test_inputs, dtype=np.int64),
        test_labels=np.asarray(test_labels, dtype=np.int64),
        ood_inputs=np.asarray(ood_inputs, dtype=np.int64),
        ood_labels=np.asarray(ood_labels, dtype=np.int64),
        vocab_size=5 + max_weight + dist_cap + 2,
        seq_len=nodes + nodes * nodes,
        task_residual_type="clrs_bellman",
        clrs_nodes=nodes,
        clrs_max_weight=max_weight,
        clrs_dist_cap=dist_cap,
    )


def make_batch(
    inputs: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    *,
    puzzle_identifiers: Optional[np.ndarray] = None,
    ignore_label_id: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    labels_np = np.asarray(labels[indices], dtype=np.int64)
    if ignore_label_id is not None:
        labels_np = labels_np.copy()
        labels_np[labels_np == ignore_label_id] = IGNORE_LABEL_ID
    if puzzle_identifiers is None:
        puzzle_ids = np.zeros(indices.shape[0], dtype=np.int64)
    else:
        puzzle_ids = np.asarray(puzzle_identifiers[indices], dtype=np.int64)
    return {
        "inputs": torch.from_numpy(np.asarray(inputs[indices], dtype=np.int32)).to(device),
        "labels": torch.from_numpy(labels_np).to(device),
        "puzzle_identifiers": torch.from_numpy(puzzle_ids.astype(np.int32)).to(device),
    }


def base_config(args: argparse.Namespace, dataset: ArrayDataset) -> Dict[str, object]:
    return {
        "batch_size": args.batch_size,
        "seq_len": dataset.seq_len,
        "puzzle_emb_ndim": dataset.puzzle_emb_ndim,
        "num_puzzle_identifiers": dataset.num_puzzle_identifiers,
        "vocab_size": dataset.vocab_size,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "expansion": args.expansion,
        "num_heads": args.num_heads,
        "pos_encodings": "rope",
        "loops": args.loops,
        "H_cycles": 1,
        "L_cycles": args.l_cycles,
        "forward_dtype": "float32",
    }


def variants(args: argparse.Namespace, dataset: ArrayDataset) -> List[Variant]:
    common = base_config(args, dataset)
    tarm = {
        **common,
        "tropical_attention_version": "v3",
        "tropical_proj": True,
        "tropical_qkv_proj": False,
        "tropical_norm": "none",
    }

    def tapr_base() -> Dict[str, object]:
        return {
            **tarm,
            "ponder_min_steps": args.ponder_min_steps,
            "ponder_eval_threshold": args.ponder_eval_threshold,
            "ponder_prior_lambda": args.ponder_prior_lambda,
            "ponder_step_cost": args.ponder_step_cost,
            "ponder_kl_weight": args.ponder_kl_weight,
            "halt_correctness_weight": args.halt_correctness_weight,
            "detach_recurrent_state": True,
            "detach_ponder_state": True,
            "eval_halt_mode": "full",
            "nextlat_loss_type": "projective",
            "nextlat_supervised_only": True,
            "nextlat_normalize_to_ce": True,
            "nextlat_weight": args.nextlat_weight,
            "tapr_distance_mode": args.tapr_distance_mode,
            "fw_bias": args.fw_bias,
            "aux_normalize_max_scale": args.aux_normalize_max_scale,
            "chart_transition": True,
            "chart_count": args.chart_count,
            "chart_transition_strength": args.chart_transition_strength,
            "task_residual_type": "none",
            "task_residual_weight": 0.0,
            "task_residual_grid_size": dataset.grid_size,
            "task_residual_clrs_nodes": dataset.clrs_nodes,
            "task_residual_clrs_max_weight": dataset.clrs_max_weight,
            "task_residual_clrs_dist_cap": dataset.clrs_dist_cap,
        }

    tapr_ce = {
        **tapr_base(),
        "ponder_enabled": False,
        "disable_halting": True,
        "ponder_step_cost": 0.0,
        "ponder_kl_weight": 0.0,
        "halt_correctness_weight": 0.0,
        "nextlat_enabled": False,
        "nextlat_weight": 0.0,
        "task_residual_weight": 0.0,
    }
    tapr_ponder = {
        **tapr_base(),
        "ponder_enabled": True,
        "disable_halting": False,
        "detach_recurrent_state": False,
        "detach_ponder_state": False,
        "nextlat_enabled": False,
        "nextlat_weight": 0.0,
        "task_residual_weight": 0.0,
    }
    tapr_nextlat = {
        **tapr_base(),
        "ponder_enabled": True,
        "disable_halting": False,
        "detach_recurrent_state": False,
        "detach_ponder_state": False,
        "nextlat_enabled": True,
        "task_residual_weight": 0.0,
    }
    tapr_full = {
        **tapr_nextlat,
        "tapr_architecture": "clean_full",
        "H_cycles": 1,
        "L_cycles": 1,
        "halt_head_type": "scalar",
        "bptt_window": args.tapr_full_bptt_window,
        "readout_every": args.tapr_full_readout_every,
        "ponder_ce_weight": args.tapr_full_ponder_ce_weight,
        "final_ce_weight": args.tapr_full_final_ce_weight,
        "halt_correctness_weight": 0.0,
        "task_residual_type": dataset.task_residual_type,
        "task_residual_weight": args.task_residual_weight,
        "task_residual_normalize_to_ce": True,
    }
    pure_tropical_transformer = {
        **tarm,
        "tapr_architecture": "pure_tropical_transformer",
        "num_layers": args.pure_tropical_layers,
        "loops": 1,
        "H_cycles": 1,
        "L_cycles": 1,
        "ponder_enabled": False,
        "disable_halting": True,
        "halt_head_type": "scalar",
        "ponder_step_cost": 0.0,
        "ponder_kl_weight": 0.0,
        "halt_correctness_weight": 0.0,
        "detach_recurrent_state": True,
        "detach_ponder_state": True,
        "bptt_window": 1,
        "nextlat_enabled": False,
        "nextlat_weight": 0.0,
        "chart_transition": False,
        "task_residual_weight": 0.0,
    }
    selected = [
        Variant("URM", URM, dict(common)),
        Variant("TARM-v3", TARM, dict(tarm)),
        Variant("TAPR-PureTARM-v3", TAPR, dict(pure_tropical_transformer), tapr_loss=False),
        Variant("TAPR-CE", TAPR, dict(tapr_ce), tapr_loss=False),
        Variant("TAPR-Ponder", TAPR, dict(tapr_ponder), tapr_loss=True),
        Variant("TAPR-NextLat", TAPR, dict(tapr_nextlat), tapr_loss=True),
        Variant("TAPR-Full", TAPR, dict(tapr_full), tapr_loss=True),
    ]
    if args.variants:
        wanted = set(args.variants)
        selected = [variant for variant in selected if variant.name in wanted]
        missing = wanted.difference({variant.name for variant in selected})
        if missing:
            raise ValueError(f"Unknown variant name(s): {sorted(missing)}")
    return selected


def run_model_to_halt(model, batch: Dict[str, torch.Tensor], *, loss_head: Optional[TAPRLossHead] = None):
    carry = model.initial_carry(batch) if loss_head is None else loss_head.initial_carry(batch)
    outputs = None
    losses = []
    step_outputs = []
    step_metrics = []
    while True:
        if loss_head is None:
            carry, outputs = model(carry, batch)
            done = bool(carry.halted.all())
        else:
            carry, loss, metrics, returned, done_tensor = loss_head(
                carry=carry,
                batch=batch,
                return_keys={"logits", "halt_prob", "ponder_prob", "nextlat_loss", "state_delta", "chart_margin"},
            )
            outputs = returned if returned is not None else {}
            for key in (
                "lm_loss",
                "ponder_ce_loss",
                "final_ce_loss",
                "nextlat_loss",
                "nextlat_scale",
                "task_residual_loss",
                "task_residual_scale",
            ):
                if key in metrics:
                    outputs[key] = metrics[key].detach()
            losses.append(loss)
            step_metrics.append(metrics)
            done = bool(done_tensor)
        if outputs is not None:
            step_outputs.append(outputs)
        if done:
            break
    return carry, outputs, step_outputs, losses, step_metrics


def train_variant(
    variant: Variant,
    dataset: ArrayDataset,
    *,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    torch.manual_seed(seed)
    model = variant.model_cls(variant.config).to(device)
    loss_head = TAPRLossHead(model, loss_type=variant.loss_type) if variant.tapr_loss else None
    trainable = loss_head if loss_head is not None else model
    trainable.train()
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed + 17)
    train_size = int(dataset.train_inputs.shape[0])
    losses: List[float] = []
    ce_losses: List[float] = []
    nextlat_losses: List[float] = []
    task_residual_losses: List[float] = []
    ponder_ce_losses: List[float] = []
    final_ce_losses: List[float] = []
    nextlat_scales: List[float] = []
    task_residual_scales: List[float] = []
    start_time = time.perf_counter()

    for step in range(1, args.train_steps + 1):
        indices = rng.integers(0, train_size, size=args.batch_size, dtype=np.int64)
        batch = make_batch(
            dataset.train_inputs,
            dataset.train_labels,
            indices,
            device,
            puzzle_identifiers=dataset.train_puzzle_identifiers,
            ignore_label_id=dataset.ignore_label_id,
        )
        optimizer.zero_grad(set_to_none=True)

        if loss_head is None:
            _, outputs, _, _, _ = run_model_to_halt(model, batch)
            logits = outputs["logits"].to(torch.float32)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1))
            ce_loss = loss.detach()
            nextlat_loss = torch.tensor(0.0)
            task_residual_loss = torch.tensor(0.0)
            ponder_ce_loss = torch.tensor(0.0)
            final_ce_loss = torch.tensor(0.0)
            nextlat_scale = torch.tensor(1.0)
            task_residual_scale = torch.tensor(1.0)
        else:
            _, outputs, _, step_losses, _ = run_model_to_halt(model, batch, loss_head=loss_head)
            loss = torch.stack(step_losses).sum() / args.batch_size
            ce_loss = F.cross_entropy(
                outputs["logits"].to(torch.float32).reshape(-1, outputs["logits"].shape[-1]),
                batch["labels"].reshape(-1),
            ).detach()
            nextlat_loss = outputs.get("nextlat_loss", torch.tensor(0.0, device=device)).to(torch.float32).mean().detach()
            task_residual_loss = outputs.get("task_residual_loss", torch.tensor(0.0, device=device)).to(torch.float32).mean().detach()
            ponder_ce_loss = outputs.get("ponder_ce_loss", torch.tensor(0.0, device=device)).to(torch.float32).mean().detach()
            final_ce_loss = outputs.get("final_ce_loss", torch.tensor(0.0, device=device)).to(torch.float32).mean().detach()
            nextlat_scale = outputs.get("nextlat_scale", torch.tensor(1.0, device=device)).to(torch.float32).mean().detach()
            task_residual_scale = outputs.get("task_residual_scale", torch.tensor(1.0, device=device)).to(torch.float32).mean().detach()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        ce_losses.append(float(ce_loss.cpu().item()))
        nextlat_losses.append(float(nextlat_loss.cpu().item()))
        task_residual_losses.append(float(task_residual_loss.cpu().item()))
        ponder_ce_losses.append(float(ponder_ce_loss.cpu().item()))
        final_ce_losses.append(float(final_ce_loss.cpu().item()))
        nextlat_scales.append(float(nextlat_scale.cpu().item()))
        task_residual_scales.append(float(task_residual_scale.cpu().item()))
        if args.log_interval and (step == 1 or step % args.log_interval == 0 or step == args.train_steps):
            print(json.dumps({
                "dataset": dataset.name,
                "variant": variant.name,
                "step": step,
                "loss": losses[-1],
                "ce_loss": ce_losses[-1],
                "ponder_ce_loss": ponder_ce_losses[-1],
                "final_ce_loss": final_ce_losses[-1],
                "nextlat_loss": nextlat_losses[-1],
                "task_residual_loss": task_residual_losses[-1],
                "nextlat_scale": nextlat_scales[-1],
                "task_residual_scale": task_residual_scales[-1],
            }), flush=True)

    return model, {
        "train_seconds": time.perf_counter() - start_time,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "train_ce_first": ce_losses[0],
        "train_ce_last": ce_losses[-1],
        "train_ponder_ce_first": ponder_ce_losses[0],
        "train_ponder_ce_last": ponder_ce_losses[-1],
        "train_final_ce_first": final_ce_losses[0],
        "train_final_ce_last": final_ce_losses[-1],
        "train_nextlat_first": nextlat_losses[0],
        "train_nextlat_last": nextlat_losses[-1],
        "train_nextlat_scale_first": nextlat_scales[0],
        "train_nextlat_scale_last": nextlat_scales[-1],
        "train_task_residual_first": task_residual_losses[0],
        "train_task_residual_last": task_residual_losses[-1],
        "train_task_residual_scale_first": task_residual_scales[0],
        "train_task_residual_scale_last": task_residual_scales[-1],
    }


def _mean_or_nan(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def evaluate_variant(
    model,
    variant: Variant,
    inputs: np.ndarray,
    labels_np: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
    prefix: str,
    puzzle_identifiers: Optional[np.ndarray] = None,
    ignore_label_id: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total_examples = min(args.eval_examples, int(inputs.shape[0]))
    total_cells = 0
    correct_cells = 0
    exact = 0
    step_sum = 0.0
    acc_per_step_sum = 0.0
    over_halt = 0
    under_halt = 0
    halt_brier_sum = 0.0
    ponder_expected_sum = 0.0
    ponder_mass_sum = 0.0
    ponder_final_halt_prob_sum = 0.0
    ponder_batches = 0
    nextlat_by_step: Dict[int, List[float]] = {}
    state_delta_by_step: Dict[int, List[float]] = {}
    chart_margins: List[float] = []
    started = time.perf_counter()
    num_batches = 0

    with torch.inference_mode():
        for start in range(0, total_examples, args.eval_batch_size):
            stop = min(total_examples, start + args.eval_batch_size)
            batch = make_batch(
                inputs,
                labels_np,
                np.arange(start, stop, dtype=np.int64),
                device,
                puzzle_identifiers=puzzle_identifiers,
                ignore_label_id=ignore_label_id,
            )
            carry = model.initial_carry(batch)
            outputs = None
            step_idx = 0
            while True:
                carry, outputs = model(carry, batch)
                step_idx += 1
                if "nextlat_loss" in outputs:
                    nextlat_by_step.setdefault(step_idx, []).extend(float(v) for v in outputs["nextlat_loss"].cpu().reshape(-1))
                if "state_delta" in outputs:
                    state_delta_by_step.setdefault(step_idx, []).extend(float(v) for v in outputs["state_delta"].cpu().reshape(-1))
                if "chart_margin" in outputs:
                    chart_margins.extend(float(v) for v in outputs["chart_margin"].cpu().reshape(-1))
                if bool(carry.halted.all()):
                    break

            preds = outputs["logits"].argmax(dim=-1)
            labels = batch["labels"]
            valid_mask = labels.ne(-100)
            cell_correct = preds.eq(labels) & valid_mask
            per_example_cells = valid_mask.sum(dim=-1).clamp_min(1)
            per_example_acc = cell_correct.sum(dim=-1).to(torch.float32) / per_example_cells
            seq_correct = (cell_correct | ~valid_mask).all(dim=-1)
            steps = carry.steps.to(torch.float32)
            if "halt_prob" in outputs:
                halt_prob = outputs["halt_prob"].to(torch.float32)
                halt_pred = halt_prob >= args.ponder_eval_threshold
                ponder_final_halt_prob_sum += float(halt_prob.sum().item())
                ponder_batches += 1
            else:
                halt_prob = torch.sigmoid(outputs["q_halt_logits"].to(torch.float32))
                halt_pred = outputs["q_halt_logits"].to(torch.float32) >= 0
            if "ponder_expected_steps" in outputs:
                ponder_expected_sum += float(outputs["ponder_expected_steps"].to(torch.float32).sum().item())
            if "ponder_mass" in outputs:
                ponder_mass_sum += float(outputs["ponder_mass"].to(torch.float32).sum().item())

            total_cells += int(valid_mask.sum().item())
            correct_cells += int(cell_correct.sum().item())
            exact += int(seq_correct.sum().item())
            step_sum += float(steps.sum().item())
            acc_per_step_sum += float((per_example_acc / steps.clamp_min(1)).sum().item())
            over_halt += int((halt_pred & ~seq_correct).sum().item())
            under_halt += int((~halt_pred & seq_correct).sum().item())
            halt_brier_sum += float((halt_prob - seq_correct.to(torch.float32)).pow(2).sum().item())
            num_batches += 1

    elapsed = max(time.perf_counter() - started, 1e-9)
    row = {
        f"{prefix}_examples": float(total_examples),
        f"{prefix}_cell_accuracy": correct_cells / max(total_cells, 1),
        f"{prefix}_exact_accuracy": exact / max(total_examples, 1),
        f"{prefix}_mean_steps": step_sum / max(total_examples, 1),
        f"{prefix}_accuracy_per_step": acc_per_step_sum / max(total_examples, 1),
        f"{prefix}_over_halt_rate": over_halt / max(total_examples, 1),
        f"{prefix}_under_halt_rate": under_halt / max(total_examples, 1),
        f"{prefix}_halt_brier": halt_brier_sum / max(total_examples, 1),
        f"{prefix}_ponder_expected_steps": ponder_expected_sum / max(total_examples, 1),
        f"{prefix}_ponder_mass": ponder_mass_sum / max(total_examples, 1),
        f"{prefix}_final_halt_prob": ponder_final_halt_prob_sum / max(total_examples, 1) if ponder_batches else float("nan"),
        f"{prefix}_examples_per_second": total_examples / elapsed,
        f"{prefix}_batches_per_second": num_batches / elapsed,
        f"{prefix}_chart_margin": _mean_or_nan(chart_margins),
    }
    for step, values in sorted(nextlat_by_step.items()):
        row[f"{prefix}_nextlat_step_{step}"] = _mean_or_nan(values)
    for step, values in sorted(state_delta_by_step.items()):
        row[f"{prefix}_state_delta_step_{step}"] = _mean_or_nan(values)
    return row


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Medium CPU comparison for URM, TARM-v3, and TAPR.")
    parser.add_argument("--sudoku-data-path", type=Path, default=Path("data/sudoku-extreme-cpu-ablation"))
    parser.add_argument("--sudoku-cache-path", type=Path, default=Path("eval_results/cache/sudoku-extreme-medium"))
    parser.add_argument("--sudoku-train-limit", type=int, default=128)
    parser.add_argument("--arc1-data-path", type=Path, default=Path("data/arc1concept-aug-16-cpu"))
    parser.add_argument("--arc2-data-path", type=Path, default=Path("data/arc2concept-aug-16-cpu"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results/medium_urm_tarm_tapr_cpu"))
    parser.add_argument("--tasks", nargs="+", default=["sudoku", "maze"])
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-examples", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--l-cycles", type=int, default=1)
    parser.add_argument("--expansion", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--maze-size", type=int, default=11)
    parser.add_argument("--maze-train-examples", type=int, default=512)
    parser.add_argument("--maze-test-examples", type=int, default=1024)
    parser.add_argument("--maze-ood-examples", type=int, default=512)
    parser.add_argument("--clrs-nodes", type=int, default=12)
    parser.add_argument("--clrs-edge-prob", type=float, default=0.35)
    parser.add_argument("--clrs-ood-edge-prob", type=float, default=0.18)
    parser.add_argument("--clrs-official-like", action="store_true")
    parser.add_argument("--clrs-max-weight", type=int, default=9)
    parser.add_argument("--clrs-dist-cap", type=int, default=32)
    parser.add_argument("--clrs-train-examples", type=int, default=512)
    parser.add_argument("--clrs-test-examples", type=int, default=1024)
    parser.add_argument("--clrs-ood-examples", type=int, default=512)
    parser.add_argument("--ponder-min-steps", type=int, default=1)
    parser.add_argument("--ponder-eval-threshold", type=float, default=0.5)
    parser.add_argument("--ponder-prior-lambda", type=float, default=0.25)
    parser.add_argument("--ponder-step-cost", type=float, default=0.001)
    parser.add_argument("--ponder-kl-weight", type=float, default=0.01)
    parser.add_argument("--halt-correctness-weight", type=float, default=0.0)
    parser.add_argument("--tapr-full-bptt-window", type=int, default=0)
    parser.add_argument("--tapr-full-readout-every", type=int, default=1)
    parser.add_argument("--tapr-full-ponder-ce-weight", type=float, default=0.5)
    parser.add_argument("--tapr-full-final-ce-weight", type=float, default=1.0)
    parser.add_argument("--nextlat-weight", type=float, default=0.05)
    parser.add_argument("--task-residual-weight", type=float, default=0.05)
    parser.add_argument("--aux-normalize-max-scale", type=float, default=10.0)
    parser.add_argument("--tapr-distance-mode", type=str, default="hilbert")
    parser.add_argument("--fw-bias", type=float, default=0.5)
    parser.add_argument("--chart-count", type=int, default=2)
    parser.add_argument("--chart-transition-strength", type=float, default=0.15)
    parser.add_argument("--pure-tropical-layers", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    datasets: List[ArrayDataset] = []
    if "sudoku" in args.tasks:
        datasets.append(
            load_sudoku(
                args.sudoku_data_path,
                args.sudoku_cache_path,
                train_limit=args.sudoku_train_limit,
                test_limit=max(args.eval_examples, args.eval_batch_size),
            )
        )
    if "arc1" in args.tasks or "arc-agi-1" in args.tasks:
        datasets.append(load_arc(args.arc1_data_path, name="arcagi1_public", hidden_size=args.hidden_size))
    if "arc2" in args.tasks or "arc-agi-2" in args.tasks:
        datasets.append(load_arc(args.arc2_data_path, name="arcagi2_public", hidden_size=args.hidden_size))
    if "maze" in args.tasks:
        datasets.append(
            make_maze_dataset(
                seed=args.seed + 101,
                size=args.maze_size,
                train_examples=args.maze_train_examples,
                test_examples=args.maze_test_examples,
                ood_examples=args.maze_ood_examples,
            )
        )
    if any(task in args.tasks for task in ("clrs", "clrs_sp", "shortest_path", "clrs_official_like")):
        datasets.append(
            make_clrs_shortest_path_dataset(
                seed=args.seed + 202,
                nodes=args.clrs_nodes,
                edge_prob=args.clrs_edge_prob,
                ood_edge_prob=args.clrs_ood_edge_prob,
                max_weight=args.clrs_max_weight,
                dist_cap=args.clrs_dist_cap,
                train_examples=args.clrs_train_examples,
                test_examples=args.clrs_test_examples,
                ood_examples=args.clrs_ood_examples,
                official_like=args.clrs_official_like or "clrs_official_like" in args.tasks,
            )
        )

    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        print(f"DATASET {dataset.name} train={dataset.train_inputs.shape} test={dataset.test_inputs.shape}", flush=True)
        dataset_variants = variants(args, dataset)
        for idx, variant in enumerate(dataset_variants, start=1):
            print(f"[{dataset.name} {idx}/{len(dataset_variants)}] {variant.name}", flush=True)
            model, train_metrics = train_variant(
                variant,
                dataset,
                args=args,
                device=device,
                seed=args.seed + idx,
            )
            param_count = sum(p.numel() for p in model.parameters())
            row: Dict[str, object] = {
                "dataset": dataset.name,
                "variant": variant.name,
                "params": param_count,
                "train_steps": args.train_steps,
                "seq_len": dataset.seq_len,
                "vocab_size": dataset.vocab_size,
                "hidden_size": int(variant.config.get("hidden_size", args.hidden_size)),
                "num_layers": int(variant.config.get("num_layers", args.num_layers)),
                "loops": int(variant.config.get("loops", args.loops)),
                "l_cycles": int(variant.config.get("L_cycles", args.l_cycles)),
                "h_cycles": int(variant.config.get("H_cycles", 1)),
                "tapr_architecture": str(variant.config.get("tapr_architecture", "legacy")),
                "bptt_window": int(variant.config.get("bptt_window", -1)),
                "readout_every": int(variant.config.get("readout_every", 1)),
                "halt_head_type": str(variant.config.get("halt_head_type", "q_pair")),
                **train_metrics,
            }
            row.update(
                evaluate_variant(
                    model,
                    variant,
                    dataset.test_inputs,
                    dataset.test_labels,
                    args=args,
                    device=device,
                    prefix="test",
                    puzzle_identifiers=dataset.test_puzzle_identifiers,
                    ignore_label_id=dataset.ignore_label_id,
                )
            )
            if dataset.ood_inputs is not None and dataset.ood_labels is not None:
                row.update(
                    evaluate_variant(
                        model,
                        variant,
                        dataset.ood_inputs,
                        dataset.ood_labels,
                        args=args,
                        device=device,
                        prefix="ood",
                        puzzle_identifiers=dataset.ood_puzzle_identifiers,
                        ignore_label_id=dataset.ignore_label_id,
                    )
                )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = args.output_dir / f"medium_cpu_{timestamp}.csv"
    json_path = args.output_dir / f"medium_cpu_{timestamp}.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()

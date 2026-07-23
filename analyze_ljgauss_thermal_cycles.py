#!/usr/bin/env python3
"""Batch structural, topological, and dynamical analysis of thermal cycles.

Expected layout (created by run_ljgauss_2d_thermal_cycles.py):
  ROOT/rho_0.40500/seed_1/{heating,cooling}/T_0.04000/trajectory.gsd

The long-time 2D diffusion coefficient is recomputed from each GSD trajectory
using an all-time-origin MSD.  Wrapped coordinates are unwrapped from stored
particle image counters when available; legacy trajectories are reconstructed
from minimum-image frame-to-frame increments.  D_VACF is read from
cycle_summary.dat.  Results are plain-text .dat files.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.setrecursionlimit(100000)

FRAME_HEADER = [
    "frame", "time", "n_bonds", "mean_degree", "endpoint_fraction",
    "branch_fraction", "isolated_fraction", "n_clusters", "largest_fraction",
    "finite_number_mean_size", "finite_weight_mean_size", "beta1_cycles",
    "euler_characteristic", "percolates", "winds_x", "winds_y",
    "global_nematic_S2", "local_bond_order_psi2", "mean_turn_angle",
    "rms_curvature", "finite_mean_Rg", "finite_mean_asphericity",
    "bond_survival", "bond_jaccard", "broken_bonds_per_particle",
    "formed_bonds_per_particle",
    "beta1_per_particle", "beta1_per_cluster", "n_cyclic_components",
    "cyclic_component_fraction", "n_winding_components",
    "beta1_winding_components", "beta1_nonwinding_components",
    "n_triangles", "triangles_per_particle", "graph_transitivity",
    "degree2_fraction", "degree3_fraction", "degree4_fraction",
    "degree_ge5_fraction", "two_core_fraction", "n_two_core_components",
    "largest_two_core_fraction", "n_linear_chains",
    "mean_chain_contour_length", "mean_chain_end_to_end",
    "mean_chain_tortuosity", "chain_persistence_length",
    "broken_bridge_edges_per_particle", "broken_cycle_edges_per_particle",
    "formed_merge_edges_per_particle", "formed_cycle_edges_per_particle",
    "step_msd_degree1", "step_msd_degree2", "step_msd_degree_ge3",
    "break_incidence_degree1", "break_incidence_degree2",
    "break_incidence_degree_ge3",
    "mean_fundamental_cycle_length", "max_fundamental_cycle_length",
    "triangle_basis_cycle_fraction",
]


class WeightedDSU:
    """Union-find with integer periodic-image offsets and winding detection."""

    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)
        # offset[x]: image coordinate of x minus image coordinate of parent[x]
        self.offset = np.zeros((n, 2), dtype=np.int64)
        self.wind = np.zeros((n, 2), dtype=bool)

    def find(self, x):
        p = int(self.parent[x])
        if p == x:
            return x, self.offset[x].copy()
        r, up = self.find(p)
        self.offset[x] += up
        self.parent[x] = r
        return r, self.offset[x].copy()

    def union(self, i, j, shift):
        """Impose image(j)-image(i)=shift for an i--j minimum-image bond."""
        ri, oi = self.find(i)
        rj, oj = self.find(j)
        shift = np.asarray(shift, dtype=np.int64)
        if ri == rj:
            loop = shift - (oj - oi)
            self.wind[ri] |= loop != 0
            return
        # image(root_j)-image(root_i) = shift + oi - oj
        d = shift + oi - oj
        if self.size[ri] < self.size[rj]:
            self.parent[ri] = rj
            self.offset[ri] = -d
            self.size[rj] += self.size[ri]
            self.wind[rj] |= self.wind[ri]
        else:
            self.parent[rj] = ri
            self.offset[rj] = d
            self.size[ri] += self.size[rj]
            self.wind[ri] |= self.wind[rj]


def read_numeric(path):
    a = np.loadtxt(path, ndmin=2)
    return np.asarray(a, dtype=float)


def parse_value(name, prefix):
    if not name.startswith(prefix):
        raise ValueError(name)
    return float(name[len(prefix):])


def trajectory_frames(path, stride=1):
    try:
        import gsd.hoomd
    except ImportError as exc:
        raise SystemExit(
            "The Python package 'gsd' is required. Run this with the same "
            "Python/conda environment used by pypresso, or install gsd."
        ) from exc
    with gsd.hoomd.open(str(path), mode="r") as trj:
        for iframe in range(0, len(trj), stride):
            fr = trj[iframe]
            box = np.asarray(fr.configuration.box[:2], dtype=float)
            pos = np.asarray(fr.particles.position[:, :2], dtype=float)
            pos = (pos + 0.5 * box) % box
            step = float(fr.configuration.step)
            yield iframe, step, pos, box


def load_unwrapped_trajectory(path):
    """Return GSD steps and continuous 2D particle coordinates.

    New trajectories contain ESPResSo image counters.  Older trajectories do
    not; for those files, continuity is reconstructed from the minimum-image
    displacement between consecutive saved frames.  The latter is exact when
    no particle travels more than half a box length between saved frames.
    """
    try:
        import gsd.hoomd
    except ImportError as exc:
        raise SystemExit(
            "The Python package 'gsd' is required. Install it in the analysis "
            "environment with `python -m pip install gsd`."
        ) from exc
    with gsd.hoomd.open(str(path), mode="r") as trj:
        if len(trj) < 8:
            raise ValueError(f"Too few frames for an MSD fit in {path}: {len(trj)}")
        steps = np.asarray(
            [float(trj[i].configuration.step) for i in range(len(trj))])
        box = np.asarray(trj[0].configuration.box[:2], dtype=float)
        wrapped = np.asarray(
            [trj[i].particles.position[:, :2] for i in range(len(trj))],
            dtype=np.float64)
        image_frames = []
        images_available = True
        for i in range(len(trj)):
            image = getattr(trj[i].particles, "image", None)
            if image is None:
                images_available = False
                image_frames.append(np.zeros_like(wrapped[i], dtype=np.int64))
            else:
                image_frames.append(np.asarray(image[:, :2], dtype=np.int64))
        images = np.asarray(image_frames, dtype=np.int64)

    raw_increments = np.diff(wrapped, axis=0)
    has_wrapped_jump = np.any(
        np.abs(raw_increments) > 0.5 * box[None, None, :])
    # Some GSD readers expose a zero-filled default even when an old file did
    # not store the image chunk.  A box-sized jump with exclusively zero image
    # counters therefore triggers the legacy minimum-image fallback.
    use_stored_images = (
        images_available and (np.any(images != 0) or not has_wrapped_jump))
    if use_stored_images:
        unwrapped = wrapped + images * box[None, None, :]
        mode = "stored_images"
    else:
        increments = raw_increments
        increments -= np.rint(
            increments / box[None, None, :]) * box[None, None, :]
        unwrapped = np.concatenate(
            [wrapped[:1],
             wrapped[:1] + np.cumsum(increments, axis=0)], axis=0)
        mode = "minimum_image"
    return steps, unwrapped, box, mode


def all_origin_msd_fft(positions):
    """Exact all-time-origin MSD for every lag using FFT correlations."""
    nframes, nparticles, _ = positions.shape
    nfft = 2 * nframes
    transform = np.fft.rfft(positions, n=nfft, axis=0)
    correlation = np.fft.irfft(
        transform * transform.conjugate(), n=nfft, axis=0)[:nframes]
    correlation = np.sum(correlation, axis=(1, 2))
    squared = np.sum(positions * positions, axis=(1, 2))
    prefix = np.concatenate(([0.0], np.cumsum(squared)))
    counts = np.arange(nframes, 0, -1, dtype=np.int64)
    msd = np.empty(nframes, dtype=float)
    for lag in range(nframes):
        left = prefix[nframes - lag]
        right = prefix[nframes] - prefix[lag]
        msd[lag] = (
            left + right - 2.0 * correlation[lag]
        ) / (counts[lag] * nparticles)
    msd[0] = 0.0
    msd[(msd < 0) & (np.abs(msd) < 1.0e-10)] = 0.0
    return msd, counts


def fit_diffusion(times, msd, max_lag_fraction, fit_fraction):
    """Fit MSD=4Dt+b over the final part of the retained lag interval."""
    if not (0 < max_lag_fraction <= 1):
        raise ValueError("--msd-max-lag-fraction must be in (0, 1]")
    if not (0 < fit_fraction <= 1):
        raise ValueError("--msd-fit-fraction must be in (0, 1]")
    max_index = min(
        len(times) - 1,
        max(5, int(math.floor(max_lag_fraction * (len(times) - 1)))))
    valid = np.flatnonzero(
        (np.arange(len(times)) > 0) &
        (np.arange(len(times)) <= max_index) &
        np.isfinite(times) & np.isfinite(msd))
    if len(valid) < 5:
        return (np.nan,) * 7
    first = int(math.floor((1.0 - fit_fraction) * len(valid)))
    selected = valid[first:]
    slope, intercept = np.polyfit(times[selected], msd[selected], 1)
    predicted = slope * times[selected] + intercept
    residual = np.sum((msd[selected] - predicted) ** 2)
    total = np.sum((msd[selected] - np.mean(msd[selected])) ** 2)
    r2 = 1.0 - residual / total if total > 0 else np.nan
    positive = selected[(times[selected] > 0) & (msd[selected] > 0)]
    alpha = np.nan
    if len(positive) >= 5:
        alpha = np.polyfit(
            np.log(times[positive]), np.log(msd[positive]), 1)[0]
    return (
        slope / 4.0, intercept, r2, alpha,
        int(selected[0]), int(selected[-1]), int(max_index))


def graph_for_frame(pos, box, cutoff):
    n = len(pos)
    pairs = cKDTree(pos, boxsize=box).query_pairs(cutoff, output_type="ndarray")
    if len(pairs) == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64)
    deg = np.bincount(pairs.ravel(), minlength=n)
    dsu = WeightedDSU(n)
    shifts = np.empty((len(pairs), 2), dtype=np.int64)
    drs = np.empty((len(pairs), 2), dtype=float)
    adjacency = [[] for _ in range(n)]
    for k, (i, j) in enumerate(pairs):
        raw = pos[j] - pos[i]
        shift = -np.rint(raw / box).astype(np.int64)
        dr = raw + shift * box
        shifts[k], drs[k] = shift, dr
        dsu.union(int(i), int(j), shift)
        adjacency[i].append((int(j), dr))
        adjacency[j].append((int(i), -dr))
    roots = np.empty(n, dtype=np.int64)
    images = np.empty((n, 2), dtype=np.int64)
    for i in range(n):
        roots[i], images[i] = dsu.find(i)
    # Path compression can change roots once more; normalize them.
    for i in range(n):
        roots[i], images[i] = dsu.find(i)
    return pairs, deg, roots, images, dsu, adjacency, drs


def component_shapes(pos, box, roots, images, dsu):
    members = defaultdict(list)
    for i, r in enumerate(roots):
        members[int(r)].append(i)
    rgs, asph = [], []
    for root, ids in members.items():
        rr, _ = dsu.find(root)
        if np.any(dsu.wind[rr]) or len(ids) < 2:
            continue
        ids = np.asarray(ids, dtype=np.int64)
        x = pos[ids] + images[ids] * box
        x -= x.mean(axis=0)
        gyr = x.T @ x / len(ids)
        ev = np.linalg.eigvalsh(gyr)
        tr = float(ev.sum())
        rgs.append(math.sqrt(max(tr, 0.0)))
        asph.append(float((ev[1] - ev[0]) / tr) if tr > 0 else 0.0)
    return (np.mean(rgs) if rgs else np.nan,
            np.mean(asph) if asph else np.nan)


def angle_metrics(adjacency):
    turns, curv = [], []
    for nei in adjacency:
        if len(nei) != 2:
            continue
        u, v = nei[0][1], nei[1][1]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0:
            continue
        # Straight chain: the two outward vectors are antiparallel -> turn=0.
        turn = math.acos(float(np.clip(-np.dot(u, v) / (nu * nv), -1, 1)))
        turns.append(turn)
        curv.append(turn / (0.5 * (nu + nv)))
    return (np.mean(turns) if turns else np.nan,
            math.sqrt(np.mean(np.square(curv))) if curv else np.nan)


def two_core_metrics(adjacency, deg):
    """Return 2-core summary and the sizes of its connected components."""
    n = len(adjacency)
    work = np.asarray(deg, dtype=int).copy()
    active = np.ones(n, dtype=bool)
    queue = deque(np.where(work < 2)[0].tolist())
    while queue:
        i = int(queue.popleft())
        if not active[i]:
            continue
        active[i] = False
        for j, _ in adjacency[i]:
            if active[j]:
                work[j] -= 1
                if work[j] == 1:
                    queue.append(j)
    seen = np.zeros(n, dtype=bool)
    sizes = []
    for start in np.where(active)[0]:
        if seen[start]:
            continue
        q = [int(start)]
        seen[start] = True
        size = 0
        while q:
            i = q.pop()
            size += 1
            for j, _ in adjacency[i]:
                if active[j] and not seen[j]:
                    seen[j] = True
                    q.append(j)
        sizes.append(size)
    return (float(np.mean(active)), len(sizes),
            max(sizes) / n if sizes else 0.0, Counter(sizes))


def linear_chain_statistics(adjacency, roots, dsu, max_lag):
    """Analyze finite, unbranched, open components and tangent correlations."""
    members = defaultdict(list)
    for i, root in enumerate(roots):
        members[int(root)].append(i)
    chains = []
    corr_sum = np.zeros(max_lag + 1, dtype=float)
    sep_sum = np.zeros(max_lag + 1, dtype=float)
    corr_count = np.zeros(max_lag + 1, dtype=np.int64)
    for root, ids in members.items():
        rr, _ = dsu.find(root)
        if np.any(dsu.wind[rr]) or len(ids) < 2:
            continue
        ends = [i for i in ids if len(adjacency[i]) == 1]
        if len(ends) != 2 or any(len(adjacency[i]) > 2 for i in ids):
            continue
        current, previous = ends[0], -1
        tangents, lengths = [], []
        visited = {current}
        while current != ends[1]:
            choices = [(j, dr) for j, dr in adjacency[current] if j != previous]
            if len(choices) != 1:
                tangents = []
                break
            nxt, dr = choices[0]
            length = float(np.linalg.norm(dr))
            if length <= 0 or nxt in visited:
                tangents = []
                break
            tangents.append(dr / length)
            lengths.append(length)
            previous, current = current, nxt
            visited.add(current)
        if not tangents or len(visited) != len(ids):
            continue
        tangents = np.asarray(tangents)
        lengths = np.asarray(lengths)
        contour = float(lengths.sum())
        # Sum of directed minimum-image bond vectors unwraps an open chain.
        ree = float(np.linalg.norm(np.sum(tangents * lengths[:, None], axis=0)))
        tort = contour / ree if ree > 0 else np.nan
        chains.append((len(ids), contour, ree, tort))
        nb = len(tangents)
        cumulative = np.r_[0.0, np.cumsum(lengths)]
        for lag in range(min(max_lag, nb - 1) + 1):
            dots = np.sum(tangents[:nb-lag] * tangents[lag:], axis=1)
            if lag == 0:
                seps = np.zeros(nb)
            else:
                seps = cumulative[lag:nb] - cumulative[:nb-lag]
            corr_sum[lag] += float(np.sum(dots))
            sep_sum[lag] += float(np.sum(seps))
            corr_count[lag] += len(dots)
    lp = fit_persistence_length(corr_sum, sep_sum, corr_count)
    return chains, corr_sum, sep_sum, corr_count, lp


def fit_persistence_length(corr_sum, sep_sum, count):
    good = count > 0
    if not np.any(good):
        return np.nan
    c = np.divide(corr_sum, count, out=np.full_like(corr_sum, np.nan),
                  where=good)
    s = np.divide(sep_sum, count, out=np.full_like(sep_sum, np.nan),
                  where=good)
    ids = np.where((np.arange(len(c)) > 0) & np.isfinite(c) &
                   np.isfinite(s) & (c > 0) & (c < 1))[0]
    # Use only the initial positive decay, before the first sign change.
    positive = []
    for i in range(1, len(c)):
        if not np.isfinite(c[i]) or c[i] <= 0:
            break
        if c[i] < 1:
            positive.append(i)
    ids = np.asarray(positive, dtype=int)
    if len(ids) < 3:
        return np.nan
    slope, _ = np.polyfit(s[ids], np.log(c[ids]), 1)
    return float(-1.0 / slope) if slope < 0 else np.nan


def graph_bridges(adjacency, n):
    """Tarjan bridge set, encoded as min(i,j)*n+max(i,j)."""
    tin = np.full(n, -1, dtype=np.int64)
    low = np.empty(n, dtype=np.int64)
    timer = 0
    bridges = set()

    def dfs(v, parent):
        nonlocal timer
        tin[v] = low[v] = timer
        timer += 1
        for to, _ in adjacency[v]:
            if to == parent:
                continue
            if tin[to] >= 0:
                low[v] = min(low[v], tin[to])
            else:
                dfs(to, v)
                low[v] = min(low[v], low[to])
                if low[to] > tin[v]:
                    bridges.add(min(v, to) * n + max(v, to))

    for i in range(n):
        if tin[i] < 0:
            dfs(i, -1)
    return bridges


def fundamental_cycle_lengths(adjacency):
    """Paton fundamental cycle basis; returns one length per independent cycle."""
    remaining = set(range(len(adjacency)))
    lengths = []
    while remaining:
        root = remaining.pop()
        stack = [root]
        pred = {root: root}
        used = {root: set()}
        while stack:
            z = stack.pop()
            zused = used[z]
            for nbr, _ in adjacency[z]:
                if nbr not in used:
                    pred[nbr] = z
                    stack.append(nbr)
                    used[nbr] = {z}
                elif nbr == z:
                    lengths.append(1)
                elif nbr not in zused:
                    nbr_used = used[nbr]
                    length = 2
                    p = pred[z]
                    while p not in nbr_used:
                        length += 1
                        p = pred[p]
                    length += 1
                    lengths.append(length)
                    used[nbr].add(z)
        remaining.difference_update(pred)
    return lengths


def local_directors(adjacency):
    q2 = np.full(len(adjacency), np.nan + 1j*np.nan, dtype=complex)
    for i, nei in enumerate(adjacency):
        if not nei:
            continue
        theta = np.asarray([math.atan2(dr[1], dr[0]) for _, dr in nei])
        z = np.mean(np.exp(2j * theta))
        if abs(z) > 0:
            q2[i] = z / abs(z)
    return q2


def orientational_correlation(pos, box, q2, rmax, bins):
    edges = np.linspace(0.0, rmax, bins + 1)
    sums = np.zeros(bins)
    counts = np.zeros(bins, dtype=np.int64)
    pairs = cKDTree(pos, boxsize=box).query_pairs(rmax, output_type="ndarray")
    if len(pairs):
        dr = pos[pairs[:, 1]] - pos[pairs[:, 0]]
        dr -= np.rint(dr / box) * box
        r = np.linalg.norm(dr, axis=1)
        valid = np.isfinite(q2[pairs[:, 0]]) & np.isfinite(q2[pairs[:, 1]])
        b = np.searchsorted(edges, r, side="right") - 1
        valid &= (b >= 0) & (b < bins)
        val = np.real(q2[pairs[:, 0]] * np.conj(q2[pairs[:, 1]]))
        np.add.at(sums, b[valid], val[valid])
        np.add.at(counts, b[valid], 1)
    return edges, sums, counts


def analyze_frame(pos, box, cutoff, previous_state, tangent_max_lag):
    n = len(pos)
    pairs, deg, roots, images, dsu, adjacency, drs = graph_for_frame(pos, box, cutoff)
    counts = Counter(map(int, roots))
    sizes = np.asarray(list(counts.values()), dtype=int)
    root_list = np.asarray(list(counts.keys()), dtype=np.int64)
    winding = np.asarray([dsu.wind[dsu.find(int(r))[0]] for r in root_list])
    percolates = bool(np.any(winding))
    wx = bool(np.any(winding[:, 0])) if len(winding) else False
    wy = bool(np.any(winding[:, 1])) if len(winding) else False
    finite = sizes[~np.any(winding, axis=1)] if len(winding) else sizes
    fn = float(np.mean(finite)) if len(finite) else np.nan
    fw = float(np.sum(finite**2) / np.sum(finite)) if np.sum(finite) else np.nan
    beta1 = len(pairs) - n + len(sizes)

    # Decompose the cycle rank component by component.  A component with V_c
    # vertices and E_c edges has beta1_c=E_c-V_c+1.  "Winding" refers to any
    # nonzero periodic winding vector, while "nonwinding" includes all finite
    # components, including finite rings.
    edge_counts = Counter(map(int, roots[pairs[:, 0]])) if len(pairs) else Counter()
    component_topology = []
    beta_winding = 0
    beta_nonwinding = 0
    n_winding = 0
    n_cyclic = 0
    for root, size in counts.items():
        rr, _ = dsu.find(int(root))
        b1 = int(edge_counts.get(int(root), 0) - size + 1)
        # Numerical/logic safeguard: the cycle rank of a graph is nonnegative.
        b1 = max(b1, 0)
        wx_c, wy_c = map(bool, dsu.wind[rr])
        if wx_c or wy_c:
            n_winding += 1
            beta_winding += b1
        else:
            beta_nonwinding += b1
        if b1 > 0:
            n_cyclic += 1
        component_topology.append((int(size), b1, int(wx_c), int(wy_c)))

    # Count graph triangles exactly.  This measures local three-particle loops;
    # it is not itself a Betti number because triangles can share edges.
    neighbor_sets = [set(j for j, _ in nei) for nei in adjacency]
    triangles_times_three = 0
    for i, j in pairs:
        triangles_times_three += len(neighbor_sets[int(i)] & neighbor_sets[int(j)])
    n_triangles = triangles_times_three // 3
    connected_triples = int(np.sum(deg * (deg - 1) // 2))
    transitivity = (3.0 * n_triangles / connected_triples
                    if connected_triples else 0.0)

    if len(drs):
        theta = np.arctan2(drs[:, 1], drs[:, 0])
        global_s2 = abs(np.mean(np.exp(2j * theta)))
        local = []
        for nei in adjacency:
            if nei:
                th = np.asarray([math.atan2(x[1][1], x[1][0]) for x in nei])
                local.append(abs(np.mean(np.exp(2j * th))))
        local_psi2 = float(np.mean(local)) if local else np.nan
    else:
        global_s2 = local_psi2 = np.nan

    turn, curvature = angle_metrics(adjacency)
    rg, shape = component_shapes(pos, box, roots, images, dsu)
    core_fraction, n_core, largest_core, core_hist = two_core_metrics(
        adjacency, deg)
    chains, tangent_sum, tangent_sep, tangent_count, persistence = \
        linear_chain_statistics(adjacency, roots, dsu, tangent_max_lag)
    if chains:
        chain_array = np.asarray(chains, dtype=float)
        mean_contour = float(np.nanmean(chain_array[:, 1]))
        mean_ree = float(np.nanmean(chain_array[:, 2]))
        mean_tort = float(np.nanmean(chain_array[:, 3]))
    else:
        mean_contour = mean_ree = mean_tort = np.nan

    edges = set((pairs[:, 0] * n + pairs[:, 1]).tolist())
    bridges = graph_bridges(adjacency, n)
    cycle_lengths = fundamental_cycle_lengths(adjacency)
    # The basis cardinality must equal the graph cycle rank.
    if len(cycle_lengths) != beta1:
        raise RuntimeError(
            f"Cycle-basis inconsistency: basis={len(cycle_lengths)}, beta1={beta1}")
    mean_cycle_length = (float(np.mean(cycle_lengths))
                         if cycle_lengths else np.nan)
    max_cycle_length = max(cycle_lengths) if cycle_lengths else 0
    triangle_cycle_fraction = (
        float(np.mean(np.asarray(cycle_lengths) == 3))
        if cycle_lengths else np.nan)
    q2 = local_directors(adjacency)
    if previous_state is None:
        survival = jaccard = broken = formed = np.nan
        broken_bridge = broken_cycle = formed_merge = formed_cycle = np.nan
        msd_k1 = msd_k2 = msd_k3 = np.nan
        break_k1 = break_k2 = break_k3 = np.nan
    else:
        previous_edges = previous_state["edges"]
        common = len(edges & previous_edges)
        survival = common / len(previous_edges) if previous_edges else np.nan
        union = len(edges | previous_edges)
        jaccard = common / union if union else 1.0
        broken_set = previous_edges - edges
        formed_set = edges - previous_edges
        broken = len(broken_set) / n
        formed = len(formed_set) / n
        broken_bridge = len(broken_set & previous_state["bridges"]) / n
        broken_cycle = len(broken_set - previous_state["bridges"]) / n
        formed_merge_count = 0
        for code in formed_set:
            i, j = divmod(int(code), n)
            formed_merge_count += int(previous_state["roots"][i] !=
                                      previous_state["roots"][j])
        formed_merge = formed_merge_count / n
        formed_cycle = (len(formed_set) - formed_merge_count) / n

        dr_step = pos - previous_state["pos"]
        dr_step -= np.rint(dr_step / box) * box
        dr2 = np.sum(dr_step * dr_step, axis=1)
        pdeg = previous_state["deg"]
        def conditional_mean(mask):
            return float(np.mean(dr2[mask])) if np.any(mask) else np.nan
        msd_k1 = conditional_mean(pdeg == 1)
        msd_k2 = conditional_mean(pdeg == 2)
        msd_k3 = conditional_mean(pdeg >= 3)

        incidences = np.zeros(n, dtype=int)
        for code in broken_set:
            i, j = divmod(int(code), n)
            incidences[i] += 1
            incidences[j] += 1
        def incidence_rate(mask):
            return (float(np.sum(incidences[mask]) / np.sum(mask))
                    if np.any(mask) else np.nan)
        break_k1 = incidence_rate(pdeg == 1)
        break_k2 = incidence_rate(pdeg == 2)
        break_k3 = incidence_rate(pdeg >= 3)

    values = [
        len(pairs), np.mean(deg), np.mean(deg == 1), np.mean(deg >= 3),
        np.mean(deg == 0), len(sizes), np.max(sizes) / n, fn, fw, beta1,
        len(sizes) - beta1, float(percolates), float(wx), float(wy), global_s2,
        local_psi2, turn, curvature, rg, shape, survival, jaccard, broken, formed,
        beta1 / n, beta1 / len(sizes), n_cyclic,
        n_cyclic / len(sizes), n_winding, beta_winding, beta_nonwinding,
        n_triangles, n_triangles / n, transitivity,
        np.mean(deg == 2), np.mean(deg == 3), np.mean(deg == 4),
        np.mean(deg >= 5), core_fraction, n_core, largest_core, len(chains),
        mean_contour, mean_ree, mean_tort, persistence,
        broken_bridge, broken_cycle, formed_merge, formed_cycle,
        msd_k1, msd_k2, msd_k3, break_k1, break_k2, break_k3,
        mean_cycle_length, max_cycle_length, triangle_cycle_fraction,
    ]
    current_state = {
        "edges": edges, "bridges": bridges, "roots": roots.copy(),
        "deg": deg.copy(), "pos": pos.copy(), "q2": q2,
    }
    return (values, current_state, Counter(sizes.tolist()),
            component_topology, chains, core_hist,
            tangent_sum, tangent_sep, tangent_count, Counter(cycle_lengths))


def mean_sem(values):
    x = np.asarray(values, dtype=float)
    good = np.isfinite(x)
    if not np.any(good):
        return np.nan, np.nan, 0
    x = x[good]
    sem = np.std(x, ddof=1) / math.sqrt(len(x)) if len(x) > 1 else np.nan
    return float(np.mean(x)), float(sem), len(x)


def save_table(path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rows, dtype=float)
    if arr.size == 0:
        path.write_text("# " + " ".join(header) + "\n", encoding="utf8")
    else:
        np.savetxt(path, arr, header=" ".join(header), fmt="%.10g")


def discover(root, selected_rhos, selected_seeds):
    states = []
    for rd in sorted(root.glob("rho_*")):
        try:
            rho = parse_value(rd.name, "rho_")
        except ValueError:
            continue
        if selected_rhos and not any(np.isclose(rho, r, atol=5e-7) for r in selected_rhos):
            continue
        for sd in sorted(rd.glob("seed_*")):
            m = re.fullmatch(r"seed_(\d+)", sd.name)
            if not m:
                continue
            seed = int(m.group(1))
            if selected_seeds and seed not in selected_seeds:
                continue
            summary = sd / "cycle_summary.dat"
            lookup = {}
            if summary.exists():
                for row in read_numeric(summary):
                    lookup[(int(row[2]), round(float(row[1]), 8))] = row
            for branch_code, branch in enumerate(("heating", "cooling")):
                for td in sorted((sd / branch).glob("T_*")):
                    traj = td / "trajectory.gsd"
                    if traj.exists():
                        T = parse_value(td.name, "T_")
                        states.append((rho, seed, branch_code, branch, T, td,
                                       lookup.get((branch_code, round(T, 8)))))
    return states


def analyze_state_job(job):
    """Analyze one (rho, seed, branch, T); safe to run in another process."""
    state, out_string, options = job
    cutoff = options["cutoff"]
    frame_stride = options["frame_stride"]
    dt_fallback = options["dt"]
    rho, seed, bcode, branch, T, td, cyc = state
    out = Path(out_string)
    metadata = td.parents[1] / "metadata.json"
    dt = dt_fallback
    if metadata.exists():
        try:
            dt = float(json.loads(metadata.read_text(encoding="utf8"))["dt"])
        except Exception:
            pass
    trajectory_path = td / "trajectory.gsd"
    steps_msd, unwrapped, _, unwrap_mode = load_unwrapped_trajectory(
        trajectory_path)
    if dt is None:
        times_msd = steps_msd - steps_msd[0]
    else:
        times_msd = (steps_msd - steps_msd[0]) * dt
    if not np.all(np.diff(times_msd) > 0):
        raise ValueError(f"Non-increasing GSD time stamps in {trajectory_path}")
    msd, msd_counts = all_origin_msd_fft(unwrapped)
    (d_msd, msd_intercept, msd_r2, msd_alpha,
     msd_ifirst, msd_ilast, msd_imax) = fit_diffusion(
        times_msd, msd, options["msd_max_lag_fraction"],
        options["msd_fit_fraction"])
    frame_rows, cluster_counts, topology_counts = [], Counter(), Counter()
    core_counts = Counter()
    cycle_length_counts = Counter()
    chain_by_size = defaultdict(lambda: np.zeros(5, dtype=float))
    tangent_sum = np.zeros(options["tangent_max_lag"] + 1)
    tangent_sep = np.zeros(options["tangent_max_lag"] + 1)
    tangent_count = np.zeros(options["tangent_max_lag"] + 1, dtype=np.int64)
    betti_records = []
    orient_sum = np.zeros(options["orient_bins"])
    orient_count = np.zeros(options["orient_bins"], dtype=np.int64)
    orient_edges = np.linspace(0, options["orient_rmax"],
                               options["orient_bins"] + 1)
    previous = None
    for analyzed_index, (iframe, step, pos, box) in enumerate(
            trajectory_frames(trajectory_path, frame_stride)):
        result = analyze_frame(pos, box, cutoff, previous,
                               options["tangent_max_lag"])
        (vals, previous, hist, component_topology, chains, core_hist,
         ts, tss, tc, cycle_hist) = result
        time = step * dt if dt is not None else step
        frame_rows.append([iframe, time] + vals)
        cluster_counts.update(hist)
        topology_counts.update(component_topology)
        core_counts.update(core_hist)
        cycle_length_counts.update(cycle_hist)
        tangent_sum += ts
        tangent_sep += tss
        tangent_count += tc
        for size, contour, ree, tort in chains:
            acc = chain_by_size[int(size)]
            acc += [1, contour, ree,
                    tort if np.isfinite(tort) else 0.0,
                    int(np.isfinite(tort))]

        if analyzed_index % options["expensive_stride"] == 0:
            for rc in options["betti_cutoffs"]:
                pp, _, rr, _, dd, _, _ = graph_for_frame(pos, box, rc)
                roots_unique = np.unique(rr)
                beta0 = len(roots_unique)
                beta1_rc = len(pp) - len(pos) + beta0
                nw = sum(bool(np.any(dd.wind[dd.find(int(root))[0]]))
                         for root in roots_unique)
                counts_rc = Counter(map(int, rr))
                largest_rc = max(counts_rc.values()) / len(pos)
                betti_records.append([
                    rho, seed, bcode, T, iframe, time, rc, beta0, beta1_rc,
                    nw, largest_rc,
                ])
            oe, osum, oc = orientational_correlation(
                pos, box, previous["q2"], options["orient_rmax"],
                options["orient_bins"])
            orient_edges = oe
            orient_sum += osum
            orient_count += oc
    save_table(out / "per_seed" / f"rho_{rho:.5f}" / f"seed_{seed}" /
               branch / f"T_{T:.5f}_frames.dat", frame_rows, FRAME_HEADER)
    save_table(
        out / "per_seed" / f"rho_{rho:.5f}" / f"seed_{seed}" /
        branch / f"T_{T:.5f}_msd.dat",
        np.column_stack([
            times_msd[:msd_imax + 1], msd[:msd_imax + 1],
            msd_counts[:msd_imax + 1]]),
        ["tau", "MSD_all_time_origins", "n_time_origins"])
    arr = np.asarray(frame_rows, dtype=float)
    means = [np.nanmean(arr[:, j]) if np.any(np.isfinite(arr[:, j])) else np.nan
             for j in range(2, arr.shape[1])]
    if cyc is not None:
        thermo = list(cyc[3:7])
        d_msd_inline = float(cyc[7])
        d_vacf = float(cyc[8])
        inline_intercept = float(cyc[9])
        inline_nsamples = float(cyc[10])
    else:
        thermo = [np.nan] * 4
        d_msd_inline = d_vacf = inline_intercept = inline_nsamples = np.nan
    mode_code = 0.0 if unwrap_mode == "stored_images" else 1.0
    extra = thermo + [
        d_msd, d_vacf, msd_intercept, msd_r2, msd_alpha,
        times_msd[msd_ifirst], times_msd[msd_ilast], len(times_msd),
        mode_code, d_msd_inline, inline_intercept, inline_nsamples,
    ]
    seed_row = [rho, seed, bcode, T, len(frame_rows)] + means + extra
    total_clusters = sum(cluster_counts.values())
    cluster_records = []
    for size, count in sorted(cluster_counts.items()):
        cluster_records.append([rho, seed, bcode, T, size, count,
                                count / total_clusters if total_clusters else np.nan])
    topology_records = []
    total_components = sum(topology_counts.values())
    for (size, beta, wx, wy), count in sorted(topology_counts.items()):
        topology_records.append([
            rho, seed, bcode, T, size, beta, wx, wy, count,
            count / total_components if total_components else np.nan,
        ])
    chain_records = []
    for size, acc in sorted(chain_by_size.items()):
        count = int(acc[0])
        chain_records.append([
            rho, seed, bcode, T, size, count, acc[1] / count,
            acc[2] / count,
            acc[3] / acc[4] if acc[4] else np.nan,
        ])
    total_cores = sum(core_counts.values())
    core_records = [
        [rho, seed, bcode, T, size, count,
         count / total_cores if total_cores else np.nan]
        for size, count in sorted(core_counts.items())
    ]
    tangent_records = []
    for lag in range(len(tangent_count)):
        if tangent_count[lag]:
            tangent_records.append([
                rho, seed, bcode, T, lag,
                tangent_sep[lag] / tangent_count[lag],
                tangent_sum[lag] / tangent_count[lag], tangent_count[lag],
            ])
    orient_records = []
    for ib in range(options["orient_bins"]):
        if orient_count[ib]:
            orient_records.append([
                rho, seed, bcode, T,
                0.5 * (orient_edges[ib] + orient_edges[ib + 1]),
                orient_sum[ib] / orient_count[ib], orient_count[ib],
            ])
    total_basis_cycles = sum(cycle_length_counts.values())
    cycle_length_records = [
        [rho, seed, bcode, T, length, count,
         count / total_basis_cycles if total_basis_cycles else np.nan]
        for length, count in sorted(cycle_length_counts.items())
    ]
    return (seed_row, cluster_records, topology_records, chain_records,
            core_records, tangent_records, betti_records, orient_records,
            cycle_length_records)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", default="thermal_cycles", help="simulation output root")
    p.add_argument("--out", default=None, help="analysis directory (default ROOT/analysis)")
    p.add_argument("--rhos", type=float, nargs="*", default=[.395, .405, .415])
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--bond-cutoff", type=float, default=1.5,
                   help="neighbor cutoff between the first (~1.2) and second (~2.0) scales")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--expensive-stride", type=int, default=20,
                   help="analyzed-frame stride for cutoff scans and C2(r)")
    p.add_argument("--betti-cutoffs", type=float, nargs="+",
                   default=[1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65])
    p.add_argument("--tangent-max-lag", type=int, default=30,
                   help="maximum bond separation in tangent correlations")
    p.add_argument("--orient-rmax", type=float, default=6.0)
    p.add_argument("--orient-bins", type=int, default=30)
    p.add_argument("--workers", type=int, default=1,
                   help="independent state points analyzed in parallel")
    p.add_argument("--dt", type=float, default=None,
                   help="MD timestep; metadata value is preferred, else this value, else time=GSD step")
    p.add_argument("--msd-max-lag-fraction", type=float, default=0.5,
                   help="largest MSD lag retained as a fraction of the GSD trajectory")
    p.add_argument("--msd-fit-fraction", type=float, default=0.5,
                   help="final fraction of the retained MSD interval used in the linear fit")
    a = p.parse_args()
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "analysis"
    states = discover(root, a.rhos, a.seeds)
    if not states:
        raise SystemExit(f"No trajectory.gsd files found below {root.resolve()}")

    seed_rows = []
    seed_header = ["rho", "seed", "branch", "T", "nframes"]
    seed_header += [f"{x}_mean" for x in FRAME_HEADER[2:]]
    # Thermodynamics and VACF diffusion come from cycle_summary.dat.  D_msd is
    # recomputed here from the stored GSD trajectory.
    seed_header += [
        "E_per_N", "E_std_per_N", "P", "P_std",
        "D_msd", "D_vacf", "msd_intercept", "msd_fit_R2",
        "msd_fit_loglog_alpha", "msd_fit_tau_min", "msd_fit_tau_max",
        "n_msd_frames", "msd_unwrap_mode_0images_1minimum_image",
        "D_msd_inline", "msd_inline_intercept", "nsamples_inline",
    ]
    cluster_records = []
    topology_records = []
    chain_records = []
    core_records = []
    tangent_records = []
    betti_records = []
    orient_records = []
    cycle_length_records = []

    options = {
        "cutoff": a.bond_cutoff, "frame_stride": a.frame_stride, "dt": a.dt,
        "expensive_stride": a.expensive_stride,
        "betti_cutoffs": a.betti_cutoffs,
        "tangent_max_lag": a.tangent_max_lag,
        "orient_rmax": a.orient_rmax, "orient_bins": a.orient_bins,
        "msd_max_lag_fraction": a.msd_max_lag_fraction,
        "msd_fit_fraction": a.msd_fit_fraction,
    }
    jobs = [(state, str(out), options) for state in states]
    if a.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
            results = pool.map(analyze_state_job, jobs)
            for istate, result in enumerate(results, 1):
                (seed_row, records, top_records, chains, cores, tangents,
                 bettis, orients, cycle_lengths) = result
                seed_rows.append(seed_row)
                cluster_records.extend(records)
                topology_records.extend(top_records)
                chain_records.extend(chains)
                core_records.extend(cores)
                tangent_records.extend(tangents)
                betti_records.extend(bettis)
                orient_records.extend(orients)
                cycle_length_records.extend(cycle_lengths)
                print(f"[{istate}/{len(states)}] completed", flush=True)
    else:
        for istate, job in enumerate(jobs, 1):
            result = analyze_state_job(job)
            (seed_row, records, top_records, chains, cores, tangents,
             bettis, orients, cycle_lengths) = result
            seed_rows.append(seed_row)
            cluster_records.extend(records)
            topology_records.extend(top_records)
            chain_records.extend(chains)
            core_records.extend(cores)
            tangent_records.extend(tangents)
            betti_records.extend(bettis)
            orient_records.extend(orients)
            cycle_length_records.extend(cycle_lengths)
            print(f"[{istate}/{len(states)}] completed", flush=True)

    save_table(out / "statepoints_by_seed.dat", seed_rows, seed_header)
    save_table(out / "cluster_size_distribution_by_seed.dat", cluster_records,
               ["rho", "seed", "branch", "T", "cluster_size", "count", "probability"])
    save_table(out / "cluster_topology_distribution_by_seed.dat", topology_records,
               ["rho", "seed", "branch", "T", "cluster_size", "cluster_beta1",
                "winds_x", "winds_y", "count", "probability"])
    save_table(out / "linear_chain_statistics_by_seed.dat", chain_records,
               ["rho", "seed", "branch", "T", "chain_size", "count",
                "mean_contour_length", "mean_end_to_end", "mean_tortuosity"])
    save_table(out / "two_core_size_distribution_by_seed.dat", core_records,
               ["rho", "seed", "branch", "T", "two_core_component_size",
                "count", "probability"])
    save_table(out / "tangent_correlation_by_seed.dat", tangent_records,
               ["rho", "seed", "branch", "T", "bond_lag",
                "mean_contour_separation", "tangent_correlation", "count"])
    save_table(out / "betti_vs_cutoff_by_seed.dat", betti_records,
               ["rho", "seed", "branch", "T", "frame", "time", "cutoff",
                "beta0", "beta1", "n_winding_components", "largest_fraction"])
    save_table(out / "orientational_correlation_by_seed.dat", orient_records,
               ["rho", "seed", "branch", "T", "r", "C2", "count"])
    save_table(out / "fundamental_cycle_length_distribution_by_seed.dat",
               cycle_length_records,
               ["rho", "seed", "branch", "T", "cycle_length", "count",
                "probability"])

    # Equal-weight aggregation over independent seeds.  Columns are mean and SEM.
    groups = defaultdict(list)
    for row in seed_rows:
        groups[(row[0], int(row[2]), row[3])].append(row)
    ensemble_rows = []
    metric_names = seed_header[5:]
    ensemble_header = ["rho", "branch", "T", "nseeds"]
    for name in metric_names:
        ensemble_header += [name + "_seedmean", name + "_sem"]
    for (rho, branch, T), rows in sorted(groups.items()):
        row = [rho, branch, T, len(rows)]
        vals = np.asarray(rows, dtype=float)
        for j in range(5, vals.shape[1]):
            mu, sem, _ = mean_sem(vals[:, j])
            row += [mu, sem]
        ensemble_rows.append(row)
    save_table(out / "ensemble_summary.dat", ensemble_rows, ensemble_header)

    (out / "README.txt").write_text(
        "Analysis of LJ+Gaussian stripe thermal cycles\n"
        "branch: 0=heating, 1=cooling. Angles are radians.\n"
        f"Bond cutoff: {a.bond_cutoff:g}; frame stride: {a.frame_stride}; workers: {a.workers}.\n\n"
        f"Expensive-observable stride: {a.expensive_stride}; Betti cutoffs: "
        + " ".join(map(str, a.betti_cutoffs)) + ".\n\n"
        "statepoints_by_seed.dat: time-averaged observables for every replica.\n"
        "ensemble_summary.dat: equal-weight seed mean and SEM at each state point.\n"
        "cluster_size_distribution_by_seed.dat: P(s), retaining replica identity.\n"
        "cluster_topology_distribution_by_seed.dat: joint distribution of cluster\n"
        "size, cluster beta1, and x/y winding, retaining replica identity.\n"
        "linear_chain_statistics_by_seed.dat: contour length, end-to-end distance,\n"
        "and tortuosity of finite unbranched open chains, grouped by chain size.\n"
        "two_core_size_distribution_by_seed.dat: sizes of connected 2-core pieces.\n"
        "tangent_correlation_by_seed.dat: tangent correlation versus contour lag.\n"
        "betti_vs_cutoff_by_seed.dat: cutoff robustness of beta0/beta1 and winding.\n"
        "orientational_correlation_by_seed.dat: local-director C2(r).\n"
        "fundamental_cycle_length_distribution_by_seed.dat: lengths in a Paton\n"
        "fundamental cycle basis; its cardinality equals beta1, while individual\n"
        "cycle lengths are basis-dependent rather than unique topological invariants.\n"
        "per_seed/.../*_frames.dat: full frame-resolved measurements.\n\n"
        "Topology uses the neighbor graph under periodic boundaries. percolates and\n"
        "winds_x/y detect nonzero winding. beta1=E-V+C; Euler=C-beta1. Shape\n"
        "metrics exclude winding clusters because their radius of gyration depends\n"
        "on the periodic representation. bond_survival, bond_jaccard and bond\n"
        "breaking/formation refer to consecutive saved GSD frames, not MD steps.\n"
        "beta1_winding_components sums beta1 over components with periodic winding;\n"
        "beta1_nonwinding_components sums it over components without winding.\n"
        "n_triangles counts local 3-cycles exactly; graph_transitivity is three\n"
        "times the triangle count divided by the number of connected triples.\n"
        "The 2-core is obtained by recursively removing vertices of degree <2.\n"
        "Linear-chain statistics exclude winding, branched, cyclic, and isolated\n"
        "components. Event classes use edge status in the previous saved frame:\n"
        "bridge/nonbridge breaks and inter/intracomponent bond formation. They are\n"
        "effective snapshot-resolved events when several changes occur per interval.\n"
        "step_msd_degree* is a one-GSD-interval displacement conditioned on the\n"
        "particle degree at the beginning of that interval, not a long-time MSD.\n"
        "D_msd is recomputed from the GSD trajectory with all time origins. New\n"
        "GSD files are unwrapped from stored particle image counters; legacy files\n"
        "use minimum-image frame-to-frame increments. The fit uses the final\n"
        f"{a.msd_fit_fraction:g} fraction of lags up to "
        f"{a.msd_max_lag_fraction:g} of the trajectory. D_vacf is copied from\n"
        "cycle_summary.dat. D_msd_inline is retained only as a diagnostic of the\n"
        "value evaluated during simulation. Per-state MSD curves and origin counts\n"
        "are written below per_seed/ as *_msd.dat files.\n",
        encoding="utf8")
    print(f"Wrote analysis to {out.resolve()}")


if __name__ == "__main__":
    main()

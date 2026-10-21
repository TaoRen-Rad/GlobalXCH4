import numpy as np


def _select_groups(weights, target, seed, max_sum):
    groups = list(weights)
    np.random.default_rng(seed).shuffle(groups)
    parents = {0: None}
    for group in groups:
        updates = {}
        for total in parents:
            new_total = total + weights[group]
            if new_total <= max_sum and new_total not in parents and new_total not in updates:
                updates[new_total] = (total, group)
        parents.update(updates)
    best = min(parents, key=lambda value: (abs(value - target), -value))
    selected = set()
    total = best
    while total > 0:
        total, group = parents[total]
        selected.add(group)
    return selected, best


def split_by_time_sequence(df, val_ratio=0.2, seed=42, spatial=True):
    """Split whole overpasses, optionally preserving spatial coverage."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if df['time_sequence'].isna().any():
        raise ValueError("time_sequence contains missing values")
    sizes = df.groupby('time_sequence', sort=False).size()
    if len(sizes) < 2:
        raise ValueError("At least two overpasses are required")
    if not spatial:
        groups = sizes.index.to_numpy(dtype=object)
        groups = groups[np.random.default_rng(seed).permutation(len(groups))]
        cumulative = sizes.loc[groups].to_numpy().cumsum()
        position = int(np.argmin(np.abs(cumulative - val_ratio * len(df)))) + 1
        position = min(max(position, 1), len(groups) - 1)
        val_groups = set(groups[:position])
    else:
        if not np.isfinite(df[['latitude', 'longitude']].to_numpy(dtype=float)).all():
            raise ValueError("Split coordinates contain NaN or Inf")
        stats = df.groupby('time_sequence', sort=False).agg(
            n=('time_sequence', 'size'), lat=('latitude', 'mean'), lon=('longitude', 'mean')
        )
        lat_bin = np.clip(np.digitize(stats['lat'], np.linspace(-90, 90, 9)) - 1, 0, 7)
        lon_bin = np.clip(np.digitize(stats['lon'], np.linspace(-180, 180, 17)) - 1, 0, 15)
        stats['bin_id'] = lat_bin * 16 + lon_bin
        target = int(round(val_ratio * len(df)))
        val_groups = set()
        for bin_id, subset in stats.groupby('bin_id', sort=True):
            rows = int(subset['n'].sum())
            chosen, _ = _select_groups(
                subset['n'].astype(int).to_dict(), int(round(val_ratio * rows)),
                seed + int(bin_id), rows,
            )
            val_groups |= chosen
        val_rows = int(stats.loc[list(val_groups), 'n'].sum())
        if val_rows > target:
            val_groups, val_rows = _select_groups(
                stats.loc[list(val_groups), 'n'].astype(int).to_dict(),
                target, seed + 1000, target,
            )
        if val_rows < target:
            remaining = {group: int(stats.loc[group, 'n']) for group in stats.index if group not in val_groups}
            chosen, added = _select_groups(remaining, target - val_rows, seed + 2000, target - val_rows)
            if added != target - val_rows:
                raise ValueError("Cannot reach the requested split size with whole overpasses")
            val_groups |= chosen
    mask = df['time_sequence'].isin(val_groups)
    train = df.loc[~mask].reset_index(drop=True)
    val = df.loc[mask].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError("Training and validation sets must be nonempty")
    return train, val

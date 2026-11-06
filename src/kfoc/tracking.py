def init_run(cfg: dict, enabled: bool = True, name: str = None):
    """Start a W&B run that logs the full config, or return None when disabled.

    Args:
        cfg: Experiment config; ``cfg["wandb"]`` holds ``project`` and ``tags``.
        enabled: Set False to run without W&B (e.g. smoke tests).
        name: Optional run name.

    Returns:
        A ``wandb.Run`` object, or None.
    """
    if not enabled:
        return None
    import wandb

    wb = cfg.get("wandb", {})
    return wandb.init(project=wb.get("project"), tags=wb.get("tags"), config=cfg, name=name)


def log(run, metrics: dict, step: int = None) -> None:
    """Log metrics to W&B if a run is active."""
    if run is not None:
        run.log(metrics, step=step)

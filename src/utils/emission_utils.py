"""
utils/emission_utils.py
=======================
Shared energy-measurement helper, built on CodeCarbon.

Methodology
-----------
Following Schodl et al. 2025 ("Investigating Carbon Footprint of Recommender
Systems Beyond Training Time", RecSys '25), we report **raw electrical energy
consumption in kWh** rather than converting it into a CO2-equivalent figure
via a fixed carbon-intensity factor. A fixed factor (e.g. the commonly used
475 gCO2/kWh world average) is only a constant scaling of the energy value -
it adds no information and can mislead readers into thinking location/time
-specific carbon impact was modelled. Reporting kWh lets anyone later apply
a local carbon-intensity value or a datacenter PUE without re-running any
experiment.

CodeCarbon samples system power sensors (CPU/GPU/RAM) throughout a run and
exposes the accumulated energy via ``tracker.final_emissions_data.energy_consumed``
(kWh) once ``tracker.stop()`` has been called.

Warm-up + repeat protocol
--------------------------
Very short operations (a single inference call, a fast preprocessing step)
complete faster than CodeCarbon's power-sampling interval, which makes a
single measurement noisy or occasionally zero. Rather than fabricating a
number from wall-clock time x a CPU TDP constant (which is not a
measurement), we follow the paper's protocol: run the operation once
untracked as a warm-up (JIT / cache warm-up, avoids first-call overhead
bias), then measure it over several repeated calls, and repeat that whole
measurement several times so we can report a mean and standard deviation.
"""

import warnings
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

warnings.filterwarnings("ignore")


def _default_results_dir(results_dir: Path | None) -> Path:
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _make_tracker(label: str, results_dir: Path, output_file: str):
    from codecarbon import EmissionsTracker

    return EmissionsTracker(
        project_name=label,
        output_dir=str(results_dir),
        output_file=output_file,
        log_level="error",
        save_to_file=True,
        measure_power_secs=0.1,
        allow_multiple_runs=True,
    )


def _run_tracked(label, results_dir, output_file, fn, args, kwargs, max_attempts: int = 1):
    """
    Start a tracker, call ``fn(*args, **kwargs)``, stop the tracker, and
    return ``(result, energy_kwh, duration_s)``.

    CodeCarbon's ``EmissionsTracker.stop()`` is internally decorated with
    ``@suppress(Exception)`` - if something goes wrong inside it (observed in
    practice on this project's Windows+NVIDIA setup as an intermittent,
    seemingly load-dependent power-sensor query failure - not simply "too
    many trackers too fast," since it did not reproduce in isolated repro
    scripts, only after sustained real GPU/CPU training load), it silently
    swallows the error and leaves ``final_emissions_data`` unset instead of
    raising. We detect that case and retry with a fresh tracker after a short
    backoff - safe here because every caller's *fn* is either a pure
    reduction function or constructs-and-trains a brand new model instance,
    so re-invoking it has no side effects to duplicate.

    If every attempt fails, we do NOT raise: a multi-hour unattended sweep
    losing all remaining jobs over one instrumentation hiccup is worse than
    recording this single measurement as missing. *result* (fn's real return
    value - the reduced dataframe, or the trained model) is still returned so
    the caller can keep using it; only the energy/duration come back as NaN,
    clearly flagged, rather than a silently wrong number.
    """
    import time

    last_error = None
    result = None
    for attempt in range(1, max_attempts + 1):
        tracker = _make_tracker(label, results_dir, output_file)
        tracker.start()
        result = fn(*args, **kwargs)
        tracker.stop()
        data = getattr(tracker, "final_emissions_data", None)
        if data is not None:
            return result, data.energy_consumed, data.duration
        last_error = (
            "CodeCarbon's tracker.stop() did not produce emissions data "
            "(likely a transient power-sensor query failure)"
        )
        if attempt < max_attempts:
            print(f"   [WARN] {label}: {last_error}; retrying ({attempt}/{max_attempts}) ...")
            time.sleep(2 * attempt)   # backoff, in case it's a brief resource-contention window

    print(f"   [WARN] {label}: {last_error} after {max_attempts} attempts - recording energy as NaN, continuing.")
    return result, float("nan"), float("nan")


def measure_emissions(
    label: str,
    fn: Callable,
    *args,
    results_dir: Path | None = None,
    output_file: str = "codecarbon_raw.csv",
    **kwargs,
):
    """
    Run *fn(*args, **kwargs)* once while tracking energy consumption.

    This is a single-shot measurement - prefer :func:`measure_repeated` for
    anything shorter than a couple of seconds, since a single CodeCarbon
    sample is noisy at that timescale.

    Returns
    -------
    result       : Whatever *fn* returns.
    energy_kwh   : Measured electrical energy in kWh (CPU + GPU + RAM).
    duration_s   : Wall-clock time in seconds.
    """
    results_dir = _default_results_dir(results_dir)
    result, energy_kwh, duration_s = _run_tracked(label, results_dir, output_file, fn, args, kwargs)

    print(f"   [{label}]  {duration_s*1000:.1f} ms  ->  {energy_kwh:.8f} kWh")
    return result, energy_kwh, duration_s


def measure_repeated(
    label: str,
    fn: Callable,
    *args,
    results_dir: Path | None = None,
    output_file: str = "codecarbon_raw.csv",
    warmup: int = 1,
    repeats: int = 10,
    runs: int = 10,
    **kwargs,
) -> Tuple[object, float, float, float, float]:
    """
    Warm-up + repeat energy measurement (Schodl et al. 2025 protocol).

    Procedure
    ---------
    1. Call *fn* *warmup* times, untracked, to discard first-call overhead.
    2. For each of *runs* outer runs:
         - start the tracker
         - call *fn* *repeats* times
         - stop the tracker; divide the run's total energy/duration by
           *repeats* to get a per-call estimate
    3. Report the mean and standard deviation of the per-call energy (and
       duration) across the *runs* outer runs.

    Parameters
    ----------
    warmup  : number of untracked warm-up calls (>=1 recommended).
    repeats : number of tracked calls per measured run (paper uses 10 for
              inference; use a smaller number for expensive operations like
              full model training).
    runs    : number of independent measured runs to average over (paper
              uses 10; reduce for expensive operations).

    Returns
    -------
    result        : Return value of *fn* from the very last call.
    mean_kwh      : Mean per-call energy across runs (kWh).
    std_kwh       : Standard deviation of per-call energy across runs (kWh).
    mean_duration : Mean per-call wall-clock duration across runs (s).
    std_duration  : Standard deviation of per-call duration across runs (s).
    """
    results_dir = _default_results_dir(results_dir)

    result = None
    for _ in range(warmup):
        result = fn(*args, **kwargs)

    per_run_kwh = []
    per_run_duration = []

    def _batch(*a, **kw):
        r = None
        for _ in range(repeats):
            r = fn(*a, **kw)
        return r

    for run_idx in range(runs):
        result, run_energy_kwh_total, run_duration_s_total = _run_tracked(
            f"{label} | run{run_idx}", results_dir, output_file, _batch, args, kwargs,
        )
        per_run_kwh.append(run_energy_kwh_total / repeats)
        per_run_duration.append(run_duration_s_total / repeats)

    # nan-safe: _run_tracked returns NaN for a run where CodeCarbon's tracker
    # failed even after its own internal retries (see _run_tracked docstring)
    mean_kwh      = float(np.nanmean(per_run_kwh))
    std_kwh       = float(np.nanstd(per_run_kwh))
    mean_duration = float(np.nanmean(per_run_duration))
    std_duration  = float(np.nanstd(per_run_duration))

    print(
        f"   [{label}]  {mean_duration*1000:.2f}±{std_duration*1000:.2f} ms/call  ->  "
        f"{mean_kwh:.8f}±{std_kwh:.8f} kWh/call  "
        f"({runs} runs x {repeats} calls, {warmup} warm-up)"
    )
    return result, mean_kwh, std_kwh, mean_duration, std_duration

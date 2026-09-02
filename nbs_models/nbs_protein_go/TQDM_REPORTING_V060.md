# LATENCE NBS v0.6.0 perfopt — tqdm/reporting optimization

This patch is presentation-only. It does **not** change sampling, model forward,
loss weights, optimizer, scheduler, checkpoint contents, DDP reduction, or the
metrics stored in `training_history.json`.

## Why the original display overflows

The original v0.6.0 code places optimization losses, supervision counts,
weak-focus diagnostics, candidate-capacity diagnostics, LR and GPU memory into a
single `tqdm.set_postfix()` mapping. `dynamic_ncols=True` can resize the bar but
cannot make an intrinsically long postfix fit a narrow terminal.

The original epoch summary has the same problem: dozens of unrelated metrics are
printed as one physical line.

## New display hierarchy

The new config `configs/bp_fixed_epoch_v0.6.0_perfopt_tqdmopt.json` now opts into
**true live multi-line mode**:

- `progress_display_mode = "multiline"`: rank 0 owns five persistent terminal
  rows: one progress bar plus four live status rows (`objective`, `sampling`,
  `weak-flow`, `runtime`).
- `progress_update_interval = 1`: update the live metrics every step.
- `progress_detail_interval = null`: do not additionally append permanent
  diagnostic blocks by default; set it to `20` (or another positive interval)
  when immutable step snapshots are useful.
- `progress_postfix_mode = "compact"`: retained as the fallback profile if the
  display is switched back to `singleline`.
- `epoch_summary_mode = "multiline"`: split objective, sampling, query, weak,
  protein, ontology and runtime summaries into readable lines.

Existing configs remain backward-compatible because the dataclass defaults are
`progress_display_mode="singleline"`, `progress_postfix_mode="full"`,
`progress_update_interval=None`, `progress_detail_interval=None`, and
`epoch_summary_mode="legacy"`.

## Profiles

`minimal` is useful for very narrow SSH terminals:

```text
NBS e3/20: ... L=0.0417, G=0.000551, P=0.118, lr=8.2e-05, M=39/42G
```

`compact` is recommended for normal training:

```text
NBS e3/20: ... L=0.0417, G=0.000551, P=0.118, gp=64, hd=159, pp=1014, bg%=98%, lr=8.2e-05, M=39/42G
```

Detailed diagnostics are printed outside the bar:

```text
[E003 S0020 G0000060] objective L=... G=... P=... cG=... cP=... cA=... H=... cH=...
    sampling  gp=64 hd=159 pp=1014 bg=1003/1024(98.0%) rows=pos:64 neg:63 posonly:1 mixed:63 capdrop=0.0%
    weak      wfq=16/16 wfa=16 wpa=96 wpq=... wphit=... wpr=... wft=... qps=... qcan=...
    runtime   hpair=... qseen=... gscale=... lr=... slr=... mem=39/42G
```

## Runtime overrides

The launcher accepts environment overrides:

```bash
export NBS_PROGRESS_DISPLAY_MODE=multiline # singleline|multiline
export NBS_PROGRESS_POSTFIX_MODE=compact    # minimal|compact|full; singleline only
export NBS_PROGRESS_UPDATE_INTERVAL=1
export NBS_PROGRESS_DETAIL_INTERVAL=20     # 0 disables detailed blocks
export NBS_PROGRESS_DETAIL_ON_LAST=1
export NBS_EPOCH_SUMMARY_MODE=multiline    # legacy|multiline
```

For a very narrow terminal, use `minimal` and increase the detailed interval;
for log-file-oriented runs, disable the progress bar entirely and retain the
existing line logger.


## True live multi-line mode

The `tqdmopt` config now sets `progress_display_mode = "multiline"`.  This is
different from periodic `tqdm.write()` diagnostics: rank 0 owns five persistent
terminal rows (one progress bar plus four status rows: objective, sampling,
weak-flow, runtime), and the rows are refreshed in-place.

Runtime override:

```bash
export NBS_PROGRESS_DISPLAY_MODE=multiline
```

`NBS_EPOCH_SUMMARY_MODE=multiline` only controls the post-epoch summary and does
not select the live training layout.  Set `NBS_PROGRESS_DETAIL_INTERVAL=20` only
if immutable 20-step diagnostic snapshots are also desired; the optimized config
leaves it disabled because the four live rows already expose the same categories.

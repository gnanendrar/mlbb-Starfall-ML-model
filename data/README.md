# Data provenance

`game/` contains real phone frames and capture manifests. No labels are assumed to be correct until they have been manually reviewed. `templates/` contains manually selected sprites and UI elements extracted from observed phone screenshots.

The test suite creates one/two/three-heart fixture variants only in memory for software verification. Those variants are synthetic tests and are never saved as real game observations or detector training data. The numerical simulator is also explicitly synthetic.

Collect distinct gameplay sessions, label frames, run validation, then split by independent session groups. Review overlapping meteors, player glow, lost-heart appearance, and unfamiliar power-ups separately. A high template score on its source frame is not independent accuracy evidence.

`player-reviewed/` contains 31 real positives with approximate avatar boxes. Codex visually reviewed template proposals against the four saved review grids, excluding the paused-screen proposal. The detector experiment uses one entire attempt per split (7 train, 12 val, 12 test); cross-attempt near duplicates were checked and none met the merge threshold. Selection favored template-detected frames, so these labels do not constitute exhaustive accuracy coverage. The original captures and manifests remain under `game/`; reviewed files are copies with annotation provenance.

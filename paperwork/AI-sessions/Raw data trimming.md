Topic: Raw data trimming
Bot: Claude
Date: 24 07 09

Here's a consolidated view of where we've landed.

## Capture setup (recap)
- Flutter `Listener` on iPad, ~8–13ms timestamp diffs (≈77–125Hz, consistent with Pencil's native rate plus minor jitter).
- Per-sample: normalized (x,y), pen down/up, timestamp, pressure (raw range `[0, 25/6]`, displayed as `pressure/pressureMax`).
- Training data (KanjiVG → de Casteljau sampling, RDP only for the ~5% of characters exceeding 256 points) is clean, geometry-driven, no time/pressure channel, and fed to the model as **relative (delta) coordinates**. This matters because cleaning ink data ultimately needs to produce clean *deltas*, and legitimate short strokes/corners must not be mistaken for artifacts.
- Contributing factors you flagged: low-friction glass (encourages overshoot/continued travel at direction changes) and above-average-but-not-native handwriting proficiency (larger variance in artifact amplitude/duration) — both argue for adaptive/relative thresholds over fixed global constants.

## Artifact taxonomy — open strokes
Across 年, セ, and 田 we found a consistent **asymmetric** picture, not one symmetric noise phenomenon:

**Start-side ("dither"):** low/near-zero speed, high and erratic curvature (repeated swings up to full ±π reversals), pressure sitting at/near its floor value then dipping briefly below it before the real ramp begins. Followed by a **sharp, simultaneous** transition into sustained speed + calm curvature — the "commit" point. Seen consistently in 年-stroke-1, セ-stroke-1-start, パ's start.

**End-side:** pressure behaves identically every time — smooth decay to exactly 0, no floor, no bump, no separate anomaly. But the *geometry* shows at least **two distinct sub-variants**:
- *Continuous overshoot* (セ's end): stays fast throughout, curvature thrashes hard, never fully stops, doesn't self-intersect.
- *Stop-then-flick* (田's loop): decelerates to a genuine momentary full stop exactly at the tightest curvature point (this is the actual self-crossing), then a short fast flick away from that stop just before pen-up.

**Practical implication:** pressure is only useful as a start-side signal (and as a "definitely near the true end" marker on the exit side) — it can't detect or characterize the end-artifact itself. End-side detection needs to tolerate both kinematic regimes rather than assuming one universal signature.

## A key confound: legitimate mid-stroke corners
セ's stroke had a genuine direction-change pause mid-stroke that reproduced the *same* low-speed/high-curvature signature as the start-dither. Kinematics alone cannot distinguish "artifact at the edge" from "honest corner in the middle." → **This is why your windowing idea matters**: bounding candidate regions to a window near the stroke's own recorded start/end (rather than scanning the whole stroke for the pattern) resolves this by scope, not by a sharper threshold.

## Windowing idea, refined
Start-dither and end-transient durations look roughly similar in absolute time (~140–180ms) despite very different total stroke durations (400ms vs. 950ms) — suggesting a fixed motor-control timescale rather than a percentage of stroke length. Proposed shape: `min(fixed_ms, frac * stroke_duration)` per side, sized independently for start vs. end (they are not symmetric), ideally calibrated from the "commit transition" timestamp distribution across a real sample (20-30 strokes) rather than eyeballed from two examples.

## Tooling built
`stroke_diagnostic_plot.py`: shared-colormap path+channel plot, plus `local_speed()` and `local_curvature()` (turning-angle based, deliberately **not** Menger curvature, since Menger's arc-length denominator blows up exactly at slow hook apexes — turning angle stays well-behaved there). Verified against straight-line (→0) and right-angle-corner (→clean π/2 spike) test cases. You added a `closed=True` mode for curvature (wraparound turning angle) and speed (with the fix still needed: the wraparound distance has no valid `dt`, should return `NaN`/gap rather than a bogus large value, as seen in ポ's opening speed spike).

## Handakuten/dakuten — parked, with reasoning
Closed loops don't have a converged signature yet — パ and ポ disagree with each other on seam curvature sign and magnitude, and ポ's legitimate mid-loop pinch is kinematically indistinguishable from an artifact. One solid finding survives: **position of a self-intersection relative to the stroke's own start point** is a much better discriminator than kinematics (パ/ポ's closures sit right at their own start; 田's artifact loop had no such relationship). Decision: **defer full handling**, but add a cheap **detect-and-flag guard** (short arc length OR self-intersection OR high cumulative |curvature| per arc length) that routes suspect strokes to manual review instead of running the open-stroke rule on them blind — which also doubles as sample collection for later.

## Where this leaves implementation
Ready to draft: (a) start-trim rule using bounded-window dither detection + pressure-dip corroboration, (b) end-trim rule handling both overshoot variants within its own bounded window, (c) the short/closed-stroke guard, (d) short-stroke safeguards so trimming can't eat a legitimate short stroke entirely. Main open question before hardening thresholds: whether to eyeball-tune from the handful of examples so far, or first pull the commit-transition timing distribution across a larger sample.
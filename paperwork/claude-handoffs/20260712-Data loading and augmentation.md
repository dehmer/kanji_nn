Totally fair — here's a compact one, aimed at giving a new conversation just enough to pick up cleanly without re-litigating everything.

---

**Project: Kanji classification with LSTM — handwritten stroke data (iPad/Apple Pencil), trained on KanjiVG**

**Background:** New to ML/deep learning, but a seasoned (35+ yr) software engineer with a Haskell/Scheme/functional leaning — strong preference for pure functions and composition over OOP/object state. PyTorch, macOS M2 Pro/32GB, MPS backend, `num_workers=0` (multiprocessing didn't help on this hardware). LSTM architecture choice was somewhat arbitrary (borrowed blueprint from a friend); transformer possible later. Currently training on ~1 sample per character deliberately, purely to learn the pipeline/mechanics — aware this won't generalize yet.

**Data formats:**
- *Source*: KanjiVG stroke data, stored as WKB (MultiLineString per character) + CSV index (label, byte offset, length).
- *Target tensor*: 4 columns — `Δx, Δy, s(t), pen`. `s(t)` = arc length normalized **per stroke** (resets each stroke, chosen over whole-character cumulative to avoid drift compounding). Two special rules: (1) very first point of a character has `Δx=Δy=0` by definition; (2) each stroke's last point is duplicated — first copy pen=1, second copy pen=0 with `Δx=Δy=0` — and the jump vector to the next stroke's first point is computed from the last *real* point, pen=1. No absolute position is ever encoded in the tensor.

**Pipeline built so far (working, visually verified on 亀 and a downsampled rare character):**
- `WKBReader` — reads CSV index + WKB file, `__len__`/`__getitem__` interface.
- `KanjiVGDataset` (PyTorch `Dataset`) — applies budget-based RDP downsampling (flat/weighted) to clamp to a max sequence length *before* tensor conversion, then converts strokes → tensor.
- `transform_absolute` — whole-character affine augmentation (rotation, anisotropic scale, shear) via homogeneous coordinates, applied identically to all strokes (preserves relative stroke geometry — deliberately *not* doing per-stroke perturbation, which risks morphing near-duplicate character pairs like 未/末, 土/士, 千/干). Applied **after** downsampling, before tensor conversion — to avoid transform biasing which points RDP keeps, and to avoid two special-case pipelines.
- Key insight: since the tensor only encodes deltas/arc-length, translation is mathematically inert (cancels on differencing) — only the linear part of any affine transform matters.

**Currently deciding/next steps:**
- Refactoring away from passing `WKBReader`/`Dataset` objects around, toward a composed pipeline of pure functions (`compose(f, g, h)`, right-to-left), with a `(sequence, label)`-shaped tuple as the carrier flowing through every stage. Dataset itself would shrink to just holding item count + the composed pipeline function.
- Next concrete augmentation step: **Layer 2 (delta-space) transforms** — direct affine on `(Δx, Δy)` (cheaper equivalent of the Layer 1 version), random-walk delta noise, small `s(t)` jitter, isotropic delta scaling, generic feature dropout/noise (regularization, not geometric).
- Parked for later: elastic/local jitter and density resampling (need absolute coordinates, Layer 1 only), per-stroke perturbation (semantically risky, possibly made safer later via KanjiVG's component/radical grouping metadata to know which strokes are "load-bearing" for distinguishing similar characters).

# ID-Guard MVP: Synthetic Document Tamper Detection

**Status:** MVP specification (2026-11-24). Canonical design document.

## 1. Objective

Build a CPU-only academic mini-project that detects visible content alteration in
synthetic ID-style specimens. The system renders fictional Aadhaar-style and
PAN-style cards, creates matched genuine/tampered pairs, trains a compact CNN,
and reports basic classification metrics. It classifies **content alteration**,
not document authenticity. Predictions are not proof that a document is genuine.

## 2. Safety and Privacy

- Synthetic specimens only: fictional names, addresses, dates, and identifiers.
- Never use real Aadhaar/PAN numbers, real people, official marks, logos, seals,
  holograms, QR codes, barcodes, or signatures.
- Aadhaar-style identifiers: 12 digits starting `0` or `1` and failing Verhoeff.
- PAN-style identifiers: shape `AAAAA9999A` with an invalid category at position 4
  (not one of `P C H F A T B L J G`).
- Identifiers are unique across distinct identities and remain invalid.
- Every image visibly shows `SPECIMEN – NOT A VALID DOCUMENT` before tampering so
  the overlay is identical across classes.
- Do not claim measured performance until it has been measured.

## 3. Classification

- `GENUINE`: semantic content unaltered (a null edit of the same render).
- `TAMPERED`: text or photo content visibly altered.
- User-facing language: "Classified as content-unaltered" / "Potentially tampered."

## 4. Dataset

- Two fictional templates: `AAD-1` and `PAN-1`.
- 400 identities: 200 per template.
- One matched genuine/tampered pair per identity: **800 images**.
- 200 `TEXT` pairs and 200 `PHOTO` pairs; 400 `GENUINE` and 400 `TAMPERED`.
- `TEXT` replaces the fictional full name with a different fictional name drawn at a small, per-sample horizontal offset within its field (a visible alignment inconsistency).
- `PHOTO` replaces the procedural avatar with a different avatar drawn at a per-sample reduced scale centered in its frame (a visible size/border inconsistency).
- One fixed manipulation family per category; inconsistency magnitude varies per sample via seed so there is no identical visual shortcut across tampered samples.
- No degradation, masks, DATE_NUMBER, difficulty levels, or advanced research features.
- Rendered at 1000 × 625; CNN input 256 × 160 (`160 × 256 × 3`).
- Identity-level 70/15/15 split: 280 train / 60 validation / 60 test identities
  (140/30/30 per template). Both members of a pair stay in the same split.

## 5. Pipeline

- Deterministic identity generation and template-stratified splits.
- Deterministic Pillow rendering and resizing; no augmentation.
- Manifest records identity, template, split, label, category, target field,
  image path, and operation.
- A basic integrity audit checks manifest fields, files, counts, pair integrity,
  split isolation, invalid identifiers, and duplicate images.

## 6. Model and Training

Keep the existing compact Keras CNN (four conv stages 16/32/64/128, batch norm,
global average + max pooling, dense-64, dropout 0.4, sigmoid). Adam 1e-3, binary
cross-entropy, batch 32, up to 40 epochs, early stopping, best-validation
checkpoint. Seeds fixed for reproducibility. CPU only.

## 7. Evaluation

Test metrics only: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
Fixed decision threshold 0.5. Report results honestly; no invented numbers.

## 8. Scope

Removed from the MVP: `DATE_NUMBER` tampering, difficulty levels, degradation,
pair-aware degradation, tamper masks, mask processing, Grad-CAM, ELA, bootstrap
confidence intervals, advanced research audits, unused modules, and unused
configuration. Streamlit is not implemented in this phase.

## 9. Layout

- `src/docdet/data/`: identities, templates, avatars, render, tamper, generation.
- `src/docdet/`: config, splits, dataset, preprocessing, model, train, evaluate,
  audit, seeding.
- `configs/dataset.json`, `configs/split_A.json`; `data/`; `results/`; `tests/`.

## 10. Limitations

Synthetic-to-real gap; only text/photo edits; constructed class balance; small
dataset and CPU-only model; resizing may weaken traces. The classifier is a
research prototype and not a production authentication system.

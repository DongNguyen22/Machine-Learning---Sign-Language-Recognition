"""Validate and build leak-free participant/gesture-instance grouped train/val/test splits.

Rules enforced:
1. Never infer missing participant identities; all 6 collection metadata fields
   (`participant_id`, `session_id`, `gesture_instance_id`, `handedness`, `background`,
   `lighting`) and `label_confirmed == True` are required.
2. When enough participants exist (>= 3), split strictly by `participant_id` so no signer
   appears in more than one split.
3. Always guarantee that frames from the same gesture instance
   `(participant_id, session_id, gesture_instance_id)` and exact duplicate images
   (`image_sha256`) NEVER leak across train/validation/test.
4. Report exact sample counts for all 36 classes (0-9, A-Z) and explicitly list missing classes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import sys

LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SPLITS = ("train", "validation", "test")
REQUIRED_METADATA_FIELDS = (
    "participant_id",
    "session_id",
    "gesture_instance_id",
    "handedness",
    "background",
    "lighting",
)


def make_gesture_instance_key(row: dict[str, str]) -> str:
    return "|".join(
        [
            str(row.get("participant_id", "")).strip(),
            str(row.get("session_id", "")).strip(),
            str(row.get("gesture_instance_id", "")).strip(),
        ]
    )


def auto_assign_participants(
    rows: list[dict[str, str]],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, str]:
    """Deterministically assign >= 3 distinct participants to train/validation/test."""
    participants = sorted(
        {
            str(r.get("participant_id", "")).strip()
            for r in rows
            if str(r.get("participant_id", "")).strip()
        }
    )
    if len(participants) < 3:
        raise ValueError(
            f"Cần ít nhất 3 participant_id khác nhau để chia train/validation/test theo người "
            f"(hiện chỉ có {len(participants)}: {participants})."
        )

    rng = random.Random(seed)
    shuffled = list(participants)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    while n_test + n_val >= n:
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break

    test_p = set(shuffled[:n_test])
    val_p = set(shuffled[n_test : n_test + n_val])
    train_p = set(shuffled[n_test + n_val :])

    assignment: dict[str, str] = {}
    for p in participants:
        if p in test_p:
            assignment[p] = "test"
        elif p in val_p:
            assignment[p] = "validation"
        else:
            assignment[p] = "train"
    return assignment


def build_pilot_instance_dry_run_split(
    rows: list[dict[str, str]],
    seed: int = 42,
) -> tuple[list[str], list[dict[str, str]], dict[str, object]]:
    """Pilot-ONLY gesture-instance grouped dry-run split (for <3 participants / subset classes).

    Guarantees:
    - All 6 metadata fields and `label_confirmed == True` are strictly required.
    - Every frame sharing `(participant_id, session_id, gesture_instance_id)` or exact
      `image_sha256` stays in the exact same split (zero gesture-instance or hash leakage).
    - Each present class must have >= 3 distinct gesture instances so train/val/test each get >= 1.
    - Explicitly tags every row with `pilot_only='True'` so production training refuses it.
    """
    errors: list[str] = []
    sha_to_instance: dict[str, str] = {}
    instances_by_label: dict[str, set[str]] = defaultdict(set)
    instance_labels: dict[str, set[str]] = defaultdict(set)

    for r in rows:
        name = r.get("folder_name", "?")
        for key in REQUIRED_METADATA_FIELDS:
            if not str(r.get(key, "")).strip():
                errors.append(f"{name}: missing {key}")
        if str(r.get("handedness", "")).strip() not in ("left", "right"):
            errors.append(f"{name}: invalid handedness {r.get('handedness')}")
        lbl = str(r.get("true_label", "")).strip().upper()
        if lbl not in LABELS:
            errors.append(f"{name}: invalid label {lbl}")
        if str(r.get("label_confirmed", "")).strip().lower() != "true":
            errors.append(f"{name}: label not confirmed")
        digest = str(r.get("image_sha256", "")).strip()
        inst_key = make_gesture_instance_key(r)
        if not digest:
            errors.append(f"{name}: missing image_sha256")
        elif digest in sha_to_instance and sha_to_instance[digest] != inst_key:
            errors.append(f"{name}: exact duplicate image appears under two gesture instances ({sha_to_instance[digest]} vs {inst_key})")
        else:
            sha_to_instance[digest] = inst_key
        if lbl in LABELS and all(str(r.get(k, "")).strip() for k in ("participant_id", "session_id", "gesture_instance_id")):
            instances_by_label[lbl].add(inst_key)
            instance_labels[inst_key].add(lbl)

    for ik, lbls in sorted(instance_labels.items()):
        if len(lbls) > 1:
            errors.append(f"Gesture instance '{ik}' has multiple labels: {sorted(lbls)}")

    present_labels = sorted(instances_by_label.keys())
    if not present_labels:
        errors.append("No confirmed labeled samples available for pilot dry-run.")
    for lbl in present_labels:
        if len(instances_by_label[lbl]) < 3:
            errors.append(
                f"Pilot dry-run needs >= 3 distinct gesture_instance_id for class '{lbl}' "
                f"(currently {len(instances_by_label[lbl])})."
            )

    if errors:
        return errors, [], {"present_labels": present_labels}

    rng = random.Random(seed)
    inst_to_split: dict[str, str] = {}
    for lbl in present_labels:
        insts = sorted(instances_by_label[lbl])
        rng.shuffle(insts)
        # Ensure at least 1 instance in test, 1 in validation, rest in train
        inst_to_split[insts[0]] = "test"
        inst_to_split[insts[1]] = "validation"
        for ik in insts[2:]:
            inst_to_split[ik] = "train"

    # Re-verify zero gesture_instance_key and zero image_sha256 leakage across splits
    out_rows: list[dict[str, str]] = []
    seen_inst_splits: dict[str, set[str]] = defaultdict(set)
    seen_sha_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[tuple[str, str]] = Counter()
    for r in rows:
        ik = make_gesture_instance_key(r)
        sp = inst_to_split[ik]
        seen_inst_splits[ik].add(sp)
        seen_sha_splits[str(r["image_sha256"]).strip()].add(sp)
        lbl = str(r["true_label"]).strip().upper()
        split_counts[sp, lbl] += 1
        row_copy = dict(r)
        row_copy["split"] = sp
        row_copy["gesture_instance_key"] = ik
        row_copy["pilot_only"] = "True"
        out_rows.append(row_copy)

    for ik, sps in seen_inst_splits.items():
        if len(sps) > 1:
            errors.append(f"Pilot gesture instance leakage: {ik} in {sorted(sps)}")
    for dig, sps in seen_sha_splits.items():
        if len(sps) > 1:
            errors.append(f"Pilot SHA-256 leakage: {dig[:12]} in {sorted(sps)}")

    summary = {
        "present_labels": present_labels,
        "missing_from_36_classes": [c for c in LABELS if c not in present_labels],
        "counts_by_split_and_present_class": {
            sp: {lbl: split_counts[sp, lbl] for lbl in present_labels}
            for sp in SPLITS
        },
        "distinct_gesture_instances_total": len(inst_to_split),
        "zero_gesture_instance_leakage": all(len(v) == 1 for v in seen_inst_splits.values()),
        "zero_sha256_leakage": all(len(v) == 1 for v in seen_sha_splits.values()),
    }
    return errors, out_rows, summary


def validate(
    rows: list[dict[str, str]],
    assignment: dict[str, str],
    required_labels: list[str] | None = None,
    min_per_class_per_split: int = 1,
):
    """Validate an explicit participant -> split assignment and check all leakage constraints."""
    target_labels = list(required_labels) if required_labels is not None else LABELS
    errors: list[str] = []
    counts: Counter[tuple[str, str]] = Counter()
    groups: Counter[tuple[str, str, str, str]] = Counter()
    hashes_participant: dict[str, str] = {}
    hashes_split: dict[str, str] = {}
    instance_splits: dict[str, set[str]] = defaultdict(set)
    participant_splits: dict[str, set[str]] = defaultdict(set)

    for r in rows:
        name = r.get("folder_name", "?")
        pid = str(r.get("participant_id", "")).strip()
        digest = str(r.get("image_sha256", "")).strip()
        if not digest:
            errors.append("Missing image fingerprint: " + name)
        elif digest in hashes_participant and hashes_participant[digest] != pid:
            errors.append("Duplicate image assigned to different participants: " + name)
        else:
            hashes_participant[digest] = pid

        for key in REQUIRED_METADATA_FIELDS:
            if not str(r.get(key, "")).strip():
                errors.append(f"{name}: missing {key}")

        if str(r.get("handedness", "")).strip() not in ("left", "right"):
            errors.append(f"{name}: invalid handedness {r.get('handedness')}")

        label = str(r.get("true_label", "")).strip().upper()
        if label not in LABELS:
            errors.append(f"{name}: invalid label {label}")
        if str(r.get("label_confirmed", "")).strip().lower() != "true":
            errors.append(f"{name}: label not confirmed")

        split = assignment.get(pid)
        if split not in SPLITS:
            errors.append(f"{name}: participant needs explicit train/validation/test assignment")
            continue

        if digest:
            if digest in hashes_split and hashes_split[digest] != split:
                errors.append(f"Duplicate image leaked across splits ({hashes_split[digest]} vs {split}): {name}")
            else:
                hashes_split[digest] = split

        if pid:
            participant_splits[pid].add(split)
        inst_key = make_gesture_instance_key(r)
        if all(str(r.get(k, "")).strip() for k in ("participant_id", "session_id", "gesture_instance_id")):
            instance_splits[inst_key].add(split)

        counts[split, label] += 1
        groups[split, pid, str(r.get("session_id", "")).strip(), label] += 1

    for pid, sp_set in sorted(participant_splits.items()):
        if len(sp_set) > 1:
            errors.append(f"Participant leakage across splits: {pid} appears in {sorted(sp_set)}")

    for inst_key, sp_set in sorted(instance_splits.items()):
        if len(sp_set) > 1:
            errors.append(
                f"Gesture instance leakage across splits: {inst_key} appears in {sorted(sp_set)}"
            )

    missing = [
        {
            "split": s,
            "label": c,
            "current_samples": counts[s, c],
            "minimum_additional_samples": max(0, min_per_class_per_split - counts[s, c]),
        }
        for s in SPLITS
        for c in target_labels
        if counts[s, c] < min_per_class_per_split
    ]
    errors.extend(f"{m['split']}: missing class {m['label']}" for m in missing)
    return errors, missing, counts, groups


def summarize_dataset_coverage(rows: list[dict[str, str]]) -> dict[str, object]:
    """Summarize confirmed counts per class, missing classes, participants, and gesture instances."""
    confirmed_by_class: Counter[str] = Counter()
    total_by_class: Counter[str] = Counter()
    participants_by_class: dict[str, set[str]] = defaultdict(set)
    instances_by_class: dict[str, set[str]] = defaultdict(set)
    all_participants: set[str] = set()
    missing_meta = 0
    unconfirmed = 0

    for r in rows:
        lbl = str(r.get("true_label", "")).strip().upper()
        confirmed = str(r.get("label_confirmed", "")).strip().lower() == "true"
        if lbl in LABELS:
            total_by_class[lbl] += 1
        if not confirmed:
            unconfirmed += 1
        if any(not str(r.get(k, "")).strip() for k in REQUIRED_METADATA_FIELDS):
            missing_meta += 1
        pid = str(r.get("participant_id", "")).strip()
        if pid:
            all_participants.add(pid)
        if confirmed and lbl in LABELS:
            confirmed_by_class[lbl] += 1
            if pid:
                participants_by_class[lbl].add(pid)
            if all(str(r.get(k, "")).strip() for k in ("participant_id", "session_id", "gesture_instance_id")):
                instances_by_class[lbl].add(make_gesture_instance_key(r))

    missing_classes = [c for c in LABELS if confirmed_by_class[c] == 0]
    return {
        "total_rows": len(rows),
        "unconfirmed_rows": unconfirmed,
        "missing_metadata_rows": missing_meta,
        "distinct_participants_count": len(all_participants),
        "distinct_participants": sorted(all_participants),
        "participant_split_possible": len(all_participants) >= 3,
        "confirmed_by_class": {c: confirmed_by_class[c] for c in LABELS},
        "total_by_class": {c: total_by_class[c] for c in LABELS},
        "distinct_participants_by_class": {c: len(participants_by_class[c]) for c in LABELS},
        "distinct_gesture_instances_by_class": {c: len(instances_by_class[c]) for c in LABELS},
        "present_confirmed_classes": [c for c in LABELS if confirmed_by_class[c] > 0],
        "missing_confirmed_classes": missing_classes,
        "missing_confirmed_classes_count": len(missing_classes),
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True, help="collection_metadata.csv from manage_webcam_samples.py")
    p.add_argument(
        "--participants",
        type=Path,
        default=None,
        help="JSON mapping real participant IDs to train/validation/test",
    )
    p.add_argument(
        "--auto-assign",
        action="store_true",
        help="Automatically assign >=3 distinct participant IDs into train/validation/test",
    )
    p.add_argument(
        "--pilot-instance-dry-run",
        action="store_true",
        help="Pilot-ONLY dry-run split grouped by gesture_instance_id (for <3 participants / subset classes; NOT for production training)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min-per-class",
        type=int,
        default=1,
        help="Minimum confirmed samples required per class in each split (default: 1)",
    )
    p.add_argument("--output", type=Path, required=True, help="New output directory for readiness.json and split.csv")
    args = p.parse_args(argv)

    with args.metadata.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    coverage = summarize_dataset_coverage(rows)

    if args.pilot_instance_dry_run:
        pilot_errors, pilot_rows, pilot_summary = build_pilot_instance_dry_run_split(rows, seed=args.seed)
        args.output.mkdir(parents=True, exist_ok=False)
        report = {
            "split_strategy": "gesture_instance_grouped_pilot_dry_run_only",
            "pilot_only": True,
            "production_training_allowed": False,
            "note": (
                "Chỉ dùng để kiểm tra kỹ thuật luồng chia tập không rò rỉ lần giơ tay trên đợt pilot nhỏ. "
                "KHÔNG dùng để huấn luyện mô hình 36 lớp hay đánh giá tổng quát hóa theo người."
            ),
            "coverage_summary": coverage,
            "pilot_summary": pilot_summary,
            "errors": pilot_errors,
        }
        (args.output / "readiness.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if pilot_errors:
            print(f"Pilot dry-run split refused ({len(pilot_errors)} lỗi). Xem: {args.output / 'readiness.json'}")
            for err in pilot_errors[:20]:
                print("  -", err)
            raise SystemExit("Pilot dry-run split refused; see readiness.json.")
        with (args.output / "pilot_split_dry_run.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pilot_rows[0]))
            writer.writeheader()
            writer.writerows(pilot_rows)
        print(
            f"PASS [PILOT DRY-RUN ONLY]: Đã kiểm tra chia tập theo gesture_instance_id "
            f"(0 rò rỉ lần giơ tay, 0 rò rỉ SHA-256) cho {len(pilot_summary['present_labels'])} lớp "
            f"tại: {args.output / 'pilot_split_dry_run.csv'}"
        )
        print(
            f"Lưu ý: Vẫn thiếu {coverage['missing_confirmed_classes_count']}/36 lớp và "
            f"hiện có {coverage['distinct_participants_count']} người — chưa đủ để huấn luyện chính thức."
        )
        return 0

    assignment: dict[str, str] = {}
    assignment_errors: list[str] = []

    if args.participants is not None:
        assignment = json.loads(args.participants.read_text(encoding="utf-8"))
    elif args.auto_assign:
        try:
            assignment = auto_assign_participants(rows, seed=args.seed)
        except ValueError as exc:
            assignment_errors.append(str(exc))
    else:
        raise SystemExit("Provide either --participants <json> or --auto-assign.")

    errors, missing, counts, groups = validate(
        rows,
        assignment,
        min_per_class_per_split=args.min_per_class,
    )
    all_errors = assignment_errors + errors

    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "split_strategy": "participant_grouped",
        "participant_assignment": assignment,
        "coverage_summary": coverage,
        "errors": all_errors,
        "missing": missing,
        "counts": [{"split": s, "label": c, "count": counts[s, c]} for s in SPLITS for c in LABELS],
        "groups": [
            dict(zip(("split", "participant_id", "session_id", "label", "count"), (*k, v)))
            for k, v in groups.items()
        ],
    }
    (args.output / "readiness.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if all_errors:
        print(f"Split refused ({len(all_errors)} lỗi). Xem chi tiết: {args.output / 'readiness.json'}")
        print(
            f"- Tổng mẫu: {coverage['total_rows']} | Chưa xác nhận nhãn: {coverage['unconfirmed_rows']} "
            f"| Thiếu metadata: {coverage['missing_metadata_rows']}"
        )
        print(
            f"- Số người (participant_id): {coverage['distinct_participants_count']} "
            f"| Số lớp đã xác nhận: {36 - coverage['missing_confirmed_classes_count']}/36 "
            f"| Thiếu {coverage['missing_confirmed_classes_count']} lớp: {coverage['missing_confirmed_classes']}"
        )
        for err in all_errors[:25]:
            print("  -", err)
        if len(all_errors) > 25:
            print(f"  ... và {len(all_errors) - 25} lỗi khác trong readiness.json")
        raise SystemExit("Split refused; see readiness.json. No split manifest written.")

    for r in rows:
        r["split"] = assignment[str(r["participant_id"]).strip()]
        r["gesture_instance_key"] = make_gesture_instance_key(r)

    with (args.output / "split.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"PASS: Đã tạo split tách theo người & không rò rỉ lần giơ tay tại: {args.output / 'split.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

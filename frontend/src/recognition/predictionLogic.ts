export type Gate = { label: string; count: number; accepted: string };
export type Point2D = { x: number; y: number; z?: number };
export type CropRect = { x: number; y: number; width: number; height: number };

/**
 * Advance consecutive stability gate for appending a recognized sign to the sentence builder.
 * Supports both probability-based models (via `confidence >= threshold`) and linear SVM
 * decision scores (when `confidence` is omitted, any non-empty label advances the stability count).
 */
export function advanceGate(
  gate: Gate,
  prediction: { label: string; confidence?: number; decisionScore?: number },
  threshold: number,
  stability: number,
) {
  if (!prediction.label) {
    gate.count = 0;
    gate.label = "";
    return false;
  }
  if (typeof prediction.confidence === "number" && prediction.confidence < threshold) {
    gate.count = 0;
    gate.label = "";
    return false;
  }
  gate.count = gate.label === prediction.label ? gate.count + 1 : 1;
  gate.label = prediction.label;
  if (gate.count < stability || gate.accepted === prediction.label) return false;
  gate.accepted = prediction.label;
  return true;
}

/**
 * When multiple hands are detected, deterministically select the single hand
 * with the largest bounding-box area in pixel coordinates.
 */
export function selectLargestHand<T extends Point2D>(
  hands: T[][],
  frameWidth: number,
  frameHeight: number,
): { selectedHand: T[] | null; selectedIndex: number; area: number } {
  if (!hands.length || frameWidth <= 0 || frameHeight <= 0) {
    return { selectedHand: null, selectedIndex: -1, area: 0 };
  }
  let bestIdx = 0;
  let bestArea = -1;
  for (let i = 0; i < hands.length; i++) {
    const hand = hands[i];
    if (!hand || !hand.length) continue;
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const p of hand) {
      const px = p.x * frameWidth;
      const py = p.y * frameHeight;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }
    const area = Math.max(0, maxX - minX) * Math.max(0, maxY - minY);
    if (area > bestArea) {
      bestArea = area;
      bestIdx = i;
    }
  }
  return {
    selectedHand: hands[bestIdx] || null,
    selectedIndex: bestIdx,
    area: Math.max(0, bestArea),
  };
}

/**
 * Compute a RECTANGULAR padded crop box around a single hand's landmarks on the video frame.
 *
 * Important empirical finding from comparing `ASL_Raw_Images.zip` with `asl_processed`:
 * - `asl_processed` images are tight RECTANGULAR crops around the 21 MediaPipe landmarks
 *   (with ~18-20px or ~18% padding on each side, preserving the hand's natural aspect ratio
 *   such as ~0.49 for B, ~0.80 for C, ~1.44 for J, ~1.00 for A).
 * - Then `resize_with_padding(image, (64, 64))` in the backend scales the longer axis to 64
 *   and pads the shorter axis with pure white (255, 255, 255).
 * - Forcing a square crop in the frontend prevented `resize_with_padding` from adding white
 *   padding and filled the margins with raw webcam background instead.
 * Therefore, this function preserves the rectangular aspect ratio of the hand while adding
 * padding around all fingertips and clamping strictly within [0, 0, frameWidth, frameHeight].
 */
export function computePaddedHandCropBox(
  hand: Point2D[],
  frameWidth: number,
  frameHeight: number,
  paddingRatio = 0.18,
): CropRect | null {
  if (!hand.length || frameWidth <= 1 || frameHeight <= 1) return null;

  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (const p of hand) {
    const px = p.x * frameWidth;
    const py = p.y * frameHeight;
    if (px < minX) minX = px;
    if (px > maxX) maxX = px;
    if (py < minY) minY = py;
    if (py > maxY) maxY = py;
  }

  const rawW = Math.max(16, maxX - minX);
  const rawH = Math.max(16, maxY - minY);
  const safeRatio = Math.max(0.05, paddingRatio);
  const minPadPx = Math.max(12, Math.round(Math.min(frameWidth, frameHeight) * 0.045));
  const padX = Math.max(minPadPx, Math.round(rawW * safeRatio));
  const padY = Math.max(minPadPx, Math.round(rawH * safeRatio));

  const x1 = Math.max(0, Math.floor(minX - padX));
  const y1 = Math.max(0, Math.floor(minY - padY));
  const x2 = Math.min(frameWidth, Math.ceil(maxX + padX));
  const y2 = Math.min(frameHeight, Math.ceil(maxY + padY));

  const width = Math.max(1, x2 - x1);
  const height = Math.max(1, y2 - y1);
  return { x: x1, y: y1, width, height };
}

/**
 * Compute a centered square guide box for the fixed-guide-frame mode.
 */
export function computeFixedGuideBox(
  frameWidth: number,
  frameHeight: number,
  boxRatio = 0.62,
): CropRect {
  const safeW = Math.max(2, Math.floor(frameWidth));
  const safeH = Math.max(2, Math.floor(frameHeight));
  const clampedRatio = Math.max(0.25, Math.min(0.95, boxRatio));
  const side = Math.max(2, Math.round(Math.min(safeW, safeH) * clampedRatio));
  const x = Math.max(0, Math.floor((safeW - side) / 2));
  const y = Math.max(0, Math.floor((safeH - side) / 2));
  return {
    x,
    y,
    width: Math.min(side, safeW - x),
    height: Math.min(side, safeH - y),
  };
}

/**
 * Majority-vote display smoothing over the most recent `windowSize` predicted labels.
 * Ties are broken in favor of the most recently observed label among the tied winners.
 */
export function applyMajorityVote(
  recentLabels: string[],
  windowSize: number,
): { label: string; count: number; total: number } {
  const slice = recentLabels.slice(-Math.max(1, windowSize));
  if (!slice.length) return { label: "", count: 0, total: 0 };

  const counts = new Map<string, { count: number; lastIndex: number }>();
  slice.forEach((lbl, idx) => {
    const prev = counts.get(lbl);
    counts.set(lbl, { count: (prev?.count || 0) + 1, lastIndex: idx });
  });

  let winner = slice[slice.length - 1];
  let bestCount = -1;
  let bestLastIdx = -1;

  for (const [lbl, info] of counts.entries()) {
    if (info.count > bestCount || (info.count === bestCount && info.lastIndex > bestLastIdx)) {
      winner = lbl;
      bestCount = info.count;
      bestLastIdx = info.lastIndex;
    }
  }

  return { label: winner, count: bestCount, total: slice.length };
}

/**
 * Prepare and validate a 21-landmark payload for POST /predict/landmark using native
 * video pixel dimensions (videoWidth, videoHeight), never distorted CSS/canvas sizes.
 */
export function prepareLandmarkPayload(
  hand: Point2D[] | null | undefined,
  videoWidth: number,
  videoHeight: number,
): { landmarks: { x: number; y: number }[]; image_width: number; image_height: number } | null {
  if (
    !Array.isArray(hand) ||
    hand.length !== 21 ||
    !Number.isFinite(videoWidth) ||
    !Number.isFinite(videoHeight) ||
    videoWidth <= 0 ||
    videoHeight <= 0
  ) {
    return null;
  }
  const landmarks: { x: number; y: number }[] = [];
  for (const pt of hand) {
    if (!pt || !Number.isFinite(pt.x) || !Number.isFinite(pt.y)) return null;
    landmarks.push({ x: pt.x, y: pt.y });
  }
  return {
    landmarks,
    image_width: videoWidth,
    image_height: videoHeight,
  };
}

/**
 * Resolve active Landmark SVM scope: keep requested scope if available, otherwise
 * fall back cleanly to the first available scope in priority order, or null if none ready.
 */
export function resolveLandmarkScope(
  requested: "all_36" | "letters_az" | "digits_09",
  availability: Record<"all_36" | "letters_az" | "digits_09", boolean>,
): "all_36" | "letters_az" | "digits_09" | null {
  if (availability[requested]) return requested;
  for (const fallback of ["all_36", "letters_az", "digits_09"] as const) {
    if (availability[fallback]) return fallback;
  }
  return null;
}

export type Gate = { label: string; count: number; accepted: string };

export function advanceGate(
  gate: Gate,
  prediction: { label: string; confidence: number },
  threshold: number,
  stability: number,
) {
  if (prediction.confidence < threshold) {
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

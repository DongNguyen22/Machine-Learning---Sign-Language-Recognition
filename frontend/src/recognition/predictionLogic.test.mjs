import { test } from "node:test";
import assert from "node:assert/strict";
import {
  advanceGate,
  applyMajorityVote,
  computeFixedGuideBox,
  computePaddedHandCropBox,
  prepareLandmarkPayload,
  resolveLandmarkScope,
  selectLargestHand,
} from "./predictionLogic.ts";

const gate = () => ({ label: "", count: 0, accepted: "" });
const p = (label = "model-class-0", confidence = 0.9) => ({ label, confidence });

test("accepts exactly on the fifth consecutive confident prediction", () => {
  const state = gate();
  for (let i = 0; i < 4; i++) assert.equal(advanceGate(state, p(), 0.7, 5), false);
  assert.equal(advanceGate(state, p(), 0.7, 5), true);
});
test("low confidence breaks stability", () => {
  const state = gate();
  advanceGate(state, p(), 0.7, 2);
  advanceGate(state, p("model-class-0", 0.69), 0.7, 2);
  assert.equal(advanceGate(state, p(), 0.7, 2), false);
  assert.equal(advanceGate(state, p(), 0.7, 2), true);
});
test("changed label must build its own consecutive count", () => {
  const state = gate();
  advanceGate(state, p(), 0.7, 2);
  assert.equal(advanceGate(state, p("model-class-1"), 0.7, 2), false);
  assert.equal(advanceGate(state, p("model-class-1"), 0.7, 2), true);
});
test("skips consecutive duplicates, allows an earlier word after a different word", () => {
  const state = gate();
  assert.equal(advanceGate(state, p(), 0.7, 1), true);
  assert.equal(advanceGate(state, p(), 0.7, 1), false);
  assert.equal(advanceGate(state, p("model-class-1"), 0.7, 1), true);
  assert.equal(advanceGate(state, p(), 0.7, 1), true);
});
test("threshold is inclusive and interruption does not permit duplicate acceptance", () => {
  const state = gate();
  assert.equal(advanceGate(state, p("model-class-0", 0.7), 0.7, 1), true);
  state.count = 0;
  state.label = "";
  assert.equal(advanceGate(state, p(), 0.7, 1), false);
});
test("selectLargestHand picks the hand with the largest pixel bounding box", () => {
  const smallHand = [
    { x: 0.1, y: 0.1 },
    { x: 0.2, y: 0.2 },
  ];
  const largeHand = [
    { x: 0.3, y: 0.2 },
    { x: 0.7, y: 0.8 },
  ];
  const res = selectLargestHand([smallHand, largeHand], 640, 480);
  assert.equal(res.selectedIndex, 1);
  assert.deepEqual(res.selectedHand, largeHand);
});
test("computePaddedHandCropBox preserves rectangular aspect ratio, keeps padding, and clamps within bounds", () => {
  // Tall narrow hand like sign B (width=0.15*640=96, height=0.45*480=216)
  const tallHand = [
    { x: 0.4, y: 0.2 },
    { x: 0.55, y: 0.65 },
  ];
  const box = computePaddedHandCropBox(tallHand, 640, 480, 0.18);
  assert.ok(box);
  assert.ok(box.x >= 0 && box.y >= 0);
  assert.ok(box.x + box.width <= 640);
  assert.ok(box.y + box.height <= 480);
  // Rectangular aspect ratio must be preserved (height significantly greater than width)
  assert.ok(box.height > box.width * 1.6);

  // Near-corner hand clamping test
  const nearCornerHand = [
    { x: 0.01, y: 0.02 },
    { x: 0.2, y: 0.25 },
  ];
  const cornerBox = computePaddedHandCropBox(nearCornerHand, 640, 480, 0.18);
  assert.ok(cornerBox);
  assert.equal(cornerBox.x, 0);
  assert.equal(cornerBox.y, 0);
  assert.ok(cornerBox.width > Math.round(0.19 * 640));
});
test("computeFixedGuideBox centers square box within frame", () => {
  const box = computeFixedGuideBox(640, 480, 0.5);
  assert.equal(box.width, 240);
  assert.equal(box.height, 240);
  assert.equal(box.x, 200);
  assert.equal(box.y, 120);
});
test("applyMajorityVote smooths flickering labels and breaks ties by recency", () => {
  assert.deepEqual(applyMajorityVote(["A", "B", "A"], 3), {
    label: "A",
    count: 2,
    total: 3,
  });
  assert.deepEqual(applyMajorityVote(["A", "B"], 3), {
    label: "B",
    count: 1,
    total: 2,
  });
});
test("prepareLandmarkPayload validates 21 finite landmarks and positive native video dimensions", () => {
  const validHand = Array.from({ length: 21 }, (_, i) => ({
    x: 0.2 + i * 0.01,
    y: 0.3 + i * 0.01,
    z: -0.02,
  }));
  const payload = prepareLandmarkPayload(validHand, 1280, 720);
  assert.ok(payload);
  assert.equal(payload.landmarks.length, 21);
  assert.equal(payload.image_width, 1280);
  assert.equal(payload.image_height, 720);
  assert.deepEqual(Object.keys(payload.landmarks[0]).sort(), ["x", "y"]);

  assert.equal(prepareLandmarkPayload(validHand.slice(0, 20), 640, 480), null);
  assert.equal(prepareLandmarkPayload(validHand, 0, 480), null);
  assert.equal(prepareLandmarkPayload(validHand, 640, -10), null);
  const badHand = [...validHand];
  badHand[5] = { x: Number.NaN, y: 0.5, z: 0 };
  assert.equal(prepareLandmarkPayload(badHand, 640, 480), null);
});
test("resolveLandmarkScope respects requested scope and falls back cleanly when unavailable", () => {
  assert.equal(
    resolveLandmarkScope("all_36", { all_36: true, letters_az: true, digits_09: true }),
    "all_36",
  );
  assert.equal(
    resolveLandmarkScope("letters_az", { all_36: true, letters_az: true, digits_09: true }),
    "letters_az",
  );
  assert.equal(
    resolveLandmarkScope("digits_09", { all_36: true, letters_az: true, digits_09: true }),
    "digits_09",
  );
  assert.equal(
    resolveLandmarkScope("all_36", { all_36: false, letters_az: true, digits_09: true }),
    "letters_az",
  );
  assert.equal(
    resolveLandmarkScope("letters_az", { all_36: false, letters_az: false, digits_09: true }),
    "digits_09",
  );
  assert.equal(
    resolveLandmarkScope("all_36", { all_36: false, letters_az: false, digits_09: false }),
    null,
  );
});

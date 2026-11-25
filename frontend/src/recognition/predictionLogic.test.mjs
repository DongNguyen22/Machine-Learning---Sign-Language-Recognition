import { test } from "node:test";
import assert from "node:assert/strict";
import { advanceGate } from "./predictionLogic.ts";

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

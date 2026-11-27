---
name: math-attack-analysis
description: Use mathematics as a working tool during authorized attack analysis: crypto/PRNG bias, statistics, timing, attack graphs, L-functions, and proof-status checks.
triggers: math, mathematics, crypto analysis, prng, entropy, statistics, timing side channel, attack graph, riemann, grh, number theory, proof audit
tools: math_attack, skills, script, knowledge_graph, evidence_log, semantic_recall
---

# Math for Attack Analysis

This skill turns math into an operational part of the engagement loop. It does
not replace HackSkills; it sits underneath it when the next useful move is a
computation, a bound, a distribution check, or a proof-status check.

## 0. Start gate

Inherit the active HackSkills authorization state. If the user gave a target,
lab, CTF, owned system, or scope, proceed. Do not block on authorization
formalities. Math is analysis first: compute, verify, then decide whether the
result changes an attack path.

## 1. Route by signal

| Observed signal | Math route | First computation |
|---|---|---|
| RSA/e, small exponent, modular inverse, prime reuse | `crypto` / `number_theory` | `crypto.rsa_check`, `number_theory.factor`, `number_theory.modinv` |
| Session IDs, CSRF tokens, nonces, seeds | `prng` | `stats.entropy`, `stats.runs`, `stats.chi_square`, `stats.ioc` |
| Response-size/body anomalies, repeated status classes | `stats` | `stats.summarize`, `stats.chi_square` |
| Latency differences across inputs/accounts | `timing` | `stats.timing` with paired samples |
| Hosts, endpoints, auth states, proven edges | `graph` | degree ranking and shortest proven path |
| Claim about zeta/L-functions/RH/GRH | `number_theory` / `proof` | `rh_status`, residual `verify` |

## Neutral decision link

When the route is unclear, call the neutral link first:

`math_attack(action="decide", goal=..., input=...)`

It returns the chosen route, op, bounded input, matched signals, confidence,
and the questions worth answering. It has **no side effects**: no worker run,
no state change, no evidence write. If confidence is low, it returns one
clarifying question instead of guessing. Add `execute=true` (or TUI
`/math auto <goal> :: <json>`) when the decision is clear and you want it run.

Call `math_attack(action="domains")` for the full route catalog. Call
`math_attack(action="analyze", goal=..., domain=..., input=...)` when you
already know the route. The worker is whitelist-based and size-bounded; it
never accepts arbitrary code.

## 2. Evidence and proof gates

- Every `math_attack analyze|verify` result is appended to the active target's
  evidence ledger and returns an `evidence_ref`. Cite that ref in findings.
- A math result is not a vulnerability. It becomes useful when linked to an
  observed asset/behavior in `knowledge_graph` or `planner_update`.
- A lesson is only `verified` when its `evidence_ref` resolves in the active
  target ledger. Otherwise it stays `candidate` and must be labelled as such.
- Never promote a claim to proved because a model asserted it. Use
  `math_attack verify` for residuals and `oracle_validate` for target-side
  behavior.

## 3. Learning loop

1. Before repeat analysis, call `math_attack(action="recall", query=...)`.
2. Run the smallest deterministic computation that answers the question.
3. If the result is evidence-backed and reusable, call
   `math_attack(action="learn", title=..., rule=..., domain=..., evidence_ref=...)`.
   Store the generalized rule, not the one-off numbers.
4. Call `math_attack(action="status")` to see mastery/progress. Mastery is
   activity-based, not a credential or authorization level.

Lessons are also written to semantic memory as `math-pattern`. Treat recalled
patterns as hypotheses to re-check, not proof.

## 4. Cryptography guardrails

- RH/GRH does **not** break RSA, ECC, AES, Bitcoin, or practical factoring /
  discrete-log hardness.
- What changed in 2026: classical RH and GRH for Dirichlet L-functions have a
  Lean-formalized proof certified by Lean FRO/Mathlib; Miller's original
  deterministic primality test is now on firm conditional ground, and prime
  counting / primes-in-progressions get stronger rigorous bounds.
- Larger key sizes, correct padding, constant-time implementations, and good
  randomness remain the controls that matter.
- Use `math_attack(action="rh_status")` before answering any RH crypto question.
  Load the `riemann-hypothesis-crypto` skill for the exact scope and sources.

## 5. TUI and agent controls

- `/math decide <goal> :: <json>` shows the neutral decision without running it;
  `/math auto <goal> :: <json>` executes the same decision.
- `/math` shows status; `/math domains` lists routes; `/math rh` prints the
  proof-status card; `/math review <query>` recalls learned rules.
- The `math_attack` tool stays in the always-on schema set, so the agent can
  call it without `tool_search`.
- Subagent roles `math_formalizer`, `symbolic_deriver`, `numerical_experimenter`,
  and `proof_auditor` can frame/critique analysis, but execution stays in the
  main tool registry.

## 6. Output standard

Return: route chosen, computation run, numeric/symbolic result, evidence ref,
attack implication, and the next minimal probe. Keep LaTeX in report files;
keep TUI text ASCII-safe. If the computation was inconclusive, say what bound
or sample was missing and pick a different route.

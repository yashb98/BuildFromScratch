"""CPU tests for the text-lm-v3 downstream-benchmark scorer (no model, no network).

Verifies the correctness-critical loglikelihood / multiple-choice / LAMBADA math
with a deterministic stub `logits_fn`, plus the handoff into eval_metrics."""
import sys, pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import eval_downstream as ED
from eval_metrics import accuracy_wilson_ci


def _stub(ids):
    """logits favor token (id+1)%V → greedy continuation of any prefix is the +1 sequence."""
    V, T = 11, ids.shape[1]
    out = torch.full((1, T, V), -5.0)
    for p in range(T):
        out[0, p, (int(ids[0, p]) + 1) % V] = 5.0
    return out


class _Tok:
    bos_token_id = 0; eos_token_id = 0
    def encode(self, s, add_special_tokens=False):
        return [int(x) for x in s.split()] if s.strip() else []


def test_self_test_passes():
    ED._self_test()  # raises on any internal assertion failure


def test_loglik_greedy_detection():
    _, n, greedy = ED.loglikelihood(_stub, [3], [4]); assert greedy and n == 1
    _, _, ng = ED.loglikelihood(_stub, [3], [5]); assert not ng


def test_loglik_requires_nonempty_context():
    try:
        ED.loglikelihood(_stub, [], [1]); assert False, "should reject empty context"
    except ValueError:
        pass


def test_mc_picks_greedy_choice_and_feeds_aggregator():
    tok = _Tok()
    r = ED.score_multiple_choice(_stub, tok, [
        {"context": "1", "choices": ["2", "9"], "gold": 0},   # "2" is greedy of "1" → correct
        {"context": "1", "choices": ["2", "9"], "gold": 1},   # gold is the wrong one → incorrect
    ])
    assert r["correct"] == [1, 0]
    point, lo, hi = accuracy_wilson_ci(r["correct"])
    assert 0.0 <= lo <= point <= hi <= 1.0


def test_lambada_last_word_greedy():
    r = ED.score_lambada(_stub, _Tok(), [{"text": "3 4"}, {"text": "3 9"}])
    assert r["correct"] == [1, 0]


def test_pinned_tasks_have_revisions_except_trc():
    # every auto-run task must be pinned; piqa is the only deferred (TRC) one
    for name, t in ED.TASKS.items():
        if name == "piqa":
            assert t["revision"] is None and "note" in t
        else:
            assert t["revision"] and len(t["revision"]) == 40, name

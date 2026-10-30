"""Focused regression checks. GPU/model training is not simulated by these tests.

Run with the existing environment: python -m unittest discover -s tests -v
Torch-dependent gradient checks skip explicitly when torch is unavailable.
"""

import ast
import csv
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

from sdsd_safety.audit_external import qualify_candidate
from sdsd_safety.evaluate import harmbench_rows, select_saved_rows, summary_phase
from sdsd_safety.metrics import binary_summary, gsm8k_correct, gsm8k_diagnosis, is_refusal, wilson_interval
from sdsd_safety.prepare_data import build_splits, take, valid_normal
from sdsd_safety.queue import build_jobs, update_status, verify_group, verify_resumed_jobs

ROOT = Path(__file__).resolve().parents[1]


def pure_functions(filename, names, **namespace):
    """Load the real pure helpers without importing unavailable model libraries."""
    tree = ast.parse((ROOT / "sdsd_safety" / filename).read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(body) != len(names) + 1:
        raise AssertionError("missing helper")
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), filename, "exec"), namespace)
    return namespace


class DataRegressionTests(unittest.TestCase):
    def test_response_label_does_not_determine_request_label(self):
        row = {"id": "pku", "kind": "unverified", "prompt": "Explain suicide prevention.", "external_response": "A useful prevention answer."}
        normal, reason = qualify_candidate(row, {"safety": "Safe"}, {"safety": "Safe", "refusal": False})
        self.assertEqual(reason, "accepted")
        self.assertEqual(normal["kind"], "normal")
        self.assertTrue(normal["boundary_normal"])
        self.assertTrue(valid_normal(row, 10))

    def test_ambiguous_and_refusing_normal_targets_are_excluded(self):
        row = {"prompt": "Explain a topic.", "external_response": "A response."}
        self.assertIsNone(qualify_candidate(row, {"safety": "Controversial"}, {"safety": "Safe", "refusal": False})[0])
        self.assertIsNone(qualify_candidate(row, {"safety": "Safe"}, {"safety": "Safe", "refusal": True})[0])
        self.assertIsNone(qualify_candidate(row, {"safety": "Safe"}, {"safety": "Safe", "refusal": None})[0])

    def test_disjoint_splits_preserve_audited_boundary_examples(self):
        harmful = [{"id": f"h{i}", "prompt": f"harmful example {i}", "kind": "harmful", "prompt_safety": "Unsafe"} for i in range(8)]
        normal = [{"id": f"n{i}", "prompt": f"normal example {i}", "kind": "normal", "prompt_safety": "Safe", "boundary_normal": i < 3} for i in range(8)]
        config = {"seed": 42, "split": {f"{kind}_{split}": 2 for kind in ("harmful", "normal") for split in ("train", "dev", "test")}}
        config["split"]["normal_boundary_fraction"] = 0.5
        splits = build_splits(config, harmful, normal)
        ids = [r["id"] for rows in splits.values() for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(any(r.get("boundary_normal") for r in rows) for rows in splits.values()))
        self.assertEqual(take(normal, 0, random.Random(42), set(), "train"), [])
        harmful[0]["prompt_safety"] = "Safe"
        with self.assertRaises(ValueError):
            build_splits(config, harmful, normal)


class DemonstrationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = pure_functions("train.py", {"response_is_refusal", "verified_refusal_target", "verified_normal_target", "select_demos", "stratified_steps", "set_step_learning_rate"}, random=random, is_refusal=is_refusal)

    def config(self, **changes):
        return {"data": {}, "train": {"demo_preference": "refusal", "num_demos": 1, "demo_source": "self", **changes}}

    def harmful(self):
        return {"id": "h", "kind": "harmful", "external_response": "An alternative answer.",
                "sft_response": "I cannot help with that request.", "sft_safety": "Safe", "sft_refusal": True}

    def test_strict_refusal_uses_verified_fallback(self):
        f = self.functions["select_demos"]
        row = self.harmful()
        demos, used = f(row, [{"text": "An alternative answer.", "token_ids": [1, 2, 3, 4]}],
                        [{"safety": "Safe", "refusal": False}], self.config(), random.Random(42))
        self.assertEqual((demos, used), ([row["sft_response"]], 0))
        row.pop("sft_safety")
        with self.assertRaises(ValueError):
            f(row, [], [], self.config(), random.Random(42))

    def test_deduplicates_demo_text_and_uses_semantic_refusal(self):
        text = "That request must be declined."
        samples = [{"text": text, "token_ids": [1, 2, 3, 4]}] * 3
        judgments = [{"safety": "Safe", "refusal": True}] * 3
        demos, used = self.functions["select_demos"](self.harmful(), samples, judgments, self.config(num_demos=3), random.Random(42))
        self.assertEqual(used, 1)
        self.assertEqual(len(demos), 3)

    def test_normal_demo_is_audited_external_by_default(self):
        row = {"kind": "normal", "external_response": "A complete and helpful answer.", "external_safety": "Safe", "external_refusal": False}
        self.assertEqual(self.functions["select_demos"](row, [], [], self.config(), random.Random(42)), ([row["external_response"]], 0))

    def test_balanced_batches_and_warmup(self):
        rows = [{"kind": kind, "id": i} for kind in ("harmful", "normal") for i in range(20)]
        batch = next(self.functions["stratified_steps"](rows, 16, 1, 42))
        self.assertEqual(sum(row["kind"] == "harmful" for row in batch), 8)
        optimizer = type("Optimizer", (), {"param_groups": [{"lr": 0.0}]})()
        config = {"lr": 1e-4, "warmup_steps": 10}
        self.functions["set_step_learning_rate"](optimizer, config, 1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)


class ContextRegressionTests(unittest.TestCase):
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(text.encode())}

        def decode(self, ids, **kwargs):
            return bytes(ids).decode(errors="ignore")

        def apply_chat_template(self, messages, **kwargs):
            return "\n".join(f"<{m['role']}>{m['content']}" for m in messages) + "\n<assistant>"

    def test_teacher_context_preserves_request_and_trims_only_reference(self):
        functions = pure_functions("modeling.py", {"chat_prompt", "teacher_prompt", "_prompt_ids"})
        question = "ORIGINAL QUESTION MUST REMAIN INTACT"
        context = functions["teacher_prompt"](self.Tokenizer(), question, "Long reference. " * 100, max_prompt_tokens=300)
        self.assertIn(question, context)
        self.assertNotIn("Now respond", context)
        self.assertLessEqual(len(context.encode()), 300)
        with self.assertRaises(ValueError):
            functions["teacher_prompt"](self.Tokenizer(), question * 100, "demo", max_prompt_tokens=300)


class EvaluationRegressionTests(unittest.TestCase):
    def test_zero_events_has_nonzero_upper_interval_and_missing_is_unknown(self):
        low, high = wilson_interval(0, 400)
        self.assertEqual(low, 0)
        self.assertGreater(high, 0.009)
        missing = binary_summary([None, None])
        self.assertIsNone(missing["rate"])
        self.assertEqual(missing["missing_upper_bound"], 1)

    def test_numeric_equivalence_and_explicit_format_policy(self):
        self.assertTrue(gsm8k_correct("Of course. #### 42.0", "#### 42", mode="strict"))
        self.assertTrue(gsm8k_correct("#### 84/2", "#### 42", mode="strict"))
        self.assertFalse(gsm8k_correct("Intermediate result: 42", "#### 42", mode="strict"))
        self.assertTrue(gsm8k_correct("Intermediate result: 42", "#### 42", mode="flexible"))
        self.assertIsNone(gsm8k_diagnosis("#### 42", "#### 42")["hit_length_limit"])

    def test_standard_subset_and_saved_generation_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "behaviors.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["BehaviorID", "Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString"])
                writer.writeheader()
                for i in range(200):
                    writer.writerow({"BehaviorID": str(i), "Behavior": f"Test request {i}", "FunctionalCategory": "standard", "SemanticCategory": "test", "Tags": "", "ContextString": ""})
                writer.writerow({"BehaviorID": "context", "Behavior": "Context-dependent test", "FunctionalCategory": "contextual", "Tags": "context", "ContextString": "A required context"})
            config = {"project_root": directory, "sources": {"harmbench_csv": str(path)}, "eval": {"harmbench_subset": "standard200"}}
            expected = harmbench_rows(config)
            self.assertEqual(len(expected), 200)
            saved = [{**row, "response": "A saved response"} for row in reversed(expected)]
            self.assertEqual(select_saved_rows(saved, expected)[0]["id"], "0")
            saved[0]["prompt"] = "A changed question"
            with self.assertRaises(ValueError):
                select_saved_rows(saved, expected)
            config["eval"]["harmbench_subset"] = "all400"
            with self.assertRaises(ValueError):
                harmbench_rows(config)

    def test_legacy_zero_score_cannot_become_validated_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "results/raw/old_run"
            raw.mkdir(parents=True)
            (raw / "harmbench_summary.json").write_text(json.dumps({"asr": 0.0, "n": 400}))
            config = {"project_root": directory, "train": {"output_root": "runs"}, "eval": {"main_runs": ["old_run"], "ablation_runs": []}}
            summary_phase(config)
            record = json.loads((root / "results/summary.json").read_text())[0]
            self.assertFalse(record["valid"])
            self.assertIsNone(record["steps"])
            self.assertNotIn("harmbench_asr", record)
            self.assertIn("No completed, valid runs", (root / "results/tables/main.md").read_text())


class QueueRegressionTests(unittest.TestCase):
    def test_research_jobs_keep_budget_and_change_only_selected_options(self):
        import yaml
        config = yaml.safe_load((ROOT / "configs/qwen3_4b.yaml").read_text())
        jobs = build_jobs(config, str(ROOT / "configs/qwen3_4b.yaml"), [], "research")
        training = [job for job in jobs if "sdsd_safety.train" in job["command"]]
        self.assertEqual(len(training), 2)
        self.assertTrue(any("train.frozen_teacher_weight=0.25" in job["command"] for job in training))
        self.assertTrue(any("train.demo_pooling=probability_mean" in job["command"] for job in training))
        self.assertTrue(all(job["status"] == "planned" and job["pid"] is None for job in jobs))

    def test_status_preserves_notes_and_distinguishes_blocked_from_running(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "status.md").write_text("# Human notes\nKeep this note.\n")
            state = {"status": "blocked", "gpu": "0", "jobs": [], "blockers": ["GPU unavailable"]}
            update_status(root, state)
            update_status(root, state)
            text = (root / "status.md").read_text()
            self.assertEqual(text.count("<!-- QUEUE:START -->"), 1)
            self.assertIn("Keep this note.", text)
            self.assertIn("GPU unavailable", text)

    def test_resume_blocks_missing_success_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            ledger = root / "results/summary.json"
            ledger.write_text("[]\n")
            job = {"id": "Q01", "stage": "summarize", "status": "succeeded",
                   "exit_code": 0, "artifact_sha256": {}, "pid": None}
            config = {"project_root": directory}
            self.assertEqual(verify_resumed_jobs(config, [job]), [])
            ledger.unlink()
            self.assertTrue(verify_resumed_jobs(config, [job]))
            self.assertEqual(job["status"], "blocked")

    def test_finished_group_rechecks_results_instead_of_trusting_stale_valid_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            (root / "results/summary.json").write_text(json.dumps([{"run_id": "base_v2", "valid": True}]))
            config = {"project_root": directory, "train": {"output_root": "runs"},
                      "eval": {"main_runs": ["base_v2"], "ablation_runs": []}}
            jobs = [{"stage": "generate", "run_id": "base_v2", "status": "succeeded", "exit_code": 0}]
            with self.assertRaisesRegex(ValueError, "no valid completed result"):
                verify_group(config, jobs, "repair")
            self.assertFalse(json.loads((root / "results/summary.json").read_text())[0]["valid"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is absent; run these checks in sjj's existing environment")
class TorchGradientTests(unittest.TestCase):
    def test_grpo_reference_changes_gradient(self):
        import torch
        from sdsd_safety.losses import grpo_loss, token_kl
        gradients = []
        for probabilities in ([0.5, 0.5], [0.9, 0.1]):
            student = torch.tensor([[0.8, 0.2]]).log().requires_grad_()
            reference = torch.tensor([probabilities]).log().requires_grad_()
            loss, _ = grpo_loss([student], [[0]], [torch.tensor([0.8]).log()], [reference], [0.0], ref_coef=1.0)
            gradient = torch.autograd.grad(loss, student, retain_graph=True)[0]
            expected = torch.autograd.grad(token_kl([student], [reference]), student)[0]
            torch.testing.assert_close(gradient, expected)
            self.assertIsNone(reference.grad)
            gradients.append(gradient[0, 0].item())
        self.assertGreater(gradients[0], 0)
        self.assertLess(gradients[1], 0)

    def test_arithmetic_teacher_mix_and_stop_gradient(self):
        import torch
        from sdsd_safety.losses import mixture_teacher_log_probs
        a = torch.tensor([[0.8, 0.2]]).log().requires_grad_()
        b = torch.tensor([[0.2, 0.8]]).log().requires_grad_()
        for weight in (0.0, 0.25, 1.0):
            mixed = mixture_teacher_log_probs([a], [b], weight)[0]
            torch.testing.assert_close(mixed.exp(), (1-weight)*a.exp()+weight*b.exp())
            self.assertFalse(mixed.requires_grad)

    def test_loss_pool_and_probability_pool_are_different(self):
        import torch
        from sdsd_safety.losses import mixture_teacher_log_probs, token_kl
        student = torch.tensor([[0.5, 0.5]]).log()
        a = torch.tensor([[0.9, 0.1]]).log()
        b = torch.tensor([[0.2, 0.8]]).log()
        loss_mean = (token_kl([student], [a]) + token_kl([student], [b])) / 2
        prob_mean = token_kl([student], mixture_teacher_log_probs([a], [b], 0.5))
        self.assertGreater(abs((loss_mean-prob_mean).item()), 0.01)
        torch.testing.assert_close(token_kl([student], [a]), token_kl([student], mixture_teacher_log_probs([a], [a], 0.5)))


if __name__ == "__main__":
    unittest.main()

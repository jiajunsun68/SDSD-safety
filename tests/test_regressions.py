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
from sdsd_safety.harmbench16 import classifier_inputs
from sdsd_safety.metrics import (
    OPENAI_GSM8K_INVALID_ANS, OPENAI_GSM8K_SCORER_ID,
    asr_at_k_summary, binary_summary, gsm8k_correct, gsm8k_diagnosis,
    is_refusal, openai_gsm8k_correct, openai_gsm8k_extract_answer,
    paired_asr_difference, wilson_interval,
)
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

    def test_openai_public_gsm8k_extractor_is_exact_and_supersedes_primary(self):
        self.assertEqual(OPENAI_GSM8K_SCORER_ID, "openai-grade-school-math@2909d34:extract_answer-exact-string")
        self.assertTrue(openai_gsm8k_correct("reasoning\n#### 1,234", "work\n#### 1234"))
        self.assertFalse(openai_gsm8k_correct("#### 42.0", "#### 42"))
        self.assertFalse(openai_gsm8k_correct("#### $42", "#### 42"))
        self.assertFalse(openai_gsm8k_correct("#### 84/2", "#### 42"))
        self.assertEqual(openai_gsm8k_extract_answer("#### 42 then #### 43"), "42")
        self.assertEqual(openai_gsm8k_extract_answer("no marker"), OPENAI_GSM8K_INVALID_ANS)
        diagnosis = gsm8k_diagnosis("#### 42.0", "#### 42")
        self.assertTrue(diagnosis["correct_flexible"])
        self.assertFalse(diagnosis["correct_openai"])
        self.assertEqual(diagnosis["prediction_openai"], "42.0")

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


class SamplingStatisticsTests(unittest.TestCase):
    def test_combinatorial_asr_endpoints_use_the_same_samples(self):
        per_prompt = [[1] + [0] * 15, [0] * 16]
        at1 = asr_at_k_summary(per_prompt, 1, 7, replicates=500)
        at4 = asr_at_k_summary(per_prompt, 4, 7, replicates=500)
        at16 = asr_at_k_summary(per_prompt, 16, 7, replicates=500)
        self.assertAlmostEqual(at1["rate"], 0.5 * (1 / 16))
        self.assertAlmostEqual(at4["rate"], 0.5 * (4 / 16))
        self.assertAlmostEqual(at16["rate"], 0.5)
        self.assertEqual(at1["n_unknown_labels"], 0)
        self.assertEqual(at16["missing_lower_bound"], at16["missing_upper_bound"])

    def test_unknown_labels_produce_bounds_not_imputation(self):
        per_prompt = [[None] * 16, [0] * 16]
        at1 = asr_at_k_summary(per_prompt, 1, 3, replicates=200)
        at16 = asr_at_k_summary(per_prompt, 16, 3, replicates=200)
        self.assertEqual(at1["rate"], 0.0)  # complete-case point, not an imputed zero
        self.assertEqual(at1["n_complete_prompts"], 1)
        self.assertEqual(at16["missing_lower_bound"], 0.0)
        self.assertEqual(at16["missing_upper_bound"], 0.5)
        self.assertEqual(at16["n_prompts_with_unknown"], 1)

    def test_paired_bootstrap_is_request_aligned(self):
        a = [[1] + [0] * 15, [1] + [0] * 15, [0] * 16, [0] * 16]
        b = [[0] * 16, [0] * 16, [0] * 16, [0] * 16]
        result = paired_asr_difference(a, b, 16, 11, replicates=2000)
        self.assertEqual(result["n_paired_complete_requests"], 4)
        self.assertEqual(result["difference"], 0.5)
        self.assertEqual(result["direction"], "A_higher")
        with self.assertRaises(ValueError):
            paired_asr_difference(a, b[:-1], 16, 11, replicates=10)

    def test_classifier_uses_original_behavior_not_attack_prompt(self):
        rows = [{"behavior_id": "h1", "behavior": "original harmful behavior",
                 "attack_prompt": "jailbreak wrapper", "responses": ["r1", "r2"]}]
        flat = classifier_inputs(rows)
        self.assertEqual(flat, [{"behavior": "original harmful behavior", "response": "r1"},
                                {"behavior": "original harmful behavior", "response": "r2"}])
        with self.assertRaises(ValueError):
            classifier_inputs([{"behavior": "", "attack_prompt": "wrapper", "responses": ["r"]}])


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

    def test_acl_groups_extend_queue_without_rerunning_old_all_group(self):
        import yaml
        config = yaml.safe_load((ROOT / "configs/qwen3_4b.yaml").read_text())
        round1 = build_jobs(config, str(ROOT / "configs/qwen3_4b.yaml"), [], "acl-round1")
        self.assertEqual(sum(job["stage"] == "harmbench16_generate" for job in round1), 5)
        self.assertEqual(sum(job["stage"] == "harmbench16_judge" for job in round1), 1)
        self.assertTrue(any(job["stage"] == "analyze_asr" for job in round1))
        round2 = build_jobs(config, str(ROOT / "configs/qwen3_4b.yaml"), [], "acl-round2")
        train = [job for job in round2 if job["stage"] == "train"]
        self.assertEqual(len(train), 3)
        commands = [" ".join(job["command"]) for job in train]
        self.assertTrue(any("train.method=sirl" in command and "train.lr=1.0e-4" in command for command in commands))
        self.assertTrue(any("train.method=opsa" in command and "train.lr=5.0e-5" in command for command in commands))
        self.assertTrue(any("train.frozen_teacher_demo=false" in command and "train.frozen_teacher_weight=0.25" in command for command in commands))
        self.assertTrue(all(job["status"] == "planned" and job["pid"] is None for job in round2))

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

    def test_sirl_entropy_reward_is_detached_full_vocab_and_per_completion(self):
        import math
        import torch
        from sdsd_safety.losses import full_vocab_entropy_rewards
        uniform = torch.zeros(3, 4, requires_grad=True)
        peaked = torch.tensor([[100.0, 0.0, 0.0, 0.0]] * 2, requires_grad=True)
        rewards = full_vocab_entropy_rewards([uniform, peaked])
        self.assertEqual(len(rewards), 2)
        self.assertAlmostEqual(rewards[0], -math.log(4), places=6)
        self.assertAlmostEqual(rewards[1], 0.0, places=6)
        self.assertIsInstance(rewards[0], float)

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


class AdapterPathTests(unittest.TestCase):
    def test_new_trained_run_binds_expected_path_before_checkpoint_exists(self):
        from sdsd_safety.queue import adapter_argument, expected_adapter_path
        config = {"project_root": "/tmp/project", "train": {"output_root": "runs"}}
        path = expected_adapter_path(config, "brand_new_run")
        self.assertEqual(path, "/tmp/project/runs/brand_new_run/checkpoint")
        self.assertEqual(adapter_argument(config, "brand_new_run"), path)
        self.assertEqual(adapter_argument(config, "base_v2", is_base=True), "none")

    def test_acl_round2_jobs_never_plan_none_for_trained_runs(self):
        from sdsd_safety.utils import load_config
        config = load_config(ROOT / "configs" / "qwen3_4b.yaml")
        jobs = build_jobs(config, str(ROOT / "configs" / "qwen3_4b.yaml"), [], "acl-round2")
        checked = 0
        for job in jobs:
            if job["stage"] in {"generate", "harmbench16_generate"}:
                command = job["command"]
                if "--adapter" in command:
                    self.assertNotEqual(command[command.index("--adapter") + 1], "none")
                    checked += 1
                if "--run" in command:
                    spec = command[command.index("--run") + 1]
                    self.assertFalse(spec.endswith(":none"), spec)
                    checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()

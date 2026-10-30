"""One sequential experiment queue with a real PID/exit-code status heartbeat.

No environment installation, model download, or GPU process termination occurs.
Use --plan-only to inspect the commands; --execute runs on the existing host.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

from .utils import load_config, now_cst, read_jsonl, resolve_path, sha256_file, write_json

START = "<!-- QUEUE:START -->"
END = "<!-- QUEUE:END -->"
GRADIENT_TESTS = (
    "test_grpo_reference_changes_gradient",
    "test_arithmetic_teacher_mix_and_stop_gradient",
    "test_loss_pool_and_probability_pool_are_different",
)


def build_jobs(config: dict[str, Any], config_path: str, overrides: list[str], group: str) -> list[dict[str, Any]]:
    python = config.get("execution", {}).get("python", sys.executable)
    common = ["--config", str(Path(config_path).resolve())]
    for item in overrides:
        common += ["--override", item]
    jobs: list[dict[str, Any]] = []

    def add(label: str, module: str, args: list[str], stage: str | None = None,
            **contract: Any) -> None:
        inferred = module if module != "evaluate" else args[args.index("--phase") + 1]
        jobs.append({"id": f"Q{len(jobs) + 1:02d}", "task": label,
                     "command": [python, "-m", f"sdsd_safety.{module}", *common, *args],
                     "stage": stage or inferred,
                     "status": "planned", "pid": None, "exit_code": None, **contract})

    jobs.append({"id": "Q01", "task": "Run regression and Torch gradient tests", "stage": "tests",
                 "command": [python, "-m", "unittest", "discover", "-s", "tests", "-v"],
                 "status": "planned", "pid": None, "exit_code": None})

    if group == "legacy-rejudge":
        root = Path(config["project_root"]) / "results" / "raw"
        selected = set(config["eval"]["main_runs"] + config["eval"]["ablation_runs"])
        runs = sorted(p.parent.name for p in root.glob("*/harmbench.jsonl")
                      if p.parent.name not in selected and not (p.parent / "harmbench_generation_meta.json").exists())
        if not runs:
            raise FileNotFoundError("No saved legacy HarmBench answers are present on this host")
        add("Rejudge saved standard-200 answers", "evaluate", ["--phase", "judge", "--runs", ",".join(runs)],
            runs=runs, legacy=True)
        gsm = [r for r in runs if (root / r / "gsm8k.jsonl").exists()
               and not (root / r / "gsm8k_generation_meta.json").exists()]
        if gsm:
            add("Diagnose saved GSM8K answers", "evaluate", ["--phase", "diagnose", "--runs", ",".join(gsm)],
                runs=gsm, legacy=True)
        add("Summarize with legacy provenance", "evaluate", ["--phase", "summarize"])
        return jobs

    if group in {"all", "repair"}:
        add("Audit request intent and helpful answers", "audit_external", [])
        add("Verify shared refusal targets", "make_sft_targets", [])

    if group in {"acl-round1", "acl-round2"}:
        round1 = list(config["acl"]["round1_runs"])
        samples = int(config["eval"].get("asr_samples", 16))
        checkpoint_root = Path(config["project_root"]) / config["train"]["output_root"]

        def adapter_for(run_id: str) -> str:
            checkpoint = checkpoint_root / run_id / "checkpoint"
            return str(checkpoint) if checkpoint.is_dir() else "none"

        if group == "acl-round1":
            for run_id in round1:
                add(f"Generate 16-sample HarmBench for {run_id}", "harmbench16",
                    ["--phase", "generate", "--samples", str(samples), "--run", f"{run_id}:{adapter_for(run_id)}"],
                    stage="harmbench16_generate", run_id=run_id)
            judge_args = ["--phase", "judge", "--samples", str(samples)]
            for run_id in round1:
                judge_args += ["--run", f"{run_id}:{adapter_for(run_id)}"]
            add("Judge repeated samples with official classifier", "harmbench16", judge_args,
                stage="harmbench16_judge", runs=round1)
            add("Analyze paired ASR@k endpoints", "analyze_asr",
                ["--runs", ",".join(round1), "--reference", round1[0],
                 "--bootstrap-seed", str(config["acl"]["bootstrap_seed"]),
                 "--bootstrap-replicates", str(config["acl"]["bootstrap_replicates"]),
                 "--output-prefix", "asr_round1"], stage="analyze_asr", output_prefix="asr_round1")
            add("Refresh validated tables and ledger", "evaluate", ["--phase", "summarize"], stage="summarize")
            return jobs

        round2 = list(config["acl"]["round2_runs"])
        recipes = config["acl"]["round2_train"]
        if set(round2) != set(recipes):
            raise ValueError("acl.round2_runs and acl.round2_train must name the same runs")
        for run_id in round2:
            recipe = recipes[run_id]
            args = ["--run-id", run_id, "--override", f"train.method={recipe['method']}"]
            for setting in recipe.get("overrides", []):
                args += ["--override", setting]
            add(f"Train ACL variant {run_id}", "train", args, stage="train",
                run_id=run_id, method=recipe["method"])
            add(f"Generate standard benchmarks for {run_id}", "evaluate",
                ["--phase", "generate", "--run-id", run_id, "--adapter", adapter_for(run_id)],
                stage="generate", run_id=run_id)
            add(f"Generate 16-sample HarmBench for {run_id}", "harmbench16",
                ["--phase", "generate", "--samples", str(samples), "--run", f"{run_id}:{adapter_for(run_id)}"],
                stage="harmbench16_generate", run_id=run_id)
        add("Judge standard HarmBench for ACL variants", "evaluate",
            ["--phase", "judge", "--runs", ",".join(round2)], stage="judge", runs=round2)
        add("Judge XSTest over-refusal for ACL variants", "evaluate",
            ["--phase", "judge_xstest", "--runs", ",".join(round2)], stage="judge_xstest", runs=round2)
        judge_args = ["--phase", "judge", "--samples", str(samples)]
        for run_id in round2:
            judge_args += ["--run", f"{run_id}:{adapter_for(run_id)}"]
        add("Judge repeated samples for ACL variants", "harmbench16", judge_args,
            stage="harmbench16_judge", runs=round2)
        add("Analyze paired ASR@k for Round 2", "analyze_asr",
            ["--runs", ",".join(round1 + round2), "--reference", round1[0],
             "--bootstrap-seed", str(config["acl"]["bootstrap_seed"]),
             "--bootstrap-replicates", str(config["acl"]["bootstrap_replicates"]),
             "--output-prefix", "asr_round2"], stage="analyze_asr", output_prefix="asr_round2")
        add("Refresh validated tables and ledger", "evaluate", ["--phase", "summarize"], stage="summarize")
        return jobs

    main = list(config["eval"]["main_runs"])
    ablations = list(config["eval"]["ablation_runs"])
    if len(main) != 5 or len(ablations) != 4:
        raise ValueError("Expected five main runs and four explicitly named ablation/research runs")
    recipes = []
    if group in {"all", "repair"}:
        recipes += [(main[0], None, []),
                    (main[1], "sft", ["train.lr=1.0e-4"]),
                    (main[2], "grpo", ["train.lr=1.0e-4", "train.ref_kl_coef=0.01"]),
                    (main[3], "opsa", ["train.lr=1.0e-4"]),
                    (main[4], "sdsd", [])]
    if group in {"all", "ablations"}:
        recipes += [(ablations[0], "sdsd", ["train.demo_source=external_refusal"]),
                    (ablations[1], "sdsd", ["train.num_demos=3", "train.demo_pooling=mean_kl"])]
    if group in {"all", "research"}:
        recipes += [(ablations[2], "sdsd", ["train.frozen_teacher_weight=0.25"]),
                    (ablations[3], "sdsd", ["train.num_demos=3", "train.demo_pooling=probability_mean"])]
    for run_id, method, settings in recipes:
        if method:
            args = ["--run-id", run_id, "--override", f"train.method={method}"]
            for setting in settings:
                args += ["--override", setting]
            add(f"Train {run_id}", "train", args, run_id=run_id, method=method)
        adapter = str(Path(config["project_root"]) / config["train"]["output_root"] / run_id / "checkpoint") if method else "none"
        add(f"Generate {run_id}", "evaluate", ["--phase", "generate", "--run-id", run_id, "--adapter", adapter], run_id=run_id)
        add(f"Score safety {run_id}", "evaluate", ["--phase", "judge", "--runs", run_id], runs=[run_id])
        add(f"Score over-refusal {run_id}", "evaluate", ["--phase", "judge_xstest", "--runs", run_id], runs=[run_id])
        add(f"Refresh tables after {run_id}", "evaluate", ["--phase", "summarize"])
    return jobs


def source_snapshot() -> dict[str, str]:
    """Fingerprint executable package code, queue scripts and the test gate."""
    checkout = Path(__file__).resolve().parents[1]
    paths = set((checkout / "sdsd_safety").rglob("*.py")) | set((checkout / "tests").rglob("*.py"))
    for suffix in ("*.py", "*.sh"):
        paths.update((checkout / "scripts").rglob(suffix))
    return {str(path.relative_to(checkout)): sha256_file(path) for path in sorted(paths)}


def checked_file(path: str | Path, expected: str | None = None) -> str:
    from .evaluate import file_hash

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Required artifact is missing or empty: {path}")
    digest = file_hash(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return digest


def verify_manifest(config: dict[str, Any], *, targets: bool) -> dict[str, str]:
    from .evaluate import is_sha256

    root = Path(config["project_root"])
    path = Path(resolve_path(config, config["data"]["manifest"]))
    checked_file(path)
    manifest = json.loads(path.read_text())
    audit = manifest.get("prompt_audit", {})
    if audit.get("completed") is not True or audit.get("protocol") != "prompt-intent-v2":
        raise ValueError("The request-intent audit is not completed under prompt-intent-v2")
    files = manifest.get("files", {})
    artifacts = {}
    for name in ("train", "dev", "test"):
        entry = files.get(name, {})
        selected = Path(resolve_path(config, config["data"][name])).resolve()
        if not entry.get("path") or Path(resolve_path(config, entry["path"])).resolve() != selected or not is_sha256(entry.get("sha256")):
            raise ValueError(f"Manifest does not identify the configured {name} artifact")
        digest = checked_file(selected, entry["sha256"])
        if name != "train" or targets:
            artifacts[str(selected)] = digest
    target_info = manifest.get("targets", {})
    if targets:
        if target_info.get("completed") is not True or target_info.get("protocol") != "verified-refusal-targets-v2":
            raise ValueError("Verified refusal targets are incomplete")
        sft = files.get("sft_train", {})
        if (not sft.get("path") or Path(resolve_path(config, sft["path"])).resolve() != Path(resolve_path(config, config["data"]["train"])).resolve()
                or sft.get("sha256") != files["train"]["sha256"]):
            raise ValueError("SFT and on-policy training must share the same verified artifact")
        artifacts[str(path.resolve())] = checked_file(path)
    else:
        # make_sft_targets legitimately enriches train.jsonl and the manifest.
        # Its input digest preserves the audit-stage artifact identity.
        audited_hash = target_info.get("input_sha256") if target_info.get("completed") else files["train"]["sha256"]
        if not is_sha256(audited_hash):
            raise ValueError("Missing original audited training digest")
        artifacts["audited_train_input_sha256"] = audited_hash
        audit_path = root / "data" / "external_audit.jsonl"
        artifacts[str(audit_path.resolve())] = checked_file(audit_path)
    return artifacts


def verify_training(config: dict[str, Any], run_id: str, method: str | None = None) -> dict[str, str]:
    from .evaluate import TRAINING_PROTOCOL_IDS, is_sha256, model_identity

    directory = Path(resolve_path(config, config["train"]["output_root"])) / run_id
    path = directory / "run.json"
    artifacts = {str(path.resolve()): checked_file(path)}
    meta = json.loads(path.read_text())
    if (meta.get("protocol_id") not in TRAINING_PROTOCOL_IDS or meta.get("run_id") != run_id
            or meta.get("status") not in {"trained", "completed"} or meta.get("training_completed") is not True
            or meta.get("valid") is not True or meta.get("data_verified") is not True):
        raise ValueError(f"{run_id}: training is not completed and verified under the repaired protocol")
    if (not isinstance(meta.get("steps_requested"), int) or meta["steps_requested"] <= 0
            or meta.get("steps_completed") != meta["steps_requested"]
            or meta["steps_requested"] != int(config["train"]["steps"])
            or meta.get("seed") != int(config["seed"]) or method and meta.get("method") != method):
        raise ValueError(f"{run_id}: training budget, seed or method differs from the queue")
    for field, digest_field in (("data_path", "data_sha256"), ("data_manifest", "data_manifest_sha256")):
        if not meta.get(field) or not is_sha256(meta.get(digest_field)):
            raise ValueError(f"{run_id}: missing training artifact identity {field}")
        artifacts[str(Path(meta[field]).resolve())] = checked_file(meta[field], meta[digest_field])
    expected = {path.name: sha256_file(path) for path in Path(__file__).parent.glob("*.py")}
    if meta.get("code_sha256") != expected:
        raise ValueError(f"{run_id}: recorded training code differs from the actual package")
    checkpoint = directory / "checkpoint"
    if not meta.get("checkpoint") or Path(meta["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError(f"{run_id}: checkpoint path does not match this run")
    identity = model_identity(checkpoint, adapter=True)
    artifacts.update({str(checkpoint.resolve() / name): value["sha256"] for name, value in identity["files"].items()})
    return artifacts


def verify_evaluation(config: dict[str, Any], run_id: str, benchmark: str, *, scored: bool,
                      legacy: bool = False) -> dict[str, str]:
    from .evaluate import PROTOCOL_ID, generation_provenance, model_identity, source_path

    directory = Path(config["project_root"]) / "results" / "raw" / run_id
    raw = directory / f"{benchmark}.jsonl"
    artifacts = {str(raw.resolve()): checked_file(raw)}
    rows = read_jsonl(raw)
    if not rows:
        raise ValueError(f"{run_id}: {benchmark} has no saved generations")
    provenance = generation_provenance(config, run_id, benchmark, rows)
    if legacy:
        if provenance.get("legacy_provenance") is not True or provenance.get("generation_protocol_valid"):
            raise ValueError(f"{run_id}: not an explicitly legacy generation")
    else:
        if not provenance.get("generation_protocol_valid"):
            raise ValueError(f"{run_id}/{benchmark}: invalid generation provenance: {provenance.get('metadata_errors')}")
        meta = provenance["generation_meta"]
        for field in ("model_identity", "adapter_identity"):
            identity = meta.get(field)
            if identity is not None and not Path(identity["path"]).is_dir():
                raise ValueError(f"{run_id}: missing {field} artifact")
        checked_file(source_path(config, benchmark), meta["source_sha256"])
        generation_path = directory / f"{benchmark}_generation_meta.json"
        artifacts[str(generation_path.resolve())] = checked_file(generation_path, provenance["generation_meta_sha256"])
        if meta.get("adapter_identity") is not None:
            training = meta.get("training_provenance", {})
            run_path = Path(resolve_path(config, config["train"]["output_root"])) / run_id / "run.json"
            if not training.get("run_json_sha256"):
                raise ValueError(f"{run_id}: missing evaluated training metadata identity")
            checked_file(run_path, training["run_json_sha256"])
    if not scored:
        return artifacts
    path = directory / f"{benchmark}_summary.json"
    artifacts[str(path.resolve())] = checked_file(path)
    summary = json.loads(path.read_text())
    recorded = summary.get("provenance", {})
    if (summary.get("protocol_id") != PROTOCOL_ID or summary.get("status") != "completed"
            or summary.get("scoring_valid") is not True or summary.get("valid") is not (not legacy)):
        raise ValueError(f"{run_id}/{benchmark}: scoring did not produce a complete result for its provenance class")
    if recorded.get("source_raw_sha256") != artifacts[str(raw.resolve())]:
        raise ValueError(f"{run_id}/{benchmark}: scored raw generations changed")
    if legacy:
        if recorded.get("legacy_provenance") is not True or recorded.get("generation_protocol_valid"):
            raise ValueError("Legacy rescoring cannot be promoted to a v2 generation")
    elif recorded.get("generation_meta_sha256") != provenance["generation_meta_sha256"]:
        raise ValueError(f"{run_id}/{benchmark}: scored generation metadata changed")
    if benchmark in {"harmbench", "xstest"}:
        judged = directory / f"{benchmark}_judged.jsonl"
        if not summary.get("judged_sha256"):
            raise ValueError(f"{run_id}/{benchmark}: missing judged artifact digest")
        artifacts[str(judged.resolve())] = checked_file(judged, summary["judged_sha256"])
        identity = summary.get("scoring", {}).get("classifier_identity")
        if not identity or model_identity(identity["path"])["sha256"] != identity.get("sha256"):
            raise ValueError(f"{run_id}/{benchmark}: scoring model is missing or changed")
    if benchmark == "harmbench" and (summary.get("n_total") != 200 or summary.get("n_valid") != 200):
        raise ValueError(f"{run_id}: HarmBench requires 200 valid standard judgments")
    return artifacts


def verify_sampling(config: dict[str, Any], run_id: str, *, scored: bool) -> dict[str, str]:
    from .evaluate import file_hash, model_identity, sampling_dir
    from .metrics import ASR_SAMPLING_PROTOCOL_ID

    directory = sampling_dir(config, run_id)
    manifest_path = directory / "sampling_manifest.json"
    raw_path = directory / "harmbench16.jsonl"
    if not manifest_path.is_file() or not raw_path.is_file():
        raise ValueError(f"{run_id}: missing repeated-sampling manifest/raw output")
    manifest = json.loads(manifest_path.read_text())
    artifacts = {str(manifest_path.resolve()): checked_file(manifest_path),
                 str(raw_path.resolve()): checked_file(raw_path, manifest.get("raw_sha256"))}
    allowed = {"generating", "generated", "completed"} if not scored else {"completed"}
    if (manifest.get("protocol_id") != ASR_SAMPLING_PROTOCOL_ID
            or manifest.get("status") not in allowed or manifest.get("samples") != 16
            or manifest.get("n") != 200):
        raise ValueError(f"{run_id}: repeated-sampling manifest is incomplete or not 200x16")
    from .harmbench16 import SAMPLING_CODE_FILES

    expected_code = {name: sha256_file(Path(__file__).parent / name) for name in SAMPLING_CODE_FILES}
    if manifest.get("code_sha256") != expected_code:
        raise ValueError(f"{run_id}: sampling code differs from the recorded evaluation code")
    adapter = manifest.get("adapter_identity")
    if adapter is not None:
        if not Path(adapter["path"]).is_dir() or model_identity(adapter["path"], adapter=True)["sha256"] != adapter["sha256"]:
            raise ValueError(f"{run_id}: evaluated adapter is missing or changed")
    training = manifest.get("training_identity") or {}
    if training.get("run_json_sha256"):
        run_json = Path(resolve_path(config, config["train"]["output_root"])) / run_id / "run.json"
        checked_file(run_json, training["run_json_sha256"])
    if not scored:
        return artifacts
    summary_path = directory / "harmbench16_summary.json"
    judged_path = directory / "harmbench16_judged.jsonl"
    counts_path = directory / "harmbench16_prompt_counts.json"
    summary = json.loads(checked_file(summary_path))
    artifacts[str(summary_path.resolve())] = file_hash(summary_path)
    artifacts[str(judged_path.resolve())] = checked_file(judged_path, summary.get("judged_sha256"))
    artifacts[str(counts_path.resolve())] = checked_file(counts_path, summary.get("prompt_counts_sha256"))
    if (summary.get("protocol_id") != ASR_SAMPLING_PROTOCOL_ID or summary.get("status") != "completed"
            or summary.get("valid") is not True or summary.get("samples") != 16
            or summary.get("n_behaviors") != 200
            or set(summary.get("asr_endpoints", {})) != {"1", "4", "16"}):
        raise ValueError(f"{run_id}: repeated-sampling scoring is incomplete")
    if (manifest.get("status") != "completed" or manifest.get("valid") is not True
            or manifest.get("summary_sha256") != file_hash(summary_path)
            or manifest.get("judged_sha256") != summary.get("judged_sha256")
            or manifest.get("prompt_counts_sha256") != summary.get("prompt_counts_sha256")):
        raise ValueError(f"{run_id}: sampling manifest and scored artifacts disagree")
    return artifacts


def verify_analysis(config: dict[str, Any], prefix: str) -> dict[str, str]:
    from .evaluate import file_hash

    root = Path(config["project_root"])
    report = root / "results" / "analysis" / f"{prefix}.json"
    paired = root / "results" / "tables" / f"{prefix}_paired.md"
    artifacts = {str(report.resolve()): checked_file(report),
                 str(paired.resolve()): checked_file(paired)}
    data = json.loads(report.read_text())
    if data.get("protocol_id") != "sdsd-safety-asr-sample-v3" or not data.get("paired"):
        raise ValueError("ASR analysis report lacks protocol or paired endpoints")
    return artifacts


def verify_job(config: dict[str, Any], job: dict[str, Any]) -> dict[str, str]:
    root, stage = Path(config["project_root"]), job["stage"]
    if stage == "tests":
        path = root / "logs" / f"queue_{job['id']}.log"
        digest = checked_file(path)
        data = path.read_bytes()
        offset = int(job.get("log_offset", 0))
        if not 0 <= offset < len(data):
            raise ValueError("Missing current unittest attempt output")
        lines = data[offset:].decode(errors="replace").splitlines()
        if not all(any(name in line and line.rstrip().endswith(" ... ok") for line in lines) for name in GRADIENT_TESTS):
            raise ValueError("The three Torch gradient/mixture tests must run and pass, not skip")
        return {str(path.resolve()): digest}
    if stage in {"harmbench16_generate", "harmbench16_judge"}:
        if "run_id" in job:
            return verify_sampling(config, job["run_id"], scored=stage == "harmbench16_judge")
        artifacts = {}
        for run_id in job["runs"]:
            artifacts.update(verify_sampling(config, run_id, scored=stage == "harmbench16_judge"))
        return artifacts
    if stage == "analyze_asr":
        return verify_analysis(config, job["output_prefix"])
    if stage in {"audit_external", "make_sft_targets"}:
        return verify_manifest(config, targets=stage == "make_sft_targets")
    if stage == "train":
        return verify_training(config, job["run_id"], job["method"])
    if stage == "summarize":
        path = root / "results" / "summary.json"
        checked_file(path)
        if not isinstance(json.loads(path.read_text()), list):
            raise ValueError("The summary ledger must be a JSON list")
        return {}  # Later jobs legitimately refresh an intermediate, possibly empty ledger.
    artifacts = {}
    runs = [job["run_id"]] if stage == "generate" else job["runs"]
    benchmarks = ("harmbench", "xstest", "gsm8k") if stage == "generate" else ({"judge": "harmbench", "judge_xstest": "xstest", "diagnose": "gsm8k"}[stage],)
    for run_id in runs:
        for benchmark in benchmarks:
            artifacts.update(verify_evaluation(config, run_id, benchmark, scored=stage != "generate", legacy=job.get("legacy", False)))
            if stage == "diagnose":
                path = root / "results" / "raw" / run_id / "gsm8k_diagnostics.json"
                artifacts[str(path.resolve())] = checked_file(path)
    return artifacts


def verify_resumed_jobs(config: dict[str, Any], jobs: list[dict[str, Any]]) -> list[str]:
    errors = []
    for job in jobs:
        if job.get("status") != "succeeded":
            continue
        try:
            if job.get("exit_code") != 0 or not isinstance(job.get("artifact_sha256"), dict):
                raise ValueError("previous success lacks an exit-code/artifact validation record")
            if verify_job(config, job) != job["artifact_sha256"]:
                raise ValueError("required artifacts differ from the successful task's recorded identities")
        except Exception as error:
            reason = f"Resume verification {job['id']}: {error}"
            job.update(status="blocked", reason=reason, pid=None)
            errors.append(reason)
    return errors


def verify_group(config: dict[str, Any], jobs: list[dict[str, Any]], group: str) -> dict[str, Any]:
    from .evaluate import PROTOCOL_ID, summary_phase

    summary_phase(config)  # Recompute the ledger; an old valid=True flag is not enough.
    rows = json.loads((Path(config["project_root"]) / "results" / "summary.json").read_text())
    by_id = {row["run_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("The summary ledger contains duplicate run identities")
    if group in {"acl-round1", "acl-round2"}:
        for job in jobs:
            if job["stage"] in {"generate", "train", "judge", "judge_xstest",
                                "harmbench16_generate", "harmbench16_judge", "analyze_asr", "summarize"}:
                verify_job(config, job)
        sampling_runs = sorted({job["run_id"] for job in jobs if job["stage"] == "harmbench16_generate"})
        if not sampling_runs:
            raise ValueError("ACL group has no repeated-sampling runs")
        for run_id in sampling_runs:
            verify_sampling(config, run_id, scored=True)
            row = by_id.get(run_id, {})
            if row.get("valid") is not True:
                raise ValueError(f"{run_id}: standard evaluation is not validated: {row.get('validation_errors', ['missing'])}")
            if row.get("sampled_asr_at_16") is None:
                raise ValueError(f"{run_id}: repeated-sampling endpoints are absent from the ledger")
        if group == "acl-round2":
            for job in jobs:
                if job["stage"] == "train":
                    verify_training(config, job["run_id"], job["method"])
        return {"scope": "acl_repeated_sampling", "runs": sampling_runs}
    if group == "legacy-rejudge":
        expected = next(job["runs"] for job in jobs if job["stage"] == "judge")
        for job in jobs:
            if job.get("legacy"):
                verify_job(config, job)
        for run_id in expected:
            row = by_id.get(run_id, {})
            if not row or row.get("valid") is not False or row.get("excluded_from_main") is not True or row.get("table_group") != "unassigned":
                raise ValueError(f"{run_id}: legacy rescoring must remain explicitly excluded from validated v2 tables")
        return {"scope": "legacy_rescoring_only", "legacy_rescored_runs": expected, "validated_v2_runs": []}
    expected = [job["run_id"] for job in jobs if job["stage"] == "generate"]
    for job in jobs:
        if job["stage"] == "train":
            verify_training(config, job["run_id"], job["method"])
    for run_id in expected:
        row = by_id.get(run_id, {})
        if row.get("valid") is not True or row.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"{run_id}: no valid completed result in summary.json: {row.get('validation_errors', ['missing'])}")
        for benchmark in ("harmbench", "xstest", "gsm8k"):
            verify_evaluation(config, run_id, benchmark, scored=True)
    if not expected:
        raise ValueError("No expected v2 runs were scheduled for this group")
    return {"scope": "validated_v2", "validated_v2_runs": expected}


def preflight(config: dict[str, Any], gpu: str, group: str) -> list[str]:
    errors = []
    if Path(config["project_root"]).resolve() != Path(__file__).resolve().parents[1]:
        errors.append("project_root must be the actual source checkout running this queue")
    python = config.get("execution", {}).get("python", sys.executable)
    if not Path(python).is_file():
        errors.append(f"Existing Python environment is unavailable: {python}")
    else:
        check = subprocess.run([python, "-c", "import torch,transformers,peft,yaml; assert torch.cuda.is_available(), 'CUDA unavailable'"],
                               capture_output=True, text=True, timeout=60)
        if check.returncode:
            errors.append("Existing environment cannot run CUDA training: " + check.stderr.strip()[-500:])
    for label, path in {"model": config["model"]["path"], "judge": config["judge"]["path"],
                        **{k: resolve_path(config, v) for k, v in config["sources"].items()}}.items():
        if not Path(path).exists():
            errors.append(f"Missing {label}: {path}")
    if shutil.which("nvidia-smi") is None:
        errors.append("No nvidia-smi/GPU is available on this host")
    if group in {"ablations", "research", "acl-round2"}:
        path = Path(resolve_path(config, config["data"]["manifest"]))
        if not path.is_file():
            errors.append("Run the audited data/target stages before the ablation or research group")
    if group in {"acl-round1", "acl-round2"}:
        for run_id in config["acl"]["round1_runs"]:
            checkpoint = Path(config["project_root"]) / config["train"]["output_root"] / run_id / "checkpoint"
            if run_id != "base_v2" and not checkpoint.is_dir():
                errors.append(f"Missing frozen Round 1 checkpoint: {run_id}")
    return errors


def last_output(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 4096))
        lines = handle.read().decode("utf-8", errors="replace").strip().splitlines()
    return (lines[-1] if lines else "").replace("|", "/")[-220:]


def update_status(root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_cst()
    write_json(root / "runs" / "queue.json", state)
    table = [START, f"更新：{state['updated_at']}；队列状态：**{state['status']}**；GPU：{state['gpu']}", "",
             "| ID | 任务 | 状态 | PID | Exit | 最近输出 / 原因 |",
             "|---|---|---|---|---|---|"]
    if state.get("blockers"):
        table += [f"| — | 环境检查 | blocked | — | — | {str(reason).replace('|', '/')} |" for reason in state["blockers"]]
    for job in state["jobs"]:
        recent = job.get("reason") or last_output(root / "logs" / f"queue_{job['id']}.log")
        table.append(f"| {job['id']} | {job['task']} | {job['status']} | {job.get('pid') or '—'} | "
                     f"{job['exit_code'] if job.get('exit_code') is not None else '—'} | {recent} |")
    if state.get("last_push"):
        table += ["", f"最近已验证推送：{state['last_push']}"]
    if state.get("publication", {}).get("status") == "failed":
        table += ["", f"推送失败（实验状态不变）：{state['publication']['error']}"]
    table.append(END)
    block = "\n".join(table)
    path = root / "status.md"
    old = path.read_text() if path.exists() else "# SDSD-Safety status\n\n"
    if START in old and END in old:
        text = old[: old.index(START)] + block + old[old.index(END) + len(END):]
    else:
        text = old.rstrip() + "\n\n" + block + "\n"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def wait_for_memory(config: dict[str, Any], gpu: str, root: Path, state: dict[str, Any], job: dict[str, Any]) -> None:
    threshold = int(config.get("execution", {}).get("min_free_gpu_mib", 30000))
    while True:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", gpu],
                                capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        free = int(result.stdout.strip().splitlines()[0])
        if free >= threshold:
            job.pop("reason", None)
            return
        job.update(status="queued", reason=f"GPU {gpu}: {free} MiB free; waiting for {threshold} MiB")
        update_status(root, state)
        time.sleep(min(60, max(1, int(config.get("execution", {}).get("heartbeat_seconds", 60)))))


def publish_results(root: Path) -> str:
    """Publish only this project's small result/status files using existing Git auth."""
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT).strip()
    remote = git("remote", "get-url", "origin")
    if remote not in {"https://github.com/jiajunsun68/SDSD-safety.git", "git@github.com:jiajunsun68/SDSD-safety.git"}:
        raise RuntimeError("origin does not match the authorized SDSD-safety repository")
    allowed = ["status.md", "results/summary.json", "results/summary.csv", "results/tables/main.csv",
               "results/tables/main.md", "results/tables/ablations.csv", "results/tables/ablations.md"]
    staged = set(git("diff", "--cached", "--name-only").splitlines())
    if staged - set(allowed):
        raise RuntimeError("Unrelated staged files exist; leaving them untouched")
    present = [p for p in allowed if (root / p).is_file()]
    git("add", "--", *present)
    if git("diff", "--cached", "--name-only"):
        git("commit", "-m", "Update validated SDSD-Safety v2 experiment results")
    branch = git("symbolic-ref", "--short", "HEAD")
    git("push", "origin", f"HEAD:refs/heads/{branch}")
    sha = git("rev-parse", "HEAD")
    advertised = git("ls-remote", "origin", f"refs/heads/{branch}").split()
    if not advertised or advertised[0] != sha:
        raise RuntimeError("Push could not be verified against the remote branch head")
    return f"{branch}@{sha}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_4b.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--group", choices=["all", "repair", "ablations", "research", "legacy-rejudge",
                                            "acl-round1", "acl-round2"], default="repair")
    parser.add_argument("--gpu", default="0")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Revalidate completed artifacts before skipping tasks with the identical code/config fingerprint")
    parser.add_argument("--push", action="store_true", help="Commit and push small results using existing authorized Git credentials")
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    root = Path(config["project_root"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    lock = (root / "runs" / "queue.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another queue runner owns this project; no duplicate jobs were started")
    jobs = build_jobs(config, args.config, args.override, args.group)
    code = source_snapshot()
    config_digest = sha256_file(args.config)
    fingerprint = hashlib.sha256(json.dumps({"config": config, "jobs": jobs, "code_sha256": code,
                                             "config_source_sha256": config_digest}, sort_keys=True).encode()).hexdigest()
    state = {"status": "planned", "group": args.group, "gpu": args.gpu, "fingerprint": fingerprint,
             "code_sha256": code, "config_source_sha256": config_digest,
             "runner_pid": os.getpid() if args.execute else None, "jobs": jobs, "blockers": []}
    state_path = root / "runs" / "queue.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        for job in previous.get("jobs", []):
            pid = job.get("pid")
            proc = Path(f"/proc/{pid}/cmdline") if pid else None
            if job.get("status") == "running" and proc and proc.exists():
                actual = proc.read_bytes().split(b"\0")
                expected = [str(part).encode() for part in job["command"]]
                if actual[:len(expected)] == expected:
                    raise SystemExit(f"Existing task {job['id']} is still running as PID {pid}; not starting a duplicate")
        if args.resume:
            if previous.get("fingerprint") != fingerprint:
                previous.update(status="blocked", runner_pid=None,
                                blockers=["Queue code, tests, scripts or configuration changed; resume fingerprint mismatch"])
                update_status(root, previous)
                raise SystemExit(previous["blockers"][0])
            state = previous
            state.update(runner_pid=os.getpid() if args.execute else None, blockers=[])
            errors = verify_resumed_jobs(config, state["jobs"])
            if errors:
                state.update(status="blocked", runner_pid=None, blockers=errors)
                update_status(root, state)
                raise SystemExit("Resume blocked: " + "; ".join(errors))
    elif args.resume:
        state.update(status="blocked", runner_pid=None, blockers=["No saved queue state exists to resume"])
        update_status(root, state)
        raise SystemExit(state["blockers"][0])
    for job in state["jobs"]:
        if job["status"] != "succeeded":
            job.update(status="planned", pid=None, exit_code=None)
            job.pop("reason", None)
            job.pop("artifact_sha256", None)
    update_status(root, state)
    if args.plan_only:
        print(json.dumps(state, indent=2))
        return
    blockers = preflight(config, args.gpu, args.group)
    if blockers:
        state.update(status="blocked", blockers=blockers, runner_pid=None)
        update_status(root, state)
        raise SystemExit("Execution blocked: " + "; ".join(blockers))
    state["status"] = "running"
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="8",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    heartbeat = min(60, max(1, int(config.get("execution", {}).get("heartbeat_seconds", 60))))
    try:
        for index, job in enumerate(state["jobs"]):
            if source_snapshot() != code or sha256_file(args.config) != config_digest:
                state["status"] = "blocked"
                raise RuntimeError("Source code/tests/scripts or configuration changed while the queue was running")
            if job["status"] == "succeeded":
                continue
            job.update(status="queued", pid=None)
            update_status(root, state)
            if job["stage"] not in {"tests", "summarize", "diagnose"}:
                wait_for_memory(config, args.gpu, root, state, job)
            log_path = root / "logs" / f"queue_{job['id']}.log"
            with log_path.open("a") as log:
                job["log_offset"] = log.tell()
                log.write(f"\n{now_cst()} {shlex.join(job['command'])}\n")
                log.flush()
                process = subprocess.Popen(job["command"], cwd=root, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                job.update(status="running", pid=process.pid, started_at=now_cst())
                update_status(root, state)
                while process.poll() is None:
                    try:
                        process.wait(timeout=heartbeat)
                    except subprocess.TimeoutExpired:
                        update_status(root, state)
                job.update(status="validating" if process.returncode == 0 else "failed",
                           exit_code=process.returncode, finished_at=now_cst())
            update_status(root, state)
            if process.returncode:
                for dependent in state["jobs"][index + 1:]:
                    dependent.update(status="blocked", reason=f"Dependency {job['id']} failed")
                raise RuntimeError(f"{job['id']} exited {process.returncode}; see {log_path}")
            try:
                artifacts = verify_job(config, job)
            except Exception as error:
                job.update(status="blocked", reason=f"Exit 0 but artifact validation failed: {error}")
                state["status"] = "blocked"
                for dependent in state["jobs"][index + 1:]:
                    dependent.update(status="blocked", reason=f"Dependency {job['id']} has unverified artifacts")
                update_status(root, state)
                raise RuntimeError(job["reason"]) from error
            job.update(status="succeeded", artifact_sha256=artifacts, artifacts_verified_at=now_cst())
            update_status(root, state)
        try:
            completion = verify_group(config, state["jobs"], args.group)
        except Exception as error:
            state["status"] = "blocked"
            raise RuntimeError(f"Group completion gate failed: {error}") from error
        state.update(status="succeeded", runner_pid=None, result=completion)
        update_status(root, state)
        if args.push:
            try:
                state["last_push"] = publish_results(root)
                state["publication"] = {"status": "succeeded", "verified_at": now_cst()}
            except Exception as error:
                state["publication"] = {"status": "failed", "error": f"{type(error).__name__}: {error}", "attempted_at": now_cst()}
                print("Results completed, but publishing failed: " + state["publication"]["error"], file=sys.stderr)
            update_status(root, state)
    except BaseException as error:
        state.update(status="blocked" if state["status"] == "blocked" or isinstance(error, KeyboardInterrupt) else "failed",
                     runner_pid=None, blockers=[f"{type(error).__name__}: {error}"])
        update_status(root, state)
        raise


if __name__ == "__main__":
    main()

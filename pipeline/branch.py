"""Branch rollouts: resample a whole trajectory from an edited reasoning prefix.

From a seed run, a branch rolls out everything after API request ``k`` again, with the
reasoning of step ``k`` replaced by an edited prefix (e.g. with one sentence removed) that
the model continues itself. Every branch is a normal pipeline run directory holding a FULL,
standalone trajectory (seed steps before the branch point + the new steps, continuous step
ids), so events / score / judge / analyze work on branches unchanged. Which text came from
the seed and which was injected is recorded only in ``branch_meta.json``.

Mechanics (Codex engine, open models via OpenRouter; see harness.prefill for support):
1. the work dir is restored from the seed's shadow git at the last snapshot before step k
   (plus a re-seeded git history when realism is on);
2. Codex resumes (``codex exec resume <id>``) from the seed rollout truncated to request
   k's input, with the seed's working-directory path rewritten to the new one;
3. the capture proxy strips what Codex adds on resume (fresh developer + environment
   messages and the resume prompt), so the model's input equals the seed's request k;
   it answers request k with a raw-completion prefill (harness.prefill) and forwards
   every later request to the same pinned provider.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from harness.config import RunConfig, SessionConfig, load_config
from harness.experiment import _build_run_meta
from harness.isolation import AgentUser, chown_tree, hand_over_work_dir
from harness.prefill import (
    BranchUnsupportedError,
    ModelSpec,
    check_branch_support,
    complete_raw,
    count_tokens,
    render_chat,
    render_prompt,
    request_effort,
    responses_sse,
)
from harness.reasoning_capture import (
    attach_reasoning,
    parse_responses_sse,
    records_from_raw_dumps,
)
from harness.resume import RESUME_MARKER, strip_resume_additions
from harness.runner import run_session
from harness.shadow_git import ShadowGit
from harness.state import StateManager

_IGNORE = {".git", "__pycache__"}


class Seed:
    """A seed run directory and everything a branch at request ``k`` needs from it."""

    def __init__(self, run_dir: str | Path, k: int, provider: str | None = None):
        self.run_dir = Path(run_dir)
        self.k = k
        self.session = self.run_dir / "session_01"
        self.config: RunConfig = load_config(self.run_dir / "config.yaml")
        raw = self.session / "raw_dumps"
        req = raw / f"request_{k:03d}.json"
        resp = raw / f"response_{k:03d}.txt"
        if not req.exists() or not resp.exists():
            raise ValueError(f"seed {self.run_dir} has no captured request/response {k}")
        self.request: dict = json.loads(req.read_text())
        self.provider = provider or (self.config.provider_order or [None])[0]
        self.spec: ModelSpec = check_branch_support(
            self.config.engine, self.config.provider, self.config.model,
            [self.provider] if self.provider else None)
        record = parse_responses_sse(resp, k)
        if not record.text:
            raise ValueError(f"request {k} of the seed produced no readable reasoning to edit")
        self.original_reasoning: str = record.text
        meta_line = (self.session / "transcript.jsonl").read_text().splitlines()[0]
        self.old_cwd: str = json.loads(meta_line)["payload"]["cwd"]
        # Seeds generated while Codex still sent web_search had their past reasoning
        # dropped by OpenRouter; mirror that condition for the branch step.
        self.past_reasoning_dropped = any(
            t.get("type") not in ("function", "namespace", "custom")
            for t in self.request.get("tools") or [] if isinstance(t, dict))
        self.provider_input_tokens: int | None = _sse_input_tokens(resp)
        self._fidelity: dict | None = None
        self._locate_branch_step()

    def template_fidelity(self) -> dict:
        """Our rendering of request k vs the provider's own (its input token count).

        The branch step is prefilled from OUR rendering of the chat template while every
        other step is rendered by the provider, so they must agree. ``gap`` = our tokens
        minus the provider's (0 = identical length).
        """
        if self._fidelity is None:
            ours = count_tokens(self.spec, render_chat(self.request, self.spec, self.past_reasoning_dropped,
                                                          self.provider))
            prov = self.provider_input_tokens
            self._fidelity = {"rendered_tokens": ours, "provider_tokens": prov,
                              "gap": None if prov is None else ours - prov}
        return self._fidelity

    def check_fidelity(self, max_gap: int) -> dict:
        f = self.template_fidelity()
        if f["gap"] is None:
            raise ValueError(f"seed {self.run_dir} response {self.k} carries no input token count; "
                             "cannot verify the prompt rendering")
        if abs(f["gap"]) > max_gap:
            raise BranchUnsupportedError(
                f"Our chat-template rendering of seed request {self.k} is {f['rendered_tokens']} tokens, "
                f"the provider's is {f['provider_tokens']} (gap {f['gap']:+d}, allowed ±{max_gap}): the "
                f"branch step would see a different prompt than the rest of the rollout. Raise "
                f"branch.max_prompt_token_gap only if that difference is acceptable.")
        return f

    def sampling(self) -> dict:
        """Reasoning effort / temperature / top_p the seed's request k was sent with."""
        return {"reasoning_effort": request_effort(self.request),
                "temperature": self.request.get("temperature"), "top_p": self.request.get("top_p")}

    def _locate_branch_step(self) -> None:
        traj = json.loads((self.session / "trajectory.json").read_text())
        steps = traj.get("steps") or []
        stats = attach_reasoning(json.loads(json.dumps(steps)), records_from_raw_dumps(self.session))
        step_id = stats["record_steps"].get(self.k)
        if step_id is None:
            raise ValueError(f"could not locate the trajectory step produced by request {self.k}")
        self.branch_step_id: int = step_id
        # Seed steps before the branch point, with their reasoning recovered.
        self.prefix_steps = [s for s in json.loads(json.dumps(steps)) if s.get("step_id", 0) < step_id]
        attach_reasoning(self.prefix_steps, records_from_raw_dumps(self.session)[: self.k - 1])
        tags = subprocess.run(["git", "--git-dir", str(self.run_dir / ".shadow_git"), "tag"],
                              capture_output=True, text=True, check=True).stdout.split()
        snaps = sorted(int(t.rsplit("_", 1)[1]) for t in tags
                       if t.startswith("_step_1_") and int(t.rsplit("_", 1)[1]) < step_id)
        self.reset_tag = f"_step_1_{snaps[-1]}" if snaps else "baseline"

    def prefix_for(self, arm: dict) -> str:
        r = self.original_reasoning
        if arm.get("prefix") is not None:
            base = arm["prefix"]
        elif arm.get("full_original"):
            base = r
        elif arm.get("prefix_until"):
            i = r.find(arm["prefix_until"])
            if i < 0:
                raise ValueError(f"prefix_until text not found in step {self.k} reasoning: {arm['prefix_until']!r}")
            base = r[:i]
        elif arm.get("prefix_through"):
            i = r.find(arm["prefix_through"])
            if i < 0:
                raise ValueError(f"prefix_through text not found in step {self.k} reasoning: {arm['prefix_through']!r}")
            base = r[: i + len(arm["prefix_through"])]
        else:
            raise ValueError("a branch arm needs one of: prefix, full_original, prefix_until, prefix_through")
        # Trailing whitespace would be its own token(s) where the model would naturally
        # emit e.g. " The" next, an unnatural tokenization at the seam.
        return (base + (arm.get("append") or "")).rstrip()

    def truncated_rollout(self, new_cwd: str) -> tuple[list[dict], str]:
        need = len(self.request.get("input") or [])
        kept, n = [], 0
        for line in (self.session / "transcript.jsonl").read_text().splitlines():
            e = json.loads(line)
            t = e.get("type")
            if t in ("session_meta", "turn_context"):
                kept.append(e)
            elif t == "response_item":
                if n >= need:
                    break
                kept.append(e)
                n += 1
        sid = str(uuid.uuid4())
        for e in kept:
            if e.get("type") == "session_meta":
                e["payload"]["id"] = sid
                e["payload"]["session_id"] = sid
        return json.loads(json.dumps(kept).replace(self.old_cwd, new_cwd)), sid


class BranchSplice:
    """Capture-proxy intercept for one branch rollout (see module docstring, step 3)."""

    def __init__(self, seed: Seed, prefix: str, new_cwd: str, api_key: str,
                 history_replace: list[dict] | None = None, max_attempts: int = 3):
        self.seed, self.prefix, self.api_key = seed, prefix, api_key
        self.max_attempts = max_attempts
        self.history_replace = history_replace or []
        self.expected = self._edit(json.loads(json.dumps(seed.request["input"]).replace(seed.old_cwd, new_cwd)))
        self.n = 0
        self.meta: dict = {}

    def _edit(self, items: list[dict]) -> list[dict]:
        if not self.history_replace:
            return items
        s = json.dumps(items)
        for r in self.history_replace:
            s = s.replace(json.dumps(r["old"])[1:-1], json.dumps(r["new"])[1:-1])
        return json.loads(s)

    async def __call__(self, request_data: dict, request_index: int) -> bytes | None:
        request_data["input"] = self._edit(strip_resume_additions(
            request_data.get("input") or [], len(self.seed.request.get("input") or [])))
        if self.n >= 1:
            self.n += 1
            return None   # forwarded to the pinned provider (capture proxy injects the pin)
        got = request_data["input"]
        self.meta["first_request_matches_seed"] = got == self.expected
        if got != self.expected:
            self.meta["first_request_mismatch"] = {
                "seed_items": len(self.expected), "branch_items": len(got),
                "first_differing_item": next((i for i, (a, b) in enumerate(zip(self.expected, got)) if a != b),
                                             min(len(self.expected), len(got)))}
        drop = self.seed.past_reasoning_dropped
        sampling = {"reasoning_effort": request_effort(request_data),
                    "temperature": request_data.get("temperature"), "top_p": request_data.get("top_p")}
        self.meta["sampling"] = sampling
        self.meta["sampling_matches_seed"] = sampling == self.seed.sampling()
        prompt, n_tok = render_prompt(request_data, self.seed.spec, self.prefix, drop_past_reasoning=drop,
                                      provider=self.seed.provider)
        # A continuation without a parseable action (cut off, malformed call) would end the
        # rollout on a broken step: re-sample it, up to max_attempts, and record every try.
        attempts = []
        for _ in range(self.max_attempts):
            d = await _complete_with_retry(self.seed.config.model, self.seed.provider, prompt, self.api_key,
                                           temperature=sampling["temperature"], top_p=sampling["top_p"])
            text = d["choices"][0].get("text") or ""
            parsed = self.seed.spec.parser(text)
            attempts.append({"continuation": text, "action_parsed": parsed.complete,
                             "finish_reason": d["choices"][0].get("finish_reason"),
                             "served_by": d.get("provider")})
            if parsed.complete:
                break
        self.meta.update({
            "prompt_tokens_rendered": n_tok, "prompt_tokens_provider": (d.get("usage") or {}).get("prompt_tokens"),
            "past_reasoning_dropped": drop, "continuation": text, "action_parsed": parsed.complete,
            "n_attempts": len(attempts), "attempts": attempts,
            "tool_calls": parsed.tool_calls, "message": parsed.message, "served_by": d.get("provider"),
        })
        self.n = 1   # the branch step is answered; every later request is forwarded
        namespaces = {t.get("name") for t in request_data.get("tools") or [] if t.get("type") == "namespace"}
        return responses_sse(self.seed.config.model, self.prefix + parsed.reasoning, parsed, d.get("usage"),
                             namespaces)


async def _complete_with_retry(*args, attempts: int = 12, **kwargs) -> dict:
    """complete_raw with backoff on transient failures (provider 429s from shared pools,
    5xx, timeouts, empty choices). Raises after ``attempts`` tries; the proxy then fails
    the request and the client retries it."""
    import asyncio
    import random

    for i in range(attempts):
        try:
            return await complete_raw(*args, **kwargs)
        except Exception:
            if i == attempts - 1:
                raise
            await asyncio.sleep(min(60.0, 2.0 ** i) * random.uniform(1.0, 1.5))
    raise RuntimeError("unreachable")


def _sse_input_tokens(path: Path) -> int | None:
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("data:") and "response.completed" in line:
            try:
                return ((json.loads(line[5:]).get("response") or {}).get("usage") or {}).get("input_tokens")
            except json.JSONDecodeError:
                return None
    return None


def _export(git_dir: Path, ref: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "--git-dir", str(git_dir), "archive", ref], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)


def _full_diff(seed_git: Path, final_dir: Path) -> str:
    """Diff from the seed's baseline (the task's initial state) to the branch's final files."""
    with tempfile.TemporaryDirectory() as tmp:
        base, gd = Path(tmp) / "base", Path(tmp) / "git"
        _export(seed_git, "baseline", base)
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(GIT_DIR=str(gd), GIT_AUTHOR_NAME="x", GIT_AUTHOR_EMAIL="x@x",
                   GIT_COMMITTER_NAME="x", GIT_COMMITTER_EMAIL="x@x",
                   GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")

        def git(*a, tree):
            return subprocess.run(["git", "-c", "core.excludesFile=/dev/null", "-c", "commit.gpgsign=false", *a],
                                  cwd=tree, env={**env, "GIT_WORK_TREE": str(tree)},
                                  capture_output=True, text=True, check=True).stdout
        subprocess.run(["git", "init", "-q"], env=env, check=True, capture_output=True)
        (gd / "info").mkdir(exist_ok=True)
        (gd / "info" / "exclude").write_text("\n".join([".git", "__pycache__", "*.pyc"]) + "\n")
        git("add", "-A", tree=base)
        git("commit", "-q", "--allow-empty", "-m", "baseline", tree=base)
        git("add", "-A", tree=final_dir)
        return git("diff", "--cached", "HEAD", tree=final_dir)


async def run_branch(cfg, base_cfg: RunConfig, arm_name: str, rep: int, run_dir: Path,
                     work_dir: Path, agent_user: AgentUser | None) -> None:
    """One branch rollout into ``run_dir`` (a pipeline trajectory dir)."""
    from pipeline.workspace import seed_git_history

    bc = cfg.branch
    seed = Seed(bc.seed_run, bc.request, bc.provider)
    arm = bc.arms[arm_name].model_dump()
    prefix = seed.prefix_for(arm)
    fidelity = seed.check_fidelity(bc.max_prompt_token_gap)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. work dir at the snapshot before the branch step (+ git history like the seed's)
    _export(seed.run_dir / ".shadow_git", seed.reset_tag, work_dir)
    if cfg.realism.seed_git_history:
        with tempfile.TemporaryDirectory() as tmp:
            _export(seed.run_dir / ".shadow_git", "baseline", Path(tmp))
            seed_git_history(Path(tmp), cfg.realism.git_author)
            if (Path(tmp) / ".git").exists():
                shutil.move(str(Path(tmp) / ".git"), str(work_dir / ".git"))

    # 2. Codex home holding the truncated seed rollout to resume from
    if agent_user:
        hand_over_work_dir(work_dir, agent_user)   # owner-only work dir, fresh ~/.codex + ~/tmp
    codex_home = (agent_user.home / ".codex") if agent_user else (run_dir / "codex_home").resolve()
    entries, sid = seed.truncated_rollout(str(work_dir))
    day = datetime.now(timezone.utc)
    rollout = codex_home / "sessions" / day.strftime("%Y/%m/%d") / f"rollout-{day.strftime('%Y-%m-%dT%H-%M-%S')}-{sid}.jsonl"
    rollout.parent.mkdir(parents=True, exist_ok=True)
    rollout.write_text("".join(json.dumps(e) + "\n" for e in entries))
    if agent_user:
        chown_tree(codex_home, agent_user)

    # 3. run config: the seed's, pinned to the branch provider, resumed with a marker prompt
    rc = seed.config.model_copy(deep=True)
    rc.work_dir = str(work_dir)
    rc.provider_order = [seed.provider]
    rc.provider_allow_fallbacks = False
    rc.capture_api_requests = True
    rc.memory_seed = None
    rc.run_as_user = agent_user.name if agent_user else None
    rc.codex_prompt_stdin = cfg.isolation.hide_process_args
    rc.work_dir_hint = False
    # exactly the seed's sampling (None = was not sent), on every request and the prefill
    seed_sampling = seed.sampling()
    rc.use_recommended_sampling = False
    rc.reasoning_effort = seed_sampling["reasoning_effort"]
    rc.temperature, rc.top_p = seed_sampling["temperature"], seed_sampling["top_p"]
    rc.codex_config_overrides = [o for o in rc.codex_config_overrides
                                 if str(o).partition("=")[0].strip() != "model_reasoning_effort"]
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    splice = BranchSplice(seed, prefix, str(work_dir), api_key, arm.get("history_replace"),
                          max_attempts=bc.max_step_attempts)
    shadow = ShadowGit(work_dir=work_dir, git_dir=run_dir / ".shadow_git")
    shadow.init()
    shadow.commit_baseline()
    state = StateManager(work_dir=work_dir, shadow_git=shadow)
    result = await run_session(
        SessionConfig(session_index=1, prompt=RESUME_MARKER), rc, run_dir / "session_01", state,
        resume_session_id=sid, cwd_override=str(work_dir), proxy_intercept=splice,
        step_offset=seed.branch_step_id - 1, prefix_steps=seed.prefix_steps,
        extra_env=None if agent_user else {"CODEX_HOME": str(codex_home)})

    # 4. run-level outputs a normal pipeline run has
    shadow.end_session(1)
    (run_dir / "full_diff.patch").write_text(_full_diff(seed.run_dir / ".shadow_git", work_dir) or "# No changes\n")
    seed_log = seed.run_dir / "state_changelog.jsonl"
    prior = [l for l in (seed_log.read_text().splitlines() if seed_log.exists() else [])
             if l.strip() and json.loads(l).get("step_id", 0) < seed.branch_step_id]
    state.save_changelog(run_dir / "_branch_changelog.jsonl")
    id_map_path = run_dir / "session_01" / "step_id_map.json"
    id_map = {int(a): b for a, b in json.loads(id_map_path.read_text()).items()} if id_map_path.exists() else {}
    new = []
    for line in (run_dir / "_branch_changelog.jsonl").read_text().splitlines():
        if line.strip():   # session step ids -> full-trajectory step ids
            w = json.loads(line)
            w["step_id"] = id_map.get(w.get("step_id", 0), w.get("step_id", 0) + seed.branch_step_id - 1)
            new.append(json.dumps(w))
    (run_dir / "state_changelog.jsonl").write_text("".join(l + "\n" for l in prior + new if l.strip()))
    (run_dir / "_branch_changelog.jsonl").unlink()
    seed_cfg = yaml.safe_load((seed.run_dir / "config.yaml").read_text())
    (run_dir / "config.yaml").write_text(yaml.safe_dump(seed_cfg, sort_keys=False))
    meta = _build_run_meta(rc, run_dir.name, [result], state)
    traj = json.loads((run_dir / "session_01" / "trajectory.json").read_text())
    meta.update(total_steps=len(traj.get("steps") or []), branch=True)
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (run_dir / "branch_meta.json").write_text(json.dumps({
        "seed_run": str(seed.run_dir.resolve()), "branch_request": seed.k, "branch_step_id": seed.branch_step_id,
        "reset_tag": seed.reset_tag, "arm": arm_name, "rep": rep, "arm_spec": arm,
        "prefix": prefix, "original_step_reasoning": seed.original_reasoning,
        "model": seed.config.model, "provider": seed.provider, "support_status": seed.spec.providers[seed.provider],
        "prefix_steps": len(seed.prefix_steps), "template_fidelity": fidelity, **splice.meta,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, default=str))


__all__ = ["BranchUnsupportedError", "Seed", "BranchSplice", "run_branch", "RESUME_MARKER"]

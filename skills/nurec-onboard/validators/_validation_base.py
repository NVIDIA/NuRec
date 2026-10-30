# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for step-boundary ``*_validator.py`` scripts."""

from __future__ import annotations

import re
import sys

from math import isfinite
from pathlib import Path
from typing import Any, Callable, TypeGuard

import yaml


# Identity of the validate_result.yaml sidecar, shared with its writer
# (ncore_validation/validate.py). Bump the version on any breaking change to
# ValidateResult; the validator rejects files with a wrong version so stale
# captures cannot be re-used after a schema change.
VALIDATE_RESULT_SCHEMA_VERSION = 1

# The validator checks the exact value to confirm the file was produced by
# validate.py (not an arbitrary YAML the agent could fabricate).
VALIDATE_RESULT_PROGRAM = "ncore_validation/validate.py"


class ValidationError(Exception):
    """Validator failure carrying one diagnostic line per violated invariant."""

    def __init__(self, diagnostics: list[str]) -> None:
        """Create a validation error from user-facing diagnostics.

        Args:
            diagnostics: Diagnostic lines describing violated invariants.

        Returns:
            None.
        """
        self.diagnostics = list(diagnostics)
        super().__init__("\n".join(self.diagnostics))


def load_yaml(path: str | Path) -> Any:
    """Read YAML or raise :class:`ValidationError` with a user-facing error.

    Args:
        path: YAML file path to read.

    Returns:
        Parsed YAML document.

    Raises:
        ValidationError: If the path is missing, not a regular file,
            unreadable, unparsable, or empty.
    """
    p = Path(path)
    if not p.exists():
        raise ValidationError([f"input file does not exist: {p}"])
    if not p.is_file():
        raise ValidationError([f"input path is not a regular file: {p}"])
    try:
        text = p.read_text()
    except OSError as exc:
        raise ValidationError([f"failed to read {p}: {exc}"]) from exc
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValidationError([f"failed to parse {p} as YAML: {exc}"]) from exc
    if doc is None:
        raise ValidationError([f"{p} is empty or contains only comments"])
    return doc


# Placeholder markers surfaced by name in validator diagnostics.
_PLACEHOLDER_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bFILL\s+IN\b", re.IGNORECASE), "FILL IN marker"),
    (re.compile(r"\bTODO\b", re.IGNORECASE), "TODO marker"),
    (re.compile(r"\bTBD\b", re.IGNORECASE), "TBD marker"),
    (re.compile(r"\bXXX\b", re.IGNORECASE), "XXX marker"),
    (re.compile(r"\[\s*\.\.\.\s*\]"), "'[ ... ]' template marker"),
)


def placeholder_label(text: str) -> str | None:
    """Return the diagnostic label for the first placeholder in ``text``, or ``None``."""
    for pat, label in _PLACEHOLDER_PATTERNS:
        if pat.search(text):
            return label
    return None


def is_blank_or_placeholder(value: Any) -> bool:
    """Return True for ``None``, blank strings, or placeholder strings."""
    if value is None:
        return True
    if isinstance(value, str):
        if not value.strip():
            return True
        return placeholder_label(value) is not None
    return False


def is_empty_container(value: Any) -> bool:
    """Return True for ``None`` or empty list/tuple/dict/string values."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, str)) and len(value) == 0:
        return True
    return False


_STEP_RUN_REQUIRED_FIELDS = (
    "attempt",
    "status",
    "started_at",
    "ended_at",
    "duration_sec",
)


def is_number(value: Any) -> TypeGuard[int | float]:
    """Return True for non-bool int/float values.

    ``bool`` is a subclass of ``int`` in Python, so callers wanting "a real
    numeric value" must exclude bools explicitly. Centralizing the rule in
    this helper keeps every numeric type check in ``validators/`` consistent.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_finite_number(value: Any) -> TypeGuard[int | float]:
    """Return True for non-bool int/float values carrying a usable magnitude.

    YAML resolves ``.nan``, ``.inf``, and ``-.inf`` to floats, so a field that
    must hold a real measurement needs this rather than :func:`is_number`:
    a metric head that failed to compute reports NaN, which is a non-value
    wearing a numeric type.
    """
    return is_number(value) and isfinite(value)


def is_int(value: Any) -> TypeGuard[int]:
    """Return True for non-bool int values.

    Same motivation as :func:`is_number`: ``isinstance(value, int)`` alone
    accepts ``True``/``False`` because ``bool`` subclasses ``int``. Use
    ``is_int`` whenever a field must be an integer (e.g. ``exit_code``,
    ``iterations``, ``attempt``).
    """
    return isinstance(value, int) and not isinstance(value, bool)


def validate_step_run_entry(
    prefix: str,
    entry: Any,
    *,
    valid_statuses: tuple[str, ...] | None = None,
) -> list[str]:
    """Validate one lightweight step run stats entry.

    Args:
        prefix: User-facing diagnostic path for the run entry.
        entry: Parsed run entry.
        valid_statuses: Optional whitelist for ``status``. When provided,
            ``status`` must be one of these exact string values.

    Returns:
        Diagnostics for missing or malformed attempt, status, or timing
        fields.
    """
    if not isinstance(entry, dict):
        return [f"{prefix} must be a mapping, got {type(entry).__name__}"]

    diagnostics: list[str] = []
    for field in _STEP_RUN_REQUIRED_FIELDS:
        # Treat explicit ``null`` the same as a missing key: both convey no
        # information, and the per-field shape checks below skip ``None``
        # short-circuit, so without this the explicit-null case would slip
        # through with no diagnostic at all.
        if field not in entry or entry[field] is None:
            diagnostics.append(f"{prefix}.{field} is missing")
    for field in ("status", "started_at", "ended_at"):
        value = entry.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            diagnostics.append(f"{prefix}.{field} must be a non-blank string")
    if valid_statuses is not None:
        status = entry.get("status")
        if isinstance(status, str) and status.strip() and status not in valid_statuses:
            diagnostics.append(f"{prefix}.status={status!r} must be one of {list(valid_statuses)!r}")
    attempt = entry.get("attempt")
    if attempt is not None and (not is_int(attempt) or attempt < 1):
        diagnostics.append(f"{prefix}.attempt must be an integer >= 1")
    duration = entry.get("duration_sec")
    if duration is not None and (not is_number(duration) or duration < 0):
        diagnostics.append(f"{prefix}.duration_sec must be numeric and >= 0")
    return diagnostics


def blank_or_placeholder_diagnostic(path: str, value: Any) -> str | None:
    """Return a standard blank/placeholder diagnostic, or ``None``."""
    if not is_blank_or_placeholder(value):
        return None
    label = (placeholder_label(value) if isinstance(value, str) else None) or "blank"
    return f"{path} is {label}"


def numeric_vec_diagnostics(
    path: str,
    value: Any,
    *,
    length: int = 3,
    hint: str = "",
) -> list[str]:
    """Validate a fixed-length list of numbers and return diagnostics.

    ``hint`` is appended (with a leading space) to the "must be a
    N-element list" diagnostic when more context is useful (e.g.
    ``"(meters in rig frame)"``).
    """
    if not isinstance(value, (list, tuple)) or len(value) != length:
        suffix = f" {hint}" if hint else ""
        got_len = len(value) if hasattr(value, "__len__") else "?"
        return [f"{path} must be a {length}-element list{suffix}; got {type(value).__name__} of length {got_len}"]
    return [f"{path}[{i}] must be numeric, got {type(v).__name__}" for i, v in enumerate(value) if not is_number(v)]


def string_field_diagnostic(
    path: str,
    value: Any,
    *,
    blank_suffix: str = "",
) -> str | None:
    """Return one diagnostic for a required string field, or ``None``.

    Covers the three failure modes for a string field: missing/blank,
    non-string, and placeholder content. ``blank_suffix`` is appended to
    the missing/blank diagnostic when a longer hint is needed (e.g. how
    to fill the field when the input declares no source).
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        msg = f"{path} is missing or blank"
        if blank_suffix:
            msg = f"{msg} {blank_suffix}"
        return msg
    if not isinstance(value, str):
        return f"{path} must be a string, got {type(value).__name__}"
    hit = placeholder_label(value)
    if hit is not None:
        return f"{path} still contains a placeholder ({hit})"
    return None


def get_path(doc: Any, dotted_path: str) -> Any:
    """Return ``doc[a][b][c]...`` for ``dotted_path = "a.b.c"``, or ``None``."""
    cur: Any = doc
    for part in dotted_path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def as_dict(value: Any) -> dict[str, Any]:
    """Return ``value`` if it is a ``dict``, else an empty ``dict``.

    Lets callers chain ``.get(...)`` reads on a block that was loaded from
    untrusted YAML without repeating an ``isinstance`` guard at every read.
    The wrong-type case is intentionally silent: when a caller also needs to
    surface a wrong-type diagnostic, that belongs in shape validation, not in
    this helper. This is a type-narrowing fallback, not a coercion -- non-dict
    values are replaced with ``{}``, not recursively converted.
    """
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    """Return ``value`` if it is a ``list``, else an empty ``list``.

    Same intent as ``as_dict`` for list-shaped blocks: lets callers iterate or
    index without an ``isinstance`` guard at every site. The wrong-type case
    is intentionally silent; surface wrong-type diagnostics in shape
    validation, not here. This is a type-narrowing fallback, not a coercion --
    non-list values are replaced with ``[]``, not wrapped in a single-element
    list.
    """
    return value if isinstance(value, list) else []


_CONVERTER_ISSUE_FIELDS: tuple[str, ...] = (
    "error_message",
    "demotion_attempted",
    "demotion_criteria_failed",
    "iterations_before_giving_up",
)


def validate_converter_issue_entries(doc: dict[str, Any], path: str = "unresolved_issues.converter") -> list[str]:
    """Validate structured converter-issue entries at ``path``.

    Args:
        doc: Parsed diagnosis or workflow-summary document.
        path: Dot-separated path to the converter issue list.

    Returns:
        Diagnostics for malformed issue entries.
    """
    diagnostics: list[str] = []
    converter = get_path(doc, path)
    if converter is None:
        return diagnostics
    if not isinstance(converter, list):
        return [f"'{path}' must be a list, got {type(converter).__name__}"]
    for i, entry in enumerate(converter):
        prefix = f"{path}[{i}]"
        if not isinstance(entry, dict):
            diagnostics.append(f"{prefix} must be a mapping, got {type(entry).__name__}")
            continue
        for field in _CONVERTER_ISSUE_FIELDS:
            if field not in entry:
                diagnostics.append(f"{prefix}: {field!r} is missing")
        em = entry.get("error_message")
        if em is not None:
            diag = blank_or_placeholder_diagnostic(f"{prefix}.error_message", em)
            if diag is not None:
                diagnostics.append(f"{diag} (must be the exact line from validate.py output)")
        if "demotion_attempted" in entry and not isinstance(entry["demotion_attempted"], bool):
            diagnostics.append(
                f"{prefix}.demotion_attempted must be a bool, got {type(entry['demotion_attempted']).__name__}"
            )
        if "demotion_criteria_failed" in entry:
            crit = entry["demotion_criteria_failed"]
            if not isinstance(crit, list):
                diagnostics.append(f"{prefix}.demotion_criteria_failed must be a list, got {type(crit).__name__}")
            else:
                for j, c in enumerate(crit):
                    if not isinstance(c, str) or not c.strip():
                        diagnostics.append(f"{prefix}.demotion_criteria_failed[{j}] must be a non-blank string")
        if "iterations_before_giving_up" in entry and not is_int(entry["iterations_before_giving_up"]):
            diagnostics.append(
                f"{prefix}.iterations_before_giving_up must be an int, got "
                f"{type(entry['iterations_before_giving_up']).__name__}"
            )
    return diagnostics


def _messages_from_entries(entries: Any) -> list[str]:
    """Return string ``message`` fields from a list of mapping entries."""
    messages: list[str] = []
    for entry in as_list(entries):
        if isinstance(entry, dict):
            msg = entry.get("message")
            if isinstance(msg, str):
                messages.append(msg)
    return messages


def _entries_have_well_formed_messages(entries: Any) -> bool:
    """Return True when ``entries`` is a list of mappings with non-blank string ``message`` fields.

    Without this per-entry check, a sidecar with
    ``failed_checks: [{}, {"foo": "bar"}]`` would satisfy the top-level
    list-type check but round-trip through :func:`_messages_from_entries`
    as an empty list, hiding failures from the step-validator cross-checks.

    Args:
        entries: Value of the ``failed_checks`` or ``warnings`` field from
            a parsed sidecar.

    Returns:
        True when every entry is a mapping with a non-blank string
        ``message``; False otherwise (including non-list inputs).
    """
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        message = entry.get("message")
        if not isinstance(message, str) or not message.strip():
            return False
    return True


def parse_validate_result(result_path: str | Path) -> dict[str, Any]:
    """Parse validate.py's ``validate_result.yaml`` sidecar.

    Args:
        result_path: Path to the ``validate_result.yaml`` file referenced
            by the diagnosis's ``validate_run.result_path``.

    Returns:
        A dict with five keys consumed by :func:`evaluate_validate_run`
        and the anti-lying invariants:

        * ``schema_valid``: ``True`` only when the file declares the
          expected :data:`VALIDATE_RESULT_SCHEMA_VERSION` and
          :data:`VALIDATE_RESULT_PROGRAM` **and** all required typed
          fields are present and well-typed: ``passed: bool``,
          ``exit_code: int``, and ``failed_checks`` / ``warnings`` each
          a list whose every entry is a mapping with a non-blank string
          ``message``. A bare ``{schema_version, program}`` stub, a
          stub with mis-typed fields, or a sidecar carrying malformed
          entries (missing ``message``, non-string ``message``, blank
          ``message``) registers as ``False`` so an agent cannot
          fabricate a minimal sidecar to bypass the honesty checks.
        * ``passed``: Mirrors validate.py's typed ``passed`` field
          (``True`` / ``False`` / ``None`` when missing or mis-typed).
        * ``result_exit_code``: Mirrors the sidecar's ``exit_code``
          field as an int, or ``None`` when missing / mis-typed.
          Cross-checked against the diagnosis's
          ``validate_run.exit_code`` by the step validators.
        * ``failed_check_messages``: List of verbatim ``message``
          strings from ``failed_checks[*]``. The downstream validator
          enforces verbatim quoting of each against the diagnosis YAML.
        * ``warning_messages``: List of verbatim ``message`` strings
          from ``warnings[*]`` (not currently enforced, but exposed for
          future use).

    Raises:
        ValidationError: When the file is missing, unreadable, or not
            a valid YAML mapping. Wrong schema_version / program / typed
            fields are surfaced via ``schema_valid=False`` in the return
            dict, not an exception, so the caller can still apply the
            exit-code invariants when the file is structurally readable
            but misidentified.
    """
    p = Path(result_path)
    if not p.exists():
        raise ValidationError([f"validate_run.result_path does not exist: {p}"])
    if not p.is_file():
        raise ValidationError([f"validate_run.result_path is not a regular file: {p}"])
    try:
        text = p.read_text(errors="replace")
    except OSError as exc:
        raise ValidationError([f"failed to read validate_run.result_path {p}: {exc}"]) from exc
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValidationError([f"failed to parse validate_run.result_path {p} as YAML: {exc}"]) from exc
    if not isinstance(doc, dict):
        raise ValidationError([f"validate_run.result_path {p} must contain a YAML mapping, got {type(doc).__name__}"])

    receipt_ok = (
        doc.get("schema_version") == VALIDATE_RESULT_SCHEMA_VERSION and doc.get("program") == VALIDATE_RESULT_PROGRAM
    )
    passed_field = doc.get("passed")
    exit_code_field = doc.get("exit_code")
    failed_checks_field = doc.get("failed_checks")
    warnings_field = doc.get("warnings")
    typed_fields_ok = (
        isinstance(passed_field, bool)
        and is_int(exit_code_field)
        and _entries_have_well_formed_messages(failed_checks_field)
        and _entries_have_well_formed_messages(warnings_field)
    )
    schema_valid = receipt_ok and typed_fields_ok

    return {
        "schema_valid": schema_valid,
        "passed": passed_field if isinstance(passed_field, bool) else None,
        "result_exit_code": exit_code_field if is_int(exit_code_field) else None,
        "failed_check_messages": _messages_from_entries(failed_checks_field),
        "warning_messages": _messages_from_entries(warnings_field),
    }


def parse_absolute_validate_result(result_path: Any, diagnostics: list[str]) -> dict[str, Any] | None:
    """Parse the sidecar at ``result_path`` if it is an absolute path.

    Callers' shape checks report relative or blank paths; skipping them here
    keeps the validator's working directory from resolving an unrelated
    same-named file.

    Args:
        result_path: The ``result_path`` value from the step's YAML, of any type.
        diagnostics: List the parse errors are appended to.

    Returns:
        The :func:`parse_validate_result` output, or None when the path is not
        an absolute path or the sidecar fails to parse.
    """
    if not (isinstance(result_path, str) and result_path.strip() and Path(result_path).is_absolute()):
        return None
    try:
        return parse_validate_result(result_path)
    except ValidationError as exc:
        diagnostics.extend(exc.diagnostics)
        return None


def validate_result_exit_code_diagnostic(
    prefix: str, parsed_result: dict[str, Any] | None, exit_code: Any
) -> str | None:
    """Return a diagnostic when reported and sidecar exit codes disagree."""
    if parsed_result is None:
        return None
    result_exit_code = parsed_result.get("result_exit_code")
    if is_int(exit_code) and is_int(result_exit_code) and exit_code != result_exit_code:
        return (
            f"{prefix}.exit_code={exit_code} disagrees with "
            f"validate_result.yaml exit_code={result_exit_code}; the "
            f"reported exit code MUST match the exit code validate.py "
            f"recorded in its sidecar"
        )
    return None


def validate_result_internal_consistency_diagnostic(
    parsed_result: dict[str, Any] | None,
) -> str | None:
    """Return a diagnostic when the sidecar's ``passed`` disagrees with ``exit_code``.

    validate.py writes the sidecar with ``passed = (exit_code == 0)`` as
    a single assignment, so any divergence in a captured sidecar is a
    tampering signal: a hand-edited or fabricated file is the only way
    the two fields can disagree. Both fields must already be present
    and well-typed (``schema_valid=True`` implies that); a missing or
    mis-typed field is reported separately via the schema-receipt
    check and this helper stays silent on those cases.

    Args:
        parsed_result: Output of :func:`parse_validate_result`, or
            ``None`` when the sidecar could not be loaded.

    Returns:
        A diagnostic string when ``passed`` and ``exit_code`` disagree,
        otherwise ``None``.
    """
    if parsed_result is None:
        return None
    passed = parsed_result.get("passed")
    result_exit_code = parsed_result.get("result_exit_code")
    if not isinstance(passed, bool) or not is_int(result_exit_code):
        return None
    expected_passed = result_exit_code == 0
    if passed != expected_passed:
        return (
            f"validate_result.yaml passed={passed!r} disagrees with "
            f"exit_code={result_exit_code} (validate.py writes passed "
            f"and exit_code together, so any mismatch indicates the "
            f"sidecar was edited or fabricated after the run)"
        )
    return None


def evaluate_validate_run(parsed_result: dict[str, Any] | None, exit_code: Any) -> tuple[bool, bool]:
    """Return ``(had_fail, schema_invalid)`` from parsed result + exit code.

    ``had_fail`` is True when ``exit_code`` is a non-zero int **or** the
    parsed result's typed ``passed`` field is ``False`` **or**
    ``failed_check_messages`` is non-empty. The exit-code branch fires
    even when ``parsed_result`` is ``None``, so a diagnosis claiming
    ``validation_passed=true`` against a non-zero exit code still gets
    caught when the result file is missing or unparseable. The
    ``passed`` and ``failed_check_messages`` branches are
    defense-in-depth duplicates: validate.py sets both consistently, so
    any disagreement among them is itself a signal that the file was
    tampered with.

    ``schema_invalid`` is True only when ``parsed_result`` is a dict and
    its ``schema_valid`` flag is False (wrong ``schema_version`` or
    ``program``). When ``parsed_result`` is ``None`` the missing-file
    diagnostic from :func:`parse_validate_result` already carries that
    signal, so we do not double-report.
    """
    parsed = as_dict(parsed_result)
    had_fail = (
        (is_int(exit_code) and exit_code != 0)
        or parsed.get("passed") is False
        or bool(parsed.get("failed_check_messages"))
    )
    schema_invalid = isinstance(parsed_result, dict) and not parsed_result.get("schema_valid")
    return had_fail, schema_invalid


def parse_cli_args(
    args: list[str],
    *,
    usage: str,
    allow_store: bool = False,
) -> tuple[str, str | None]:
    """Parse one positional path plus optional ``--store`` / ``--help``.

    Writes ``usage`` to stdout and raises ``SystemExit(0)`` on ``-h/--help``;
    raises ``ValueError`` on any other malformed invocation.
    """
    summary_path: str | None = None
    store_path: str | None = None
    it = iter(args)
    for tok in it:
        if tok in ("-h", "--help"):
            sys.stdout.write(usage)
            raise SystemExit(0)
        if tok == "--store" and allow_store:
            try:
                store_path = next(it)
            except StopIteration as exc:
                raise ValueError("--store requires a path argument") from exc
            continue
        if tok.startswith("-"):
            raise ValueError(f"unknown option {tok!r}")
        if summary_path is not None:
            raise ValueError("at most one positional argument is supported")
        summary_path = tok
    if summary_path is None:
        raise ValueError("missing positional argument")
    return summary_path, store_path


def run_validator(
    argv: list[str],
    validate_fn: Callable[[list[str]], None],
    *,
    usage: str,
    progname: str | None = None,
) -> int:
    """Run a validator with the common exit-code and stderr contract.

    Args:
        argv: Command-line argument vector, including program name.
        validate_fn: Validator callback that receives positional arguments.
        usage: Usage text printed for malformed invocations.
        progname: Optional program name for diagnostics.

    Returns:
        Process exit code: ``0`` for success, ``1`` for validation failure,
        and ``2`` for malformed input or unexpected errors.

    Raises:
        SystemExit: Re-raised if ``validate_fn`` intentionally exits.
    """
    name = progname if progname is not None else (argv[0] if argv else "validator")
    try:
        validate_fn(argv[1:])
    except ValidationError as exc:
        sys.stderr.write(f"{name}: found {len(exc.diagnostics)} problem(s):\n")
        for diag in exc.diagnostics:
            sys.stderr.write(f"  - {diag}\n")
        return 1
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - usage / arg errors funnel here
        sys.stderr.write(f"{name}: {exc}\n")
        sys.stderr.write(usage)
        return 2
    return 0


def step_validator_main(
    *,
    progname: str,
    usage: str,
    validate_fn: Callable[[str], Any],
    allow_store: bool = False,
    post_store_fn: Callable[[str, Any], None] | None = None,
) -> Callable[[list[str]], int]:
    """Build a one-positional-path CLI entry point for a step validator.

    Wraps ``validate_fn`` in the standard ``run_validator`` exit-code and
    stderr contract: ``--help`` prints ``usage`` and exits 0, missing or
    unknown arguments exit 2, ``ValidationError`` raised from
    ``validate_fn`` (or ``post_store_fn``) prints diagnostics to stderr
    and exits 1, and a clean run exits 0.

    Args:
        progname: Program name used in the ``run_validator`` stderr
            preamble.
        usage: ``--help`` text. Also shown on argument-parsing errors.
        validate_fn: Step-specific validator. Receives the positional
            YAML path and may return any value. Raise ``ValidationError``
            with a list of diagnostics on failure.
        allow_store: When ``True``, the CLI accepts an optional
            ``--store <path>`` flag in addition to the positional YAML.
        post_store_fn: Optional follow-up step run only when ``--store``
            was supplied and ``validate_fn`` succeeded. Called as
            ``post_store_fn(store_path, validate_fn_return_value)``.
            Any ``ValidationError`` it raises flows through the same
            exit-code contract. Ignored unless ``allow_store=True``.

    Returns:
        A ``main(argv)`` callable suitable for assignment to a module's
        ``main`` attribute and ``sys.exit(main(sys.argv))``.
    """

    def _validate(args: list[str]) -> None:
        path, store_path = parse_cli_args(args, usage=usage, allow_store=allow_store)
        result = validate_fn(path)
        if allow_store and store_path is not None and post_store_fn is not None:
            post_store_fn(store_path, result)

    def main(argv: list[str]) -> int:
        return run_validator(argv, _validate, usage=usage, progname=progname)

    return main

#!/usr/bin/env python3
"""Evaluate clinical-keyword recall for real consultation ASR transcripts."""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import html
import hashlib
import json
import math
import os
import re
import shutil
import ssl
import subprocess
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


AUDIO_MIME_TYPES = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
}
APPOINTMENT_SUFFIXES = (
    ".appointment.txt",
    ".appointment.json",
    ".record_appointment.json",
)
DEFAULT_DASHSCOPE_HTTP_BASE = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_DASHSCOPE_NATIVE_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/"
    "aigc/multimodal-generation/generation"
)
CATEGORIES = (
    "症狀", "診斷", "鑑別診斷", "藥物", "檢查", "檢查結果", "數值",
    "病程", "暴露", "病史", "敏感史", "社交史", "旅遊史", "職業",
    "治療", "處置", "危險警號", "其他",
)
KEYWORD_PROMPT_VERSION = 8
KEYWORD_VALIDATION_ATTEMPTS = 5
MANDARIN_ONLY_FRAGMENTS = ("這", "这")
DOSAGE_PATTERN = re.compile(
    r"(?:\d+\s*(?:mg|ml|毫克|毫升)|[零〇一二兩两三四五六七八九十百千]+\s*(?:毫克|毫升))",
    re.I,
)
KEYWORD_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Short diagnosis or consultation title.",
        },
        "keywords": {
            "type": "array",
            "minItems": 10,
            "maxItems": 16,
            "items": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Canonical clinical concept from the appointment.",
                    },
                    "category": {"type": "string"},
                    "accepted_forms": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 8,
                        "items": {"type": "string"},
                        "description": (
                            "Three to eight interchangeable Traditional Chinese "
                            "Cantonese, English, numeric, brand, treatment-role, or "
                            "abbreviation surface forms for the same atomic concept."
                        ),
                    },
                    "doctor_evidence": {
                        "type": "string",
                        "description": (
                            "Exact contiguous excerpt copied from the supplied "
                            "plain-text appointment."
                        ),
                    },
                },
                "required": [
                    "keyword", "category", "accepted_forms", "doctor_evidence"
                ],
            },
        },
    },
    "required": ["title", "keywords"],
}


@dataclass(frozen=True)
class Keyword:
    text: str
    category: str
    accepted_forms: tuple[str, ...]
    doctor_evidence: str


@dataclass(frozen=True)
class ASRModel:
    model_id: str
    repo: str
    backend: str


@dataclass(frozen=True)
class PipelineConfig:
    """All environment-backed settings needed by the zero-argument pipeline."""

    root: Path
    input_dir: Path
    output_dir: Path
    keyword_config: Path
    keyword_model: str
    asr_models: tuple[ASRModel, ...]
    gemini_api_key: str
    dashscope_api_key: str
    dashscope_http_base: str = DEFAULT_DASHSCOPE_HTTP_BASE
    dashscope_native_url: str = DEFAULT_DASHSCOPE_NATIVE_URL
    dashscope_proxy: str | None = None


ASR_MODELS = {
    "qwen3-asr-flash": ASRModel(
        "qwen3-asr-flash", "qwen3-asr-flash-2026-02-10", "dashscope"
    ),
    "qwen-audio-3.0-asr-flash": ASRModel(
        "qwen-audio-3.0-asr-flash",
        "qwen-audio-3.0-asr-flash",
        "dashscope-audio3",
    ),
}


@dataclass(frozen=True)
class Consultation:
    consultation_id: str
    title: str
    audio_path: Path
    record_path: Path
    record_text: str
    keywords: tuple[Keyword, ...]


def normalize_text(text: str) -> str:
    """Normalize formatting while preserving wording and Chinese-script choices."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in {"L", "N"})


def has_negation(text: str) -> bool:
    """Return whether a surface form explicitly expresses negation."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    compact = normalize_text(text)
    if compact in {"nkda", "nka", "nsnd", "nad", "nil", "none"}:
        return True
    if any(
        marker in normalized
        for marker in ("冇", "無", "没有", "沒有", "未", "不", "非", "否認")
    ):
        return True
    return bool(re.search(
        r"(?:^|\W)(?:no|not|without|negative|-ve|nil|none|den(?:y|ies|ied))"
        r"(?:\W|$)",
        normalized,
    ))


def mandarin_only_fragment(text: str) -> str:
    """Return a known non-Cantonese spoken fragment, if one is present."""
    return next((item for item in MANDARIN_ONLY_FRAGMENTS if item in text), "")


def flatten_json_text(value: object) -> str:
    """Return searchable plain text from a doctor-record JSON value."""
    parts: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            without_tags = re.sub(r"<[^>]+>", " ", html.unescape(item))
            parts.append(re.sub(r"\s+", " ", without_tags).strip())

    visit(value)
    return "\n".join(part for part in parts if part)


def load_appointment_text(path: Path) -> str:
    """Load either a plain-text appointment note or a structured JSON export."""
    if path.name.endswith(".txt"):
        text = path.read_text(encoding="utf-8-sig").strip()
    else:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        text = flatten_json_text(value).strip()
    if not text:
        raise ValueError(f"Appointment note contains no readable text: {path}")
    return text


def load_env(path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE entries without overriding exported variables."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(f"Invalid .env line {line_number}: expected KEY=VALUE")
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[:1] == value[-1:] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid .env variable name on line {line_number}")
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_setting_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def build_pipeline_config(
    script_dir: Path,
    *,
    root: Path | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    keyword_config: Path | None = None,
    asr_models: str | None = None,
) -> PipelineConfig:
    """Build and validate the production pipeline configuration from environment."""
    resolved_root = (root or script_dir).resolve()
    resolved_input = (
        input_dir.resolve()
        if input_dir
        else _resolve_setting_path(os.environ.get("INPUT_DIR", "input"), resolved_root)
    )
    resolved_output = (
        output_dir.resolve()
        if output_dir
        else _resolve_setting_path(
            os.environ.get("OUTPUT_DIR", "results"), resolved_root
        )
    )
    resolved_keyword_config = (
        keyword_config.resolve()
        if keyword_config
        else _resolve_setting_path(
            os.environ.get(
                "KEYWORD_CONFIG", str(resolved_output / "generated_keywords.json")
            ),
            resolved_root,
        )
    )
    keyword_model = (
        os.environ.get("KEYWORD_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or "gemini-2.5-flash"
    ).strip()
    if not keyword_model:
        raise ValueError("KEYWORD_MODEL must not be empty")
    selected_asr_models = tuple(parse_asr_models(
        asr_models
        or os.environ.get(
            "ASR_MODELS", "qwen3-asr-flash,qwen-audio-3.0-asr-flash"
        )
    ))
    return PipelineConfig(
        root=resolved_root,
        input_dir=resolved_input,
        output_dir=resolved_output,
        keyword_config=resolved_keyword_config,
        keyword_model=keyword_model,
        asr_models=selected_asr_models,
        gemini_api_key=os.environ.get("GEMINI_API_KEY", "").strip(),
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "").strip(),
        dashscope_http_base=os.environ.get(
            "DASHSCOPE_HTTP_BASE", DEFAULT_DASHSCOPE_HTTP_BASE
        ).strip(),
        dashscope_native_url=os.environ.get(
            "DASHSCOPE_NATIVE_URL", DEFAULT_DASHSCOPE_NATIVE_URL
        ).strip(),
        dashscope_proxy=os.environ.get("DASHSCOPE_PROXY", "").strip() or None,
    )


def tls_context() -> ssl.SSLContext:
    """Use certifi when available while always retaining TLS verification."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def discover_input_pairs(input_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Discover and strictly validate appointment-note/recording pairs."""
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}. Create it and add "
            "paired appointment notes and recordings."
        )
    note_paths: dict[str, Path] = {}
    for suffix in APPOINTMENT_SUFFIXES:
        for note_path in sorted(input_dir.glob(f"*{suffix}")):
            consultation_id = note_path.name.removesuffix(suffix)
            existing = note_paths.get(consultation_id)
            if existing is not None:
                raise ValueError(
                    f"Multiple appointment notes found for {consultation_id}: "
                    f"{existing.name}, {note_path.name}"
                )
            note_paths[consultation_id] = note_path

    audio_by_id: dict[str, list[Path]] = {}
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.casefold() in AUDIO_MIME_TYPES:
            audio_by_id.setdefault(path.stem, []).append(path)

    missing_notes = sorted(set(audio_by_id) - set(note_paths))
    if missing_notes:
        raise ValueError(
            "Recording(s) have no matching appointment note: "
            + ", ".join(missing_notes)
        )
    if not note_paths:
        raise ValueError(
            f"No appointment notes found in {input_dir}. Expected "
            "<consultation-id>.appointment.txt or "
            "<consultation-id>.record_appointment.json"
        )

    pairs: dict[str, tuple[Path, Path]] = {}
    for consultation_id, record_path in sorted(note_paths.items()):
        audio_paths = audio_by_id.get(consultation_id, [])
        if not audio_paths:
            raise FileNotFoundError(
                f"Appointment note has no matching recording: {record_path.name}"
            )
        if len(audio_paths) > 1:
            raise ValueError(
                f"Multiple recordings found for {consultation_id}: "
                + ", ".join(path.name for path in audio_paths)
            )
        if record_path.stat().st_size == 0 or audio_paths[0].stat().st_size == 0:
            raise ValueError(
                f"Empty appointment note or recording for {consultation_id}"
            )
        load_appointment_text(record_path)
        pairs[consultation_id] = (record_path, audio_paths[0])
    return pairs


def _keyword_prompt(
    consultation_id: str,
    appointment_text: str,
    validation_feedback: str = "",
) -> str:
    categories = ", ".join(CATEGORIES)
    feedback = (
        "\nPrevious responses failed validation. Regenerate the entire response and "
        "fix EVERY listed error; do not repeat an earlier error:\n"
        f"{validation_feedback}\n"
        if validation_feedback else ""
    )
    return f"""You are creating a reproducible clinical-keyword recall benchmark for
Hong Kong Cantonese ASR.

Consultation ID: {consultation_id}

Select 10-16 clinically important concepts explicitly supported by the appointment
text below. Never return more than 16 concepts. Choose concepts that a patient or
doctor is likely to say aloud and that matter when judging a medical ASR transcript.

Make every keyword an atomic scoring concept. Split separate symptoms, and split a
symptom, its count, and its duration into separate concepts (for example, `runny nose`
and `nasal congestion`, never `runny nose/nasal congestion`; `diarrhoea` and `3 times`,
not `diarrhoea 3 times today`). Always split a medication and its dose into separate
concepts so that ordinary names for the medicine remain interchangeable.

For each concept:
- keyword: a short canonical keyword in natural, daily-use Hong Kong Cantonese written
  in Traditional Chinese. DO NOT copy the English appointment wording into this field.
  Use the phrase people actually say in consultation, for example Gastroenteritis ->
  腸胃炎, diarrhoea -> 肚屙, sore throat -> 喉嚨痛, no fever -> 冇發燒, yesterday ->
  琴日, and paracetamol -> 撲熱息痛. Preserve clinically important negation, number,
  dose, or duration. Avoid formal Mainland Chinese and textbook-style translations.
- category: exactly one of: {categories}.
- accepted_forms: return 3-8 concise, independently matchable ways to express exactly
  this concept in a Hong Kong Cantonese medical consultation. The first accepted form
  MUST exactly equal `keyword`. Balance the group with common daily Cantonese variants
  followed by the appointment's English term, abbreviation, brand, or numeric form.
  Chinese forms must be natural Hong Kong Cantonese, not Standard Mandarin or
  Mandarin-only written variants (for example, use 呢兩日, never 這兩日).
  For example, the medicine concept paracetamol may use 撲熱息痛, Paracetamol, Panadol,
  止痛藥, and 退燒藥 when those treatment roles match this appointment. Include both 冇
  and 無 variants for a negative fact, and preserve negation in EVERY form. Preserve any
  clinically important number or duration in every form belonging to that numeric or
  duration concept. Every form must be a complete alternative for the atomic concept,
  not one component of a combined concept and not merely a word taken from a longer
  phrase. Do not reuse a form in another keyword group. Keep forms short enough for
  strict substring matching; do not add loose or clinically different synonyms.
- doctor_evidence: copy an exact contiguous excerpt from APPOINTMENT TEXT. Do not
  paraphrase it.

Prioritize symptoms, duration/onset, clinically meaningful numbers, exposures,
diagnoses, tests, medicines, treatments, disposition, and red flags. Preserve
negation. Prefer consultation-specific facts over routine normal background history
such as NKDA/NSND unless that history is important to this encounter. Treat `SL` as
sick leave, never as symptom length; `SL x 1/7` means one day of sick leave. Avoid
duplicating a broad umbrella concept when a more precise diagnosis already represents
it. Exclude patient identity, clinician names, fees, payment, delivery logistics,
greetings, diagnosis codes, record-template boilerplate, and generic conversation.
Use category 職業 for an occupation and category 處置 for sick leave.

APPOINTMENT TEXT
----------------
{appointment_text}
{feedback}
"""


def _extract_response_text(response: object) -> str:
    if not isinstance(response, dict):
        raise ValueError("Gemini returned a non-object response")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        feedback = response.get("promptFeedback", {})
        raise ValueError(f"Gemini returned no candidates: {feedback}")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise ValueError("Gemini returned an invalid candidate")
    content = candidate.get("content", {})
    parts = content.get("parts", []) if isinstance(content, dict) else []
    texts = [
        str(part["text"])
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    if not texts:
        raise ValueError(
            f"Gemini candidate contained no text; finishReason="
            f"{candidate.get('finishReason', 'unknown')}"
        )
    return "".join(texts)


def gemini_generate_keywords(
    api_key: str,
    model: str,
    consultation_id: str,
    appointment_text: str,
    validation_feedback: str = "",
    timeout_seconds: int = 180,
) -> dict[str, object]:
    """Ask Gemini for structured appointment-grounded clinical keywords."""
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": _keyword_prompt(
                        consultation_id, appointment_text, validation_feedback
                    )},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseSchema": KEYWORD_RESPONSE_SCHEMA,
        },
    }
    encoded_model = urllib.parse.quote(model, safe="")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{encoded_model}:generateContent"
    )
    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    retryable_codes = {429, 500, 502, 503, 504}
    for attempt in range(3):
        request = urllib.request.Request(
            url,
            data=request_body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout_seconds, context=tls_context()
            ) as response_handle:
                response = json.loads(response_handle.read().decode("utf-8"))
            generated = json.loads(_extract_response_text(response))
            if not isinstance(generated, dict):
                raise ValueError("Gemini structured output was not a JSON object")
            return generated
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            if exc.code in retryable_codes and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(
                f"Gemini API HTTP {exc.code} for {consultation_id}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(
                f"Gemini API network error for {consultation_id}: {exc.reason}"
            ) from exc
    raise RuntimeError(f"Gemini API failed for {consultation_id}")


def validate_generated_definition(
    consultation_id: str,
    definition: object,
    appointment_text: str,
) -> dict[str, object]:
    if not isinstance(definition, dict):
        raise ValueError(f"Generated definition for {consultation_id} is not an object")
    title = str(definition.get("title", "")).strip()
    raw_keywords = definition.get("keywords")
    if not title or not isinstance(raw_keywords, list) or not 10 <= len(raw_keywords) <= 16:
        raise ValueError(
            f"{consultation_id} requires a title and 10-16 generated keywords"
        )
    cleaned: list[dict[str, object]] = []
    seen: set[str] = set()
    form_owners: dict[str, str] = {}
    normalized_appointment = normalize_text(appointment_text)
    for raw_keyword in raw_keywords:
        if not isinstance(raw_keyword, dict):
            raise ValueError(f"Invalid generated keyword in {consultation_id}")
        keyword = str(raw_keyword.get("keyword", "")).strip()
        category = str(raw_keyword.get("category", "")).strip()
        evidence = str(raw_keyword.get("doctor_evidence", "")).strip()
        raw_accepted_forms = [
            str(item).strip()
            for item in raw_keyword.get("accepted_forms", [])
            if str(item).strip()
        ]
        accepted_forms: list[str] = []
        normalized_forms: set[str] = set()
        for form in raw_accepted_forms:
            normalized_form = normalize_text(form)
            if not normalized_form or normalized_form in normalized_forms:
                raise ValueError(
                    f"{keyword!r} has an empty or duplicate accepted form "
                    f"{form!r} in {consultation_id}"
                )
            normalized_forms.add(normalized_form)
            accepted_forms.append(form)
        normalized_keyword = normalize_text(keyword)
        if not keyword or normalized_keyword in seen:
            raise ValueError(
                f"Missing or duplicate generated keyword in {consultation_id}: "
                f"{keyword!r}"
            )
        if category not in CATEGORIES:
            raise ValueError(
                f"Invalid category {category!r} in {consultation_id}"
            )
        if not 3 <= len(accepted_forms) <= 8:
            raise ValueError(
                f"{keyword!r} requires 3-8 accepted forms in {consultation_id}"
            )
        if normalize_text(accepted_forms[0]) != normalized_keyword:
            raise ValueError(
                f"The first accepted form must equal keyword {keyword!r} in "
                f"{consultation_id}"
            )
        for form in accepted_forms:
            mandarin_fragment = mandarin_only_fragment(form)
            if mandarin_fragment:
                raise ValueError(
                    f"{keyword!r} uses non-Hong-Kong-Cantonese form {form!r} "
                    f"({mandarin_fragment!r}) in {consultation_id}"
                )
        if has_negation(keyword):
            non_negative_forms = [
                form for form in accepted_forms if not has_negation(form)
            ]
            if non_negative_forms:
                raise ValueError(
                    f"Negative keyword {keyword!r} has forms without negation in "
                    f"{consultation_id}: {', '.join(non_negative_forms)}"
                )
        for form in accepted_forms:
            normalized_form = normalize_text(form)
            owner = form_owners.get(normalized_form)
            if owner is not None and owner != keyword:
                raise ValueError(
                    f"Accepted form {form!r} is shared by keywords {owner!r} and "
                    f"{keyword!r} in {consultation_id}"
                )
            form_owners[normalized_form] = keyword
        if not evidence or normalize_text(evidence) not in normalized_appointment:
            raise ValueError(
                f"Evidence {evidence!r} for {keyword!r} is not an exact appointment "
                f"excerpt in {consultation_id}"
            )
        seen.add(normalized_keyword)
        cleaned.append({
            "keyword": keyword,
            "category": category,
            "accepted_forms": accepted_forms,
            "doctor_evidence": evidence,
        })
    for item in cleaned:
        keyword = str(item["keyword"])
        if not DOSAGE_PATTERN.search(keyword):
            continue
        normalized_keyword = normalize_text(keyword)
        embedded_medicine = next(
            (
                str(other["keyword"])
                for other in cleaned
                if other is not item
                and normalize_text(str(other["keyword"])) in normalized_keyword
                and not DOSAGE_PATTERN.search(str(other["keyword"]))
            ),
            "",
        )
        if embedded_medicine:
            raise ValueError(
                f"Medication and dose must be separate atomic keywords in "
                f"{consultation_id}: replace {keyword!r} with a dose-only keyword; "
                f"the medicine is already represented by {embedded_medicine!r}"
            )
    return {"title": title, "keywords": cleaned}


def _write_json(path: Path, value: object) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )


def _write_text_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Replace a text artifact atomically so interrupted runs keep the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def generate_keyword_config(
    input_dir: Path,
    config_path: Path,
    api_key: str,
    model: str,
    refresh: bool = False,
) -> dict[str, object]:
    """Generate or reuse hash-keyed keyword definitions for every input pair."""
    pairs = discover_input_pairs(input_dir)
    cached: dict[str, object] = {}
    if config_path.is_file() and not refresh:
        loaded = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if isinstance(loaded, dict) and isinstance(loaded.get("consultations"), dict):
            cached = loaded["consultations"]

    generated_consultations: dict[str, object] = {}
    output: dict[str, object] = {
        "schema_version": 1,
        "generator": {
            "provider": "Google Gemini",
            "model": model,
            "prompt_version": KEYWORD_PROMPT_VERSION,
        },
        "consultations": generated_consultations,
    }
    for consultation_id, (record_path, _audio_path) in pairs.items():
        record_hash = sha256_file(record_path)
        cached_entry = cached.get(consultation_id)
        if (
            not refresh
            and isinstance(cached_entry, dict)
            and cached_entry.get("generator_model") == model
            and cached_entry.get("prompt_version") == KEYWORD_PROMPT_VERSION
            and cached_entry.get("appointment_sha256") == record_hash
        ):
            generated_consultations[consultation_id] = cached_entry
            continue

        appointment_text = load_appointment_text(record_path)
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required to generate missing or stale keywords"
            )
        print(f"Generating keywords with {model}: {consultation_id}")
        validation_feedback = ""
        validation_errors: list[str] = []
        cleaned: dict[str, object] | None = None
        for validation_attempt in range(KEYWORD_VALIDATION_ATTEMPTS):
            definition = gemini_generate_keywords(
                api_key,
                model,
                consultation_id,
                appointment_text,
                validation_feedback,
            )
            try:
                cleaned = validate_generated_definition(
                    consultation_id, definition, appointment_text
                )
                break
            except ValueError as exc:
                error = str(exc)
                if error not in validation_errors:
                    validation_errors.append(error)
                validation_feedback = "\n".join(
                    f"- {item}" for item in validation_errors
                )
                if validation_attempt == KEYWORD_VALIDATION_ATTEMPTS - 1:
                    raise ValueError(
                        "Gemini keyword output failed validation "
                        f"{KEYWORD_VALIDATION_ATTEMPTS} times. Last error: {exc}"
                    ) from exc
                print(
                    f"Retrying invalid keyword output for {consultation_id}: {exc}"
                )
        if cleaned is None:  # pragma: no cover - defensive invariant
            raise RuntimeError(f"No valid keywords generated for {consultation_id}")
        generated_consultations[consultation_id] = {
            **cleaned,
            "generator_model": model,
            "prompt_version": KEYWORD_PROMPT_VERSION,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "appointment_file": record_path.name,
            "appointment_sha256": record_hash,
        }
        _write_json(config_path, output)
    _write_json(config_path, output)
    return output


def parse_asr_models(value: str) -> list[ASRModel]:
    model_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not model_ids:
        raise ValueError("ASR_MODELS must contain at least one model ID")
    unknown = [model_id for model_id in model_ids if model_id not in ASR_MODELS]
    if unknown:
        raise ValueError(
            "Unknown ASR model(s): " + ", ".join(unknown)
            + "; choices: " + ", ".join(ASR_MODELS)
        )
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("ASR_MODELS contains duplicate model IDs")
    return [ASR_MODELS[model_id] for model_id in model_ids]


def _urllib_opener(proxy: str | None) -> urllib.request.OpenerDirector:
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.HTTPSHandler(context=tls_context())
    ]
    if proxy:
        handlers.append(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    else:
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


def _post_json(
    url: str,
    payload: object,
    headers: dict[str, str],
    description: str,
    timeout_seconds: int,
    proxy: str | None = None,
) -> dict[str, object]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    retryable_codes = {429, 500, 502, 503, 504}
    opener = _urllib_opener(proxy)
    for attempt in range(3):
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with opener.open(request, timeout=timeout_seconds) as response_handle:
                response = json.loads(response_handle.read().decode("utf-8"))
            if not isinstance(response, dict):
                raise ValueError(f"{description} returned a non-object response")
            return response
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            if exc.code in retryable_codes and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(
                f"{description} HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(
                f"{description} network error: {exc.reason}"
            ) from exc
    raise RuntimeError(f"{description} failed")


def dashscope_transcribe(
    api_key: str,
    model: ASRModel,
    audio_path: Path,
    http_base: str = DEFAULT_DASHSCOPE_HTTP_BASE,
    native_url: str = DEFAULT_DASHSCOPE_NATIVE_URL,
    proxy: str | None = None,
    timeout_seconds: int = 300,
) -> str:
    """Transcribe one recording using the ASR Playground DashScope contracts."""
    content_type = AUDIO_MIME_TYPES.get(audio_path.suffix.casefold())
    if not content_type:
        raise ValueError(f"Unsupported audio type: {audio_path.suffix}")
    encoded_audio = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    data_url = f"data:{content_type};base64,{encoded_audio}"
    headers = {"Authorization": f"Bearer {api_key}"}
    description = f"DashScope {model.model_id} for {audio_path.name}"
    if model.backend == "dashscope":
        response = _post_json(
            f"{http_base.rstrip('/')}/chat/completions",
            {
                "model": model.repo,
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {"data": data_url},
                    }],
                }],
                "stream": False,
            },
            headers,
            description,
            timeout_seconds,
            proxy,
        )
        try:
            transcript = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected {description} response shape") from exc
    elif model.backend == "dashscope-audio3":
        audio_format = content_type.split(";", 1)[0].rsplit("/", 1)[-1].lower()
        response = _post_json(
            native_url,
            {
                "model": model.repo,
                "input": {"messages": [{
                    "role": "user",
                    "content": [{"audio": data_url}],
                }]},
                "parameters": {
                    "result_format": "message",
                    "format": audio_format,
                },
            },
            headers,
            description,
            timeout_seconds,
            proxy,
        )
        output = response.get("output", {})
        transcript = output.get("text", "") if isinstance(output, dict) else ""
    else:
        raise ValueError(f"Unsupported ASR backend: {model.backend}")
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError(f"{description} returned an empty transcript")
    return transcript.strip()


def prepare_asr_upload(audio_path: Path, cache_dir: Path) -> Path:
    """Create a compact speech-quality AAC copy for large direct API uploads."""
    if audio_path.suffix.casefold() != ".wav" or audio_path.stat().st_size < 2_000_000:
        return audio_path
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return audio_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(audio_path)
    target = cache_dir / f"{audio_path.stem}-{source_hash[:12]}.m4a"
    if not target.is_file() or target.stat().st_size == 0:
        completed = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio_path),
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "48k",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(target),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0 or not target.is_file():
            raise RuntimeError(
                f"ffmpeg could not prepare {audio_path.name}: "
                f"{completed.stderr.strip()[:1000]}"
            )
    return target if target.stat().st_size < audio_path.stat().st_size else audio_path


def generate_asr_outputs(
    input_dir: Path,
    output_dir: Path,
    api_key: str,
    models: Sequence[ASRModel],
    http_base: str = DEFAULT_DASHSCOPE_HTTP_BASE,
    native_url: str = DEFAULT_DASHSCOPE_NATIVE_URL,
    proxy: str | None = None,
    refresh: bool = False,
) -> dict[Path, str]:
    """Generate or reuse hash-keyed transcripts for each recording and model."""
    pairs = discover_input_pairs(input_dir)
    transcript_root = output_dir / "transcripts"
    manifest_path = output_dir / "asr_manifest.json"
    cached: dict[str, object] = {}
    if manifest_path.is_file() and not refresh:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if isinstance(loaded, dict) and isinstance(loaded.get("transcripts"), dict):
            cached = loaded["transcripts"]
    # Preserve completed entries while a refresh is in progress. This makes the
    # central checkpoint resilient to interruption and overlapping invocations.
    manifest_entries: dict[str, object] = dict(cached)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "provider": "Alibaba Cloud DashScope",
        "transcripts": manifest_entries,
    }
    upload_paths = {
        consultation_id: prepare_asr_upload(
            audio_path, output_dir / "upload_cache"
        )
        for consultation_id, (_record_path, audio_path) in pairs.items()
    }
    result: dict[Path, str] = {}
    for model in models:
        model_dir = transcript_root / model.model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        result[model_dir] = model.model_id
        for consultation_id, (_record_path, audio_path) in pairs.items():
            audio_hash = sha256_file(audio_path)
            cache_key = f"{model.model_id}/{consultation_id}"
            transcript_path = model_dir / f"{consultation_id}.txt"
            cached_entry = cached.get(cache_key)
            if (
                not refresh
                and transcript_path.is_file()
                and transcript_path.stat().st_size > 0
                and isinstance(cached_entry, dict)
                and cached_entry.get("model_repo") == model.repo
                and cached_entry.get("audio_sha256") == audio_hash
            ):
                manifest_entries[cache_key] = cached_entry
                continue
            if not api_key:
                raise ValueError(
                    "DASHSCOPE_API_KEY is required to generate missing or stale "
                    "ASR transcripts"
                )
            print(f"Transcribing with {model.model_id}: {consultation_id}")
            transcript = dashscope_transcribe(
                api_key,
                model,
                upload_paths[consultation_id],
                http_base=http_base,
                native_url=native_url,
                proxy=proxy,
            )
            _write_text_atomic(transcript_path, transcript + "\n")
            manifest_entries[cache_key] = {
                "model_id": model.model_id,
                "model_repo": model.repo,
                "backend": model.backend,
                "consultation_id": consultation_id,
                "audio_file": audio_path.name,
                "upload_file": upload_paths[consultation_id].name,
                "audio_sha256": audio_hash,
                "transcript_file": str(transcript_path.relative_to(output_dir)),
                "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            _write_json(manifest_path, manifest)
    _write_json(manifest_path, manifest)
    return result


def keyword_match(keyword: Keyword, hypothesis: str) -> str:
    """Return the accepted form found by strict normalized substring matching."""
    normalized_hypothesis = normalize_text(hypothesis)
    forms = tuple(dict.fromkeys((keyword.text, *keyword.accepted_forms)))
    for form in sorted(forms, key=lambda item: len(normalize_text(item)), reverse=True):
        normalized_form = normalize_text(form)
        if normalized_form and normalized_form in normalized_hypothesis:
            return form
    return ""


def load_keyword_csv(path: Path) -> dict[str, object]:
    """Convert the exported keyword-list CSV into keyword definitions."""
    required = {
        "consultation_id",
        "diagnosis",
        "keyword",
        "category",
        "accepted_forms",
        "doctor_evidence",
    }
    consultations: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Keyword CSV is missing columns: {', '.join(sorted(missing))}"
            )
        for line_number, row in enumerate(reader, start=2):
            consultation_id = (row.get("consultation_id") or "").strip()
            title = (row.get("diagnosis") or "").strip()
            keyword = (row.get("keyword") or "").strip()
            category = (row.get("category") or "").strip()
            evidence = (row.get("doctor_evidence") or "").strip()
            if not all((consultation_id, title, keyword, category, evidence)):
                raise ValueError(
                    f"Keyword CSV line {line_number} has an empty required value"
                )
            definition = consultations.setdefault(
                consultation_id, {"title": title, "keywords": []}
            )
            if definition["title"] != title:
                raise ValueError(
                    f"Keyword CSV has inconsistent diagnoses for {consultation_id}"
                )
            accepted_forms = [
                item.strip()
                for item in (row.get("accepted_forms") or "").split("|")
                if item.strip() and normalize_text(item) != normalize_text(keyword)
            ]
            definition["keywords"].append(
                {
                    "keyword": keyword,
                    "category": category,
                    "accepted_forms": accepted_forms,
                    "doctor_evidence": evidence,
                }
            )
    if not consultations:
        raise ValueError("Keyword CSV must contain at least one data row")
    return {"consultations": consultations}


def load_consultations(input_dir: Path, keyword_config: Path) -> list[Consultation]:
    if keyword_config.suffix.lower() == ".csv":
        raw_config = load_keyword_csv(keyword_config)
    else:
        raw_config = json.loads(keyword_config.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict) or not raw_config:
        raise ValueError("Keyword config must be a non-empty JSON object")
    definitions = raw_config.get("consultations")
    if not isinstance(definitions, dict):
        # Backward-compatible support for the earlier hand-authored config shape.
        definitions = raw_config
    pairs = discover_input_pairs(input_dir)
    unconfigured = sorted(set(pairs) - set(definitions))
    if unconfigured:
        raise ValueError(
            "Consultation pairs missing from keyword config: "
            + ", ".join(unconfigured)
        )

    consultations: list[Consultation] = []
    for consultation_id, (record_path, audio_path) in pairs.items():
        definition = definitions[consultation_id]
        if not isinstance(definition, dict):
            raise ValueError(f"Invalid definition for {consultation_id}")
        record_text = load_appointment_text(record_path)
        raw_keywords = definition.get("keywords")
        if not isinstance(raw_keywords, list) or not raw_keywords:
            raise ValueError(f"{consultation_id} must have at least one keyword")

        keywords: list[Keyword] = []
        seen: set[str] = set()
        for raw_keyword in raw_keywords:
            if not isinstance(raw_keyword, dict):
                raise ValueError(f"Invalid keyword entry in {consultation_id}")
            text = str(raw_keyword.get("keyword", "")).strip()
            category = str(raw_keyword.get("category", "")).strip()
            evidence = str(raw_keyword.get("doctor_evidence", "")).strip()
            accepted = tuple(
                str(item).strip()
                for item in raw_keyword.get("accepted_forms", [])
                if str(item).strip()
            )
            if not text or not category or not evidence:
                raise ValueError(
                    f"Every keyword needs keyword/category/doctor_evidence: "
                    f"{consultation_id}"
                )
            normalized = normalize_text(text)
            if normalized in seen:
                raise ValueError(f"Duplicate keyword {text!r} in {consultation_id}")
            seen.add(normalized)
            keyword = Keyword(text, category, accepted, evidence)
            if normalize_text(evidence) not in normalize_text(record_text):
                raise ValueError(
                    f"Doctor evidence {evidence!r} is absent from {record_path}"
                )
            keywords.append(keyword)

        consultations.append(
            Consultation(
                consultation_id=str(consultation_id),
                title=str(definition.get("title", consultation_id)).strip(),
                audio_path=audio_path,
                record_path=record_path,
                record_text=record_text,
                keywords=tuple(keywords),
            )
        )
    return consultations


def parse_model_outputs(values: Sequence[str], root: Path) -> dict[Path, str]:
    models: dict[Path, str] = {}
    labels: set[str] = set()
    for value in values:
        if "=" in value:
            label, raw_path = value.split("=", 1)
            label = label.strip()
        else:
            raw_path = value
            label = Path(raw_path).name
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not label:
            raise ValueError(f"Missing label in --model-output {value!r}")
        if not path.is_dir():
            raise ValueError(f"Model output directory does not exist: {path}")
        if label in labels:
            raise ValueError(f"Duplicate model label: {label}")
        labels.add(label)
        models[path] = label
    if not models:
        raise ValueError("At least one model output is required")
    return models


def relative_label(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def _winner(rows: Sequence[dict[str, object]]) -> str:
    best = max(float(row["keyword_recall"]) for row in rows)
    names = [
        str(row["model"])
        for row in rows
        if math.isclose(float(row["keyword_recall"]), best, abs_tol=5e-10)
    ]
    return names[0] if len(names) == 1 else "Tie"


def build_summary(
    file_rows: Sequence[dict[str, object]],
    keyword_rows: Sequence[dict[str, object]],
    model_names: Sequence[str],
    consultations: Sequence[Consultation],
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []

    def append_scope(
        scope_type: str,
        scope_value: str,
        rows_by_model: dict[str, list[dict[str, object]]],
    ) -> None:
        scope_rows: list[dict[str, object]] = []
        for model in model_names:
            rows = rows_by_model[model]
            hits = sum(int(row["hit"]) for row in rows)
            expected = len(rows)
            scope_row = {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "model": model,
                "consultation_count": len(
                    {str(row["consultation_id"]) for row in rows}
                ),
                "keyword_hits": hits,
                "keyword_expected": expected,
                "keyword_recall": f"{hits / expected:.6f}",
                "keyword_winner": "",
            }
            scope_rows.append(scope_row)
        winner = _winner(scope_rows)
        for row in scope_rows:
            row["keyword_winner"] = winner
        summary.extend(scope_rows)

    append_scope(
        "overall",
        "all",
        {
            model: [row for row in keyword_rows if row["model"] == model]
            for model in model_names
        },
    )
    for consultation in consultations:
        append_scope(
            "consultation",
            consultation.consultation_id,
            {
                model: [
                    row
                    for row in keyword_rows
                    if row["model"] == model
                    and row["consultation_id"] == consultation.consultation_id
                ]
                for model in model_names
            },
        )
    categories = sorted({str(row["category"]) for row in keyword_rows})
    for category in categories:
        append_scope(
            "category",
            category,
            {
                model: [
                    row
                    for row in keyword_rows
                    if row["model"] == model and row["category"] == category
                ]
                for model in model_names
            },
        )

    if len(file_rows) != len(model_names) * len(consultations):
        raise ValueError("File-metric row count does not reconcile")
    return summary


def summary_row(
    summary: Sequence[dict[str, object]], scope_type: str, scope_value: str, model: str
) -> dict[str, object]:
    return next(
        row
        for row in summary
        if row["scope_type"] == scope_type
        and row["scope_value"] == scope_value
        and row["model"] == model
    )


def render_report(
    summary: Sequence[dict[str, object]],
    keyword_rows: Sequence[dict[str, object]],
    consultations: Sequence[Consultation],
    models: dict[Path, str],
    root: Path,
    keyword_config: Path,
    keyword_model: str,
    asr_generated: bool,
) -> str:
    model_names = list(models.values())
    overall = [
        summary_row(summary, "overall", "all", model) for model in model_names
    ]
    winner = str(overall[0]["keyword_winner"])
    title = (
        " 與 ".join(model_names)
        + " Real Consultation ASR 醫療關鍵字準確率評測"
    )
    lines = [
        f"# {title}",
        "",
        "## 結論摘要",
        "",
    ]
    if len(model_names) == 1:
        row = overall[0]
        lines.append(
            f"- **{row['model']} 整體醫療關鍵字 Recall："
            f"{float(row['keyword_recall']):.2%}**（{row['keyword_hits']}/"
            f"{row['keyword_expected']}）。"
        )
    else:
        values = "，".join(
            f"{row['model']} 為 {float(row['keyword_recall']):.2%}"
            for row in overall
        )
        winner_text = "平手" if winner == "Tie" else winner
        lines.append(f"- **整體醫療關鍵字辨識較佳：{winner_text}**。{values}。")
        for consultation in consultations:
            consultation_rows = [
                summary_row(
                    summary,
                    "consultation",
                    consultation.consultation_id,
                    model,
                )
                for model in model_names
            ]
            consultation_winner = str(
                consultation_rows[0]["keyword_winner"]
            )
            consultation_winner_text = (
                "平手" if consultation_winner == "Tie"
                else consultation_winner
            )
            consultation_values = "，".join(
                f"{row['model']} 為 {float(row['keyword_recall']):.2%}"
                for row in consultation_rows
            )
            lines.append(
                f"- **`{consultation.consultation_id}` 關鍵字辨識較佳："
                f"{consultation_winner_text}**。{consultation_values}。"
            )
    lines += [
        "",
        "## 整體結果",
        "",
        "| 模型 | 關鍵字命中 | 關鍵字總數 | 關鍵字 Recall |",
        "|---|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['model']} | {row['keyword_hits']} | "
            f"{row['keyword_expected']} | "
            f"{float(row['keyword_recall']):.2%} |"
        )

    lines += ["", "## Consultation 比較", ""]
    if len(model_names) == 2:
        name_a, name_b = model_names
        lines += [
            f"| Consultation | 診斷 | {name_a} 關鍵字 | "
            f"{name_b} 關鍵字 | 關鍵字較佳 |",
            "|---|---|---:|---:|---|",
        ]
        for consultation in consultations:
            row_a = summary_row(
                summary, "consultation", consultation.consultation_id, name_a
            )
            row_b = summary_row(
                summary, "consultation", consultation.consultation_id, name_b
            )
            comparison_winner = (
                "平手"
                if row_a["keyword_winner"] == "Tie"
                else row_a["keyword_winner"]
            )
            lines.append(
                f"| {consultation.consultation_id} | {consultation.title} | "
                f"{float(row_a['keyword_recall']):.2%} "
                f"({row_a['keyword_hits']}/{row_a['keyword_expected']}) | "
                f"{float(row_b['keyword_recall']):.2%} "
                f"({row_b['keyword_hits']}/{row_b['keyword_expected']}) | "
                f"{comparison_winner} |"
            )
    else:
        lines += [
            "| Consultation | 診斷 | 模型 | 命中 | 預期 | 關鍵字 Recall |",
            "|---|---|---|---:|---:|---:|",
        ]
        for consultation in consultations:
            for model in model_names:
                row = summary_row(
                    summary,
                    "consultation",
                    consultation.consultation_id,
                    model,
                )
                lines.append(
                    f"| {consultation.consultation_id} | "
                    f"{consultation.title} | {model} | "
                    f"{row['keyword_hits']} | {row['keyword_expected']} | "
                    f"{float(row['keyword_recall']):.2%} |"
                )

    categories = sorted(
        {
            str(row["scope_value"])
            for row in summary
            if row["scope_type"] == "category"
        }
    )
    lines += [
        "",
        "## 關鍵字類別",
        "",
        "| 類別 | 模型 | 命中 | 預期 | Recall |",
        "|---|---|---:|---:|---:|",
    ]
    for category in categories:
        for model in model_names:
            row = summary_row(summary, "category", category, model)
            lines.append(
                f"| {category} | {model} | {row['keyword_hits']} | "
                f"{row['keyword_expected']} | {float(row['keyword_recall']):.2%} |"
            )

    lines += ["", "## 未命中關鍵字", ""]
    for model in model_names:
        misses = [
            row
            for row in keyword_rows
            if row["model"] == model and int(row["hit"]) == 0
        ]
        lines += [f"### {model}", ""]
        if not misses:
            lines.append("- 無")
        else:
            grouped: dict[str, list[str]] = {}
            for row in misses:
                grouped.setdefault(str(row["consultation_id"]), []).append(
                    f"{row['keyword']}（{row['category']}）"
                )
            for consultation_id, values in grouped.items():
                lines.append(f"- `{consultation_id}`：{'、'.join(values)}")
        lines.append("")

    config_label = relative_label(keyword_config, root)
    asr_method = (
        "- ASR transcript 由本工具直接把 `input/` 錄音送到所列 DashScope "
        "模型產生；不需要預先提供 ASR output。"
        if asr_generated
        else "- 本次驗證使用 `--model-output` 提供的現有 transcript；預設自動模式"
        "會把 `input/` 錄音送到所選 DashScope 模型產生 transcript。"
    )
    if keyword_config.suffix.lower() == ".csv":
        keyword_method = (
            f"- 關鍵字直接由人工整理的 `{config_label}` 載入；本次不會呼叫 "
            "Gemini，也不會重新產生或改寫關鍵字。"
        )
        coverage_method = (
            "- 本報告使用 legacy manual 關鍵字評估 ASR 醫療關鍵字 coverage，"
            "不計算 CER/WER。"
        )
    else:
        keyword_method = (
            f"- 關鍵字由 Google `{keyword_model}` 根據 appointment 自動產生，"
            "並建立預期的 Cantonese/English accepted forms；appointment 會送到 "
            "Gemini，原始錄音不會送到 Gemini。"
        )
        coverage_method = (
            "- 本報告評估 appointment-grounded 醫療關鍵字 coverage，不使用人工 "
            "reference transcript，也不計算 CER/WER；若 appointment 內容未在錄音"
            "中說出，該項仍可能被計作未命中，因此結果同時反映文件與錄音的一致性。"
        )
    lines += [
        "## 方法與解讀",
        "",
        keyword_method,
        asr_method,
        coverage_method,
        "- 命中採 Unicode NFKC、英文字母大小寫、空白及標點正規化後的完整短語匹配；不自動合併繁簡字、粵語異體字、同義詞或數字寫法。",
        "- 同一臨床概念如有 accepted forms，只計一次命中。此指標是 recall，不處罰 ASR 額外產生的假陽性內容。",
        "- 詳細關鍵字來源、accepted forms、逐模型命中和逐 consultation 結果見同目錄 CSV。",
        "",
        "## 資料範圍",
        "",
        f"- 模型：{'、'.join(model_names)}",
        f"- Consultation：{len(consultations)} 段 × {len(model_names)} 模型，共 {len(consultations) * len(model_names)} 份 ASR 輸出",
        "- Recording：`input/<consultation-id>.(wav|mp3|m4a|flac|aac|ogg)`",
        "- Appointment note：`input/<consultation-id>.appointment.txt` 或 "
        "`input/<consultation-id>.record_appointment.json`",
        f"- 關鍵字設定：`{config_label}`",
    ]
    return "\n".join(lines) + "\n"


def validate_outputs(
    output_dir: Path,
    file_rows: Sequence[dict[str, object]],
    keyword_rows: Sequence[dict[str, object]],
    summary: Sequence[dict[str, object]],
    consultations: Sequence[Consultation],
    models: dict[Path, str],
) -> None:
    expected_files = len(consultations) * len(models)
    expected_keyword_rows = (
        sum(len(consultation.keywords) for consultation in consultations)
        * len(models)
    )
    if len(file_rows) != expected_files:
        raise ValueError(
            f"Expected {expected_files} file rows, found {len(file_rows)}"
        )
    if len(keyword_rows) != expected_keyword_rows:
        raise ValueError(
            f"Expected {expected_keyword_rows} keyword rows, "
            f"found {len(keyword_rows)}"
        )
    file_keys = {
        (str(row["model"]), str(row["consultation_id"])) for row in file_rows
    }
    if len(file_keys) != expected_files:
        raise ValueError("Duplicate model/consultation file metric")
    for row in keyword_rows:
        if int(row["hit"]) not in {0, 1}:
            raise ValueError("Keyword hit must be 0 or 1")
        if bool(int(row["hit"])) != bool(row["matched_form"]):
            raise ValueError("Keyword hit and matched_form do not agree")
    for name in (
        "file_metrics.csv",
        "summary.csv",
        "keyword_list.csv",
        "keyword_results.csv",
    ):
        path = output_dir / name
        if not path.read_bytes().startswith(b"\xef\xbb\xbf"):
            raise ValueError(f"CSV is missing UTF-8 BOM: {name}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            list(csv.DictReader(handle))
    for model in models.values():
        overall = summary_row(summary, "overall", "all", model)
        selected = [row for row in keyword_rows if row["model"] == model]
        if sum(int(row["hit"]) for row in selected) != int(
            overall["keyword_hits"]
        ):
            raise ValueError(f"Overall summary does not reconcile for {model}")


def evaluate(
    root: Path,
    input_dir: Path,
    keyword_config: Path,
    output_dir: Path,
    models: dict[Path, str],
    keyword_model: str,
    asr_generated: bool = True,
) -> None:
    consultations = load_consultations(input_dir, keyword_config)
    keyword_rows: list[dict[str, object]] = []
    file_rows: list[dict[str, object]] = []

    for model_dir, model_name in models.items():
        for consultation in consultations:
            source = model_dir / f"{consultation.consultation_id}.txt"
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty ASR output: {source}")
            hypothesis = source.read_text(encoding="utf-8-sig").strip()
            hits = 0
            for keyword in consultation.keywords:
                matched_form = keyword_match(keyword, hypothesis)
                hit = bool(matched_form)
                hits += int(hit)
                keyword_rows.append(
                    {
                        "model": model_name,
                        "consultation_id": consultation.consultation_id,
                        "diagnosis": consultation.title,
                        "keyword": keyword.text,
                        "category": keyword.category,
                        "accepted_forms": " | ".join(
                            dict.fromkeys((keyword.text, *keyword.accepted_forms))
                        ),
                        "doctor_evidence": keyword.doctor_evidence,
                        "normalized_keyword": normalize_text(keyword.text),
                        "hit": int(hit),
                        "matched_form": matched_form,
                        "source_file": relative_label(source, root),
                    }
                )
            file_rows.append(
                {
                    "model": model_name,
                    "consultation_id": consultation.consultation_id,
                    "diagnosis": consultation.title,
                    "source_file": relative_label(source, root),
                    "keyword_hits": hits,
                    "keyword_expected": len(consultation.keywords),
                    "keyword_recall": f"{hits / len(consultation.keywords):.6f}",
                }
            )

    summary = build_summary(
        file_rows, keyword_rows, list(models.values()), consultations
    )
    keyword_list_rows = [
        {
            "consultation_id": consultation.consultation_id,
            "diagnosis": consultation.title,
            "keyword": keyword.text,
            "category": keyword.category,
            "accepted_forms": " | ".join(
                dict.fromkeys((keyword.text, *keyword.accepted_forms))
            ),
            "doctor_evidence": keyword.doctor_evidence,
            "audio_file": relative_label(consultation.audio_path, root),
            "doctor_record_file": relative_label(consultation.record_path, root),
        }
        for consultation in consultations
        for keyword in consultation.keywords
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "file_metrics.csv", list(file_rows[0]), file_rows)
    write_csv(output_dir / "summary.csv", list(summary[0]), summary)
    write_csv(
        output_dir / "keyword_list.csv",
        list(keyword_list_rows[0]),
        keyword_list_rows,
    )
    write_csv(
        output_dir / "keyword_results.csv",
        list(keyword_rows[0]),
        keyword_rows,
    )
    _write_text_atomic(
        output_dir / "ASR-evaluation-report.md",
        render_report(
            summary,
            keyword_rows,
            consultations,
            models,
            root,
            keyword_config,
            keyword_model,
            asr_generated,
        ),
    )
    validate_outputs(
        output_dir,
        file_rows,
        keyword_rows,
        summary,
        consultations,
        models,
    )
    print(
        f"Evaluated {len(file_rows)} ASR files and "
        f"{len(keyword_rows)} model-keyword pairs; results written to {output_dir}"
    )


def run_pipeline(
    config: PipelineConfig,
    *,
    refresh_keywords: bool = False,
    refresh_asr: bool = False,
    keywords_only: bool = False,
    model_outputs: dict[Path, str] | None = None,
) -> Path:
    """Run the complete production workflow behind one small interface."""
    started_at = dt.datetime.now(dt.timezone.utc)
    keyword_config = config.keyword_config
    if keyword_config.suffix.lower() == ".csv":
        if refresh_keywords:
            raise ValueError("Cannot refresh generated keywords when using a CSV")
        if not keyword_config.is_file():
            raise FileNotFoundError(
                f"Keyword CSV does not exist: {keyword_config}"
            )
        load_consultations(config.input_dir, keyword_config)
    else:
        generate_keyword_config(
            config.input_dir,
            keyword_config,
            config.gemini_api_key,
            config.keyword_model,
            refresh=refresh_keywords,
        )

    if keywords_only:
        print(f"Keyword config written to {keyword_config}")
        return keyword_config

    if model_outputs is None:
        models = generate_asr_outputs(
            config.input_dir,
            config.output_dir,
            config.dashscope_api_key,
            config.asr_models,
            http_base=config.dashscope_http_base,
            native_url=config.dashscope_native_url,
            proxy=config.dashscope_proxy,
            refresh=refresh_asr,
        )
        asr_generated = True
    else:
        models = model_outputs
        asr_generated = False

    evaluate(
        config.root,
        config.input_dir,
        keyword_config,
        config.output_dir,
        models,
        config.keyword_model,
        asr_generated,
    )
    pairs = discover_input_pairs(config.input_dir)
    completed_at = dt.datetime.now(dt.timezone.utc)
    _write_json(config.output_dir / "run_manifest.json", {
        "schema_version": 1,
        "status": "completed",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "keyword_model": config.keyword_model,
        "asr_models": list(models.values()),
        "consultations": {
            consultation_id: {
                "appointment_file": note_path.name,
                "appointment_sha256": sha256_file(note_path),
                "audio_file": audio_path.name,
                "audio_sha256": sha256_file(audio_path),
            }
            for consultation_id, (note_path, audio_path) in pairs.items()
        },
        "report": "ASR-evaluation-report.md",
    })
    return config.output_dir / "ASR-evaluation-report.md"


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--keyword-config", type=Path, default=None)
    parser.add_argument(
        "--asr-models",
        default=None,
        help=(
            "Comma-separated ASR model IDs (default: ASR_MODELS from .env)"
        ),
    )
    parser.add_argument(
        "--model-output",
        action="append",
        metavar="[LABEL=]PATH",
        help=(
            "Use an existing ASR output directory instead of calling DashScope; "
            "repeat for multiple models"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--refresh-keywords", action="store_true",
        help="Regenerate Gemini keywords even when the input hashes match",
    )
    parser.add_argument(
        "--refresh-asr", action="store_true",
        help="Regenerate DashScope transcripts even when the audio hashes match",
    )
    parser.add_argument(
        "--keywords-only", action="store_true",
        help="Generate/validate the Gemini keyword cache, then stop",
    )
    args = parser.parse_args()

    try:
        load_env(script_dir / ".env")
        config = build_pipeline_config(
            script_dir,
            root=args.root,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            keyword_config=args.keyword_config,
            asr_models=args.asr_models,
        )
        model_outputs = (
            parse_model_outputs(args.model_output, config.root)
            if args.model_output
            else None
        )
        report_path = run_pipeline(
            config,
            refresh_keywords=args.refresh_keywords,
            refresh_asr=args.refresh_asr,
            keywords_only=args.keywords_only,
            model_outputs=model_outputs,
        )
        if not args.keywords_only:
            print(f"Report: {report_path}")
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

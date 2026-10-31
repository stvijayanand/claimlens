"""Streamlit app: upload -> progress -> evidence-backed report.

Single-column, mobile-friendly layout (spec section 9). The app talks to
the ClaimLens API over HTTP only -- it makes no model calls itself. Run
with the API up, e.g.::

    uvicorn claimlens.api:app --port 8000
    CLAIMLENS_API_BASE=http://127.0.0.1:8000 streamlit run -m claimlens.ui.app

Demo replay needs no API key: "Load a previous report" fetches the stored
report for a car id, which the server assembles from the database alone.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
import streamlit as st

API_BASE = os.environ.get("CLAIMLENS_API_BASE", "http://127.0.0.1:8000")
POLL_SECONDS = 2.0
REQUEST_TIMEOUT = 30.0

FLAG_TITLES = {
    "phantom_line": "Billed but not visible",
    "missing_line": "Visible damage not on the estimate",
    "repair_vs_replace": "Repair vs replace disagreement",
    "duplicate": "Possible duplicate lines",
    "labor_out_of_range": "Labor hours outside the typical range",
}


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE, timeout=REQUEST_TIMEOUT)


def _api_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail")
        except ValueError:
            detail = exc.response.text
        return f"API error {exc.response.status_code}: {detail}"
    return f"Request failed: {exc}"


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _active_flags() -> dict[str, bool]:
    """Feature flags from the latest run status, else the local env."""
    saved = st.session_state.get("feature_flags")
    if isinstance(saved, dict):
        return {
            "labor_check": bool(saved.get("labor_check", True)),
            "repair_replace": bool(saved.get("repair_replace", True)),
        }
    return {
        "labor_check": _env_flag("CLAIMLENS_FEATURE_LABOR_CHECK", True),
        "repair_replace": _env_flag("CLAIMLENS_FEATURE_REPAIR_REPLACE", True),
    }


def _visible_flags(flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hide disabled flag types per the feature flags (spec section 14)."""
    active = _active_flags()
    hidden_types = set()
    if not active["labor_check"]:
        hidden_types.add("labor_out_of_range")
    if not active["repair_replace"]:
        hidden_types.add("repair_vs_replace")
    return [f for f in flags if f.get("flag_type") not in hidden_types]


def _reset_to_upload() -> None:
    for key in ("page", "car_id", "run_id", "report", "feature_flags", "error"):
        st.session_state.pop(key, None)
    st.session_state["page"] = "upload"


def _upload_view() -> None:
    st.title("ClaimLens")
    st.write(
        "Check a body-shop repair estimate against photos of the damage. "
        "Upload 4-10 photos plus the estimate to start."
    )

    photos = st.file_uploader(
        "Damage photos (4-10, JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    estimate_file = st.file_uploader(
        "Repair estimate (PDF, JPG or PNG)",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if st.button("Start analysis", type="primary", use_container_width=True):
        photo_list = list(photos or [])
        if not 4 <= len(photo_list) <= 10:
            st.error(
                f"Upload 4-10 photos to start (you selected"
                f" {len(photo_list)})."
            )
        elif estimate_file is None:
            st.error("Upload the repair estimate to start.")
        else:
            _start_run(photo_list, estimate_file)

    with st.expander("Load a previous report (demo replay, no key needed)"):
        st.write(
            "Enter a car id from an earlier run to render its stored report"
            " without running the pipeline again."
        )
        car_id = st.text_input("Car id", key="replay_car_id")
        if st.button("Load report", use_container_width=True):
            if not car_id.strip():
                st.error("Enter a car id first.")
            else:
                _load_report(car_id.strip())


def _start_run(photo_list: list[Any], estimate_file: Any) -> None:
    try:
        with _client() as client:
            car_id = client.post("/cars").raise_for_status().json()["car_id"]
            files = [
                ("files", (p.name, p.getvalue(), p.type or "image/jpeg"))
                for p in photo_list
            ]
            client.post(f"/cars/{car_id}/photos", files=files).raise_for_status()
            client.post(
                f"/cars/{car_id}/estimate",
                files={
                    "file": (
                        estimate_file.name,
                        estimate_file.getvalue(),
                        estimate_file.type or "application/octet-stream",
                    )
                },
            ).raise_for_status()
            run_id = (
                client.post(f"/cars/{car_id}/run").raise_for_status().json()["run_id"]
            )
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        st.error(_api_error(exc))
        return
    st.session_state["car_id"] = car_id
    st.session_state["run_id"] = run_id
    st.session_state["page"] = "progress"
    st.rerun()


def _load_report(car_id: str) -> None:
    try:
        with _client() as client:
            response = client.get(f"/cars/{car_id}/report")
            response.raise_for_status()
            st.session_state["report"] = response.json()
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        st.error(_api_error(exc))
        return
    st.session_state["car_id"] = car_id
    st.session_state["run_id"] = st.session_state["report"].get("run_id")
    st.session_state["page"] = "report"
    st.rerun()


def _progress_view() -> None:
    car_id = st.session_state.get("car_id", "")
    run_id = st.session_state.get("run_id", "")
    st.title("Analyzing your estimate")
    st.write(f"Car `{car_id}` - run `{run_id}`")
    bar = st.progress(0.0)
    status_line = st.empty()

    try:
        with _client() as client:
            response = client.get(f"/cars/{car_id}/runs/{run_id}")
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        st.error(_api_error(exc))
        if st.button("Back to upload"):
            _reset_to_upload()
            st.rerun()
        return

    st.session_state["feature_flags"] = payload.get("feature_flags", {})
    status = payload.get("status", "queued")
    bar.progress(min(max(float(payload.get("progress", 0.0)), 0.0), 1.0))
    if status in {"queued", "running"}:
        status_line.write(f"Status: **{status}** - checking again shortly.")
        time.sleep(POLL_SECONDS)
        st.rerun()
    elif status == "done":
        st.session_state["report"] = payload.get("report", {})
        st.session_state["page"] = "report"
        st.rerun()
    else:
        st.error(f"Run failed: {payload.get('cause') or payload.get('error')}")
        if st.button("Back to upload"):
            _reset_to_upload()
            st.rerun()


def _fetch_crop(client: httpx.Client, crop_id: str) -> bytes | None:
    try:
        response = client.get(f"/crops/{crop_id}")
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    return response.content


def _format_lines(nos: list[int]) -> str:
    ordered = sorted(nos)
    if len(ordered) == 1:
        return f"Line {ordered[0]}"
    return "Lines " + ", ".join(str(no) for no in ordered)


def _report_view() -> None:
    built: dict[str, Any] = st.session_state.get("report", {})
    flags = _visible_flags(list(built.get("flags", [])))
    notes = list(built.get("notes", []))

    st.title("Estimate check")
    st.write(f"Car `{built.get('car_id', '')}`")
    counts = built.get("counts", {})
    total = counts.get("total", len(flags))
    if total == 0:
        st.success("No issues found: every line matches the visible damage.")
    else:
        st.write(f"Found **{total}** item(s) worth asking the shop about.")

    for flag in flags:
        title = FLAG_TITLES.get(flag.get("flag_type", ""), flag.get("flag_type"))
        st.subheader(f"{title} - {flag.get('canonical_part', '')}")
        st.write(flag.get("reason", ""))
        line_nos = list(flag.get("evidence_line_nos", []))
        if line_nos:
            st.write(f"Cited evidence: **{_format_lines(line_nos)}**")
        st.write("Confidence")
        st.progress(min(max(float(flag.get("confidence", 0.0)), 0.0), 1.0))
        crop_ids = list(flag.get("evidence_crop_ids", []))
        if crop_ids:
            st.write("Damage evidence:")
            with _client() as client:
                for crop_id in crop_ids:
                    blob = _fetch_crop(client, crop_id)
                    if blob:
                        st.image(blob, caption=crop_id)
        questions = list(flag.get("shop_questions", []))
        if questions:
            st.write("Ask the shop:")
            for question in questions:
                st.write(f"- {question}")
        st.divider()

    if notes:
        st.header("Notes")
        for note in notes:
            where = (
                f"Line {note['line_no']}: " if note.get("line_no") else ""
            )
            st.write(f"- {where}{note.get('reason', '')}")

    duration_ms = built.get("duration_ms")
    cost = built.get("api_cost_usd")
    footer_bits = []
    if duration_ms is not None:
        footer_bits.append(f"Time: {duration_ms / 1000:.1f}s")
    if cost is not None:
        footer_bits.append(f"Estimated API cost: ${cost:.4f}")
    if built.get("unparsed_lines"):
        footer_bits.append(
            f"Unparsed estimate lines: {built['unparsed_lines']}"
        )
    if footer_bits:
        st.caption(" - ".join(footer_bits))

    if st.button("Check another car", use_container_width=True):
        _reset_to_upload()
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="ClaimLens", page_icon=":mag:", layout="centered"
    )
    page = st.session_state.get("page", "upload")
    if page == "progress":
        _progress_view()
    elif page == "report":
        _report_view()
    else:
        _upload_view()


main()

from __future__ import annotations

import base64
from typing import Any


def capture_region_as_base64(screenshot_region: dict[str, Any]) -> str:
    """Capture a fixed screen region and return PNG base64.

    The first version intentionally captures only the explicit rectangle passed
    by the frontend. It does not scan the whole screen or locate app windows.
    """
    region = _normalize_region(screenshot_region)
    if region.get("mock_image_base64"):
        return str(region["mock_image_base64"])

    try:
        import mss
        from mss import tools as mss_tools
    except Exception as exc:  # pragma: no cover - depends on local desktop deps
        raise RuntimeError("mss is required for screen capture. Install it with: pip install mss") from exc

    monitor = {
        "left": region["left"],
        "top": region["top"],
        "width": region["width"],
        "height": region["height"],
    }
    try:
        with mss.mss() as screen_capture:
            raw = screen_capture.grab(monitor)
            png_bytes = mss_tools.to_png(raw.rgb, raw.size)
    except Exception as exc:  # pragma: no cover - depends on local desktop state
        raise RuntimeError(f"screenshot_failed:{exc}") from exc

    return base64.b64encode(png_bytes).decode("ascii")


def _normalize_region(screenshot_region: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(screenshot_region, dict):
        raise ValueError("screenshot_region must be a dict")
    normalized = dict(screenshot_region)
    for key in ("left", "top", "width", "height"):
        try:
            normalized[key] = int(normalized[key])
        except Exception as exc:
            raise ValueError(f"screenshot_region.{key} must be an integer") from exc
    if normalized["width"] <= 0 or normalized["height"] <= 0:
        raise ValueError("screenshot_region width/height must be positive")
    return normalized

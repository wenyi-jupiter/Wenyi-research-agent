"""HTTP client with TLS fingerprint impersonation, UA rotation, and retry.

Replaces plain httpx with curl_cffi for browser-grade TLS fingerprinting,
which bypasses most anti-bot detection systems (Cloudflare, Akamai, etc.).
Falls back to httpx transparently if curl_cffi is not installed.

Optionally uses Playwright to render JS-heavy pages when static fetch
returns an empty template.
"""

import asyncio
import json as _json
import logging
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
try:
    from curl_cffi.requests import AsyncSession as _CffiSession  # noqa: F401

    _HAS_CURL_CFFI = True
except ImportError:
    _CffiSession = None  # type: ignore[assignment,misc]
    _HAS_CURL_CFFI = False
    logger.info("[http_client] curl_cffi not available; falling back to httpx")

# ---------------------------------------------------------------------------
# Browser-realistic User-Agent pool
# ---------------------------------------------------------------------------
USER_AGENTS = [
    # Chrome 120 – Windows
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    # Chrome 119 – Windows
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    ),
    # Chrome 120 – macOS
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    # Chrome 120 – Linux
    (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    # Firefox 121 – Windows
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
        "Gecko/20100101 Firefox/121.0"
    ),
    # Safari 17 – macOS
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.2 Safari/605.1.15"
    ),
]

# TLS fingerprint profiles supported by curl_cffi
IMPERSONATE_PROFILES = ["chrome120", "chrome119", "chrome110"]


# ---------------------------------------------------------------------------
# Unified response object
# ---------------------------------------------------------------------------
@dataclass
class HttpResponse:
    """Backend-agnostic HTTP response."""

    status_code: int
    headers: dict[str, str]  # keys are always lowercase
    text: str
    content: bytes
    url: str  # final URL after any redirects

    def json(self) -> dict:
        """Parse response body as JSON."""
        return _json.loads(self.text)


# ---------------------------------------------------------------------------
# Smart HTTP Client
# ---------------------------------------------------------------------------
class SmartHttpClient:
    """Async HTTP client with TLS impersonation, UA rotation, and retry.

    Usage::

        async with SmartHttpClient() as client:
            resp = await client.get("https://example.com")
            print(resp.status_code, resp.text[:200])
    """

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        max_retries: int = 3,
        default_headers: dict[str, str] | None = None,
    ):
        self._timeout = timeout
        self._max_retries = max_retries
        self._default_headers = default_headers or {}
        self._session: _CffiSession | None = None  # type: ignore[annotation-unchecked]
        self._httpx_client = None

    # -- context manager ---------------------------------------------------

    async def __aenter__(self):
        if _HAS_CURL_CFFI:
            from curl_cffi.requests import AsyncSession

            self._session = AsyncSession()
        else:
            import httpx

            self._httpx_client = httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
            )
        return self

    async def __aexit__(self, *exc):
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
        if self._httpx_client is not None:
            await self._httpx_client.aclose()
            self._httpx_client = None

    # -- internal helpers --------------------------------------------------

    def _build_headers(
        self, override: dict[str, str] | None = None
    ) -> dict[str, str]:
        """Merge default + randomised + caller-supplied headers."""
        h: dict[str, str] = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.google.com/",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        h.update(self._default_headers)
        if override:
            h.update(override)
        return h

    async def _do_request(
        self,
        url: str,
        headers: dict[str, str],
        timeout: float,
        impersonate: str,
    ) -> HttpResponse:
        """Execute a single HTTP GET via the active backend."""
        if self._session is not None:
            r = await self._session.get(
                url,
                headers=headers,
                impersonate=impersonate,
                timeout=timeout,
                allow_redirects=True,
            )
            return HttpResponse(
                status_code=r.status_code,
                headers={k.lower(): v for k, v in r.headers.items()},
                text=r.text,
                content=r.content,
                url=str(r.url),
            )
        elif self._httpx_client is not None:
            r = await self._httpx_client.get(
                url,
                headers=headers,
                timeout=timeout,
            )
            return HttpResponse(
                status_code=r.status_code,
                headers={k.lower(): v for k, v in r.headers.items()},
                text=r.text,
                content=r.content,
                url=str(r.url),
            )
        else:
            raise RuntimeError(
                "SmartHttpClient not initialised – use 'async with'"
            )

    # -- public API --------------------------------------------------------

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        impersonate: str | None = None,
    ) -> HttpResponse:
        """GET *url* with TLS impersonation, UA rotation, and auto-retry.

        Automatically retries on HTTP 403 / 429 with exponential back-off
        and rotated User-Agent / TLS profile.  The **final** response is
        always returned (even if it is still 403) so the caller can apply
        domain-specific fallback logic.

        If the caller provides an explicit ``User-Agent`` in *headers*, it
        will be preserved across retries (no random rotation).
        """
        eff_timeout = timeout if timeout is not None else self._timeout
        eff_retries = (
            max_retries if max_retries is not None else self._max_retries
        )

        merged = self._build_headers(headers)
        profile = impersonate or random.choice(IMPERSONATE_PROFILES)

        # If the caller pinned a User-Agent, don't rotate it on retry.
        caller_set_ua = bool(
            headers and any(k.lower() == "user-agent" for k in headers)
        )

        last_resp: HttpResponse | None = None
        last_err: BaseException | None = None

        for attempt in range(eff_retries):
            try:
                resp = await self._do_request(
                    url, merged, eff_timeout, profile
                )
                last_resp = resp

                # 429 – honour Retry-After header
                if resp.status_code == 429:
                    wait = int(resp.headers.get("retry-after", "10"))
                    logger.warning(
                        "[http_client] 429 at %s, waiting %ds (attempt %d/%d)",
                        url,
                        wait,
                        attempt + 1,
                        eff_retries,
                    )
                    await asyncio.sleep(wait)
                    if not caller_set_ua:
                        merged["User-Agent"] = random.choice(USER_AGENTS)
                    profile = random.choice(IMPERSONATE_PROFILES)
                    continue

                # 403 – cool-down + rotate fingerprint
                if resp.status_code == 403 and attempt < eff_retries - 1:
                    wait = 3.0 * (attempt + 1) + random.uniform(0, 2)
                    logger.warning(
                        "[http_client] 403 at %s, cooldown %.1fs "
                        "(attempt %d/%d)",
                        url,
                        wait,
                        attempt + 1,
                        eff_retries,
                    )
                    await asyncio.sleep(wait)
                    if not caller_set_ua:
                        merged["User-Agent"] = random.choice(USER_AGENTS)
                    profile = random.choice(IMPERSONATE_PROFILES)
                    continue

                return resp

            except Exception as exc:
                last_err = exc
                logger.warning(
                    "[http_client] Error fetching %s: %s (attempt %d/%d)",
                    url,
                    exc,
                    attempt + 1,
                    eff_retries,
                )
                if attempt < eff_retries - 1:
                    await asyncio.sleep(
                        random.uniform(1.5, 3.5) * (attempt + 1)
                    )
                    if not caller_set_ua:
                        merged["User-Agent"] = random.choice(USER_AGENTS)
                    profile = random.choice(IMPERSONATE_PROFILES)

        # Exhausted retries – return whatever we have
        if last_resp is not None:
            return last_resp
        raise TimeoutError(
            f"Failed to fetch {url} after {eff_retries} retries"
        ) from last_err


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Playwright JS-rendering fallback
# ---------------------------------------------------------------------------

# Common cookie consent / gate button selectors (ordered by specificity).
# Each entry is a CSS selector that targets "Accept" / "Accept All" / "I agree"
# buttons found on popular cookie consent frameworks (OneTrust, CookieBot,
# Quantcast, generic, pharma HCP gates, etc.).
_CONSENT_BUTTON_SELECTORS = [
    # --- Cookie consent frameworks ---
    # OneTrust (used by many pharma/biotech sites including BeOne)
    "button#onetrust-accept-btn-handler",
    # CookieBot
    "button#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
    "a#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
    # Quantcast / TrustArc
    "button.css-47sehv",  # Quantcast "Agree" button
    "button#truste-consent-button",
    # Didomi
    "button#didomi-notice-agree-button",
    # Klaro
    "button.cm-btn-accept",
    # Iubenda
    "a.iubenda-cs-accept-btn",
    # Complianz
    "button.cmplz-accept",
    # --- Generic text-based selectors (catch-all) ---
    'button:has-text("Accept All Cookies")',
    'button:has-text("Accept All")',
    'button:has-text("Accept Cookies")',
    'button:has-text("I Accept")',
    'button:has-text("Accept")',
    'button:has-text("Agree")',
    'button:has-text("OK")',
    'a:has-text("Accept All Cookies")',
    'a:has-text("Accept All")',
    # --- Healthcare Professional (HCP) gate ---
    'button:has-text("I confirm")',
    'a:has-text("I confirm")',
    'input[type="checkbox"] + label:has-text("I confirm")',
    'button:has-text("I am a healthcare professional")',
    'button:has-text("Enter Site")',
    'a:has-text("Enter Site")',
]


async def _dismiss_cookie_consent(page, timeout_ms: int = 3000) -> int:
    """Try to dismiss cookie consent banners and HCP gates.

    Iterates through known selectors and clicks the first visible, enabled
    button that matches.  Returns the number of buttons clicked (0-2 typically,
    since some sites have both a cookie banner AND an HCP gate).

    Args:
        page: Playwright Page object.
        timeout_ms: Max time to wait for each selector to appear.

    Returns:
        Number of consent buttons successfully clicked.
    """
    clicked = 0
    for selector in _CONSENT_BUTTON_SELECTORS:
        try:
            locator = page.locator(selector).first
            # Wait briefly for the element — many banners appear after a short
            # JS delay, but we don't want to block too long per selector.
            await locator.wait_for(state="visible", timeout=timeout_ms)
            if await locator.is_enabled():
                await locator.click()
                clicked += 1
                logger.info(
                    "[http_client] Clicked consent button: %s", selector
                )
                # After clicking, wait a beat for the overlay to disappear
                # and any gate-protected content to load.
                await page.wait_for_timeout(1000)
                # Some sites have a two-step gate (cookie + HCP), so continue
                # checking remaining selectors.
        except Exception:
            # Element not found or not visible — expected for most selectors
            continue
    return clicked


async def fetch_with_playwright(
    url: str,
    *,
    timeout: float = 30.0,
    wait_until: str = "networkidle",
) -> HttpResponse | None:
    """Render a JS-heavy page via headless Chromium (Playwright).

    Automatically handles:
    - Cookie consent banners (OneTrust, CookieBot, Quantcast, generic)
    - Healthcare Professional (HCP) gate pages (common on pharma sites)
    - JavaScript-rendered content that static fetch can't access

    Returns ``None`` if Playwright is not installed or rendering fails.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.debug(
            "[http_client] playwright not installed – skipping JS render"
        )
        return None

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            ctx = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={"width": 1920, "height": 1080},
            )
            page = await ctx.new_page()
            try:
                resp = await page.goto(
                    url,
                    wait_until=wait_until,
                    timeout=int(timeout * 1000),
                )
                status = resp.status if resp else 200

                # Attempt to dismiss cookie consent / HCP gates so the
                # underlying content becomes accessible.
                consent_clicks = await _dismiss_cookie_consent(page)
                if consent_clicks > 0:
                    # Wait for content to settle after dismissing overlays
                    try:
                        await page.wait_for_load_state(
                            "networkidle", timeout=5000
                        )
                    except Exception:
                        pass  # Best effort — content may already be loaded

                html = await page.content()
                return HttpResponse(
                    status_code=status,
                    headers={},
                    text=html,
                    content=html.encode("utf-8", errors="replace"),
                    url=page.url,
                )
            finally:
                await browser.close()
    except Exception as exc:
        logger.warning(
            "[http_client] Playwright render failed for %s: %s", url, exc
        )
        return None

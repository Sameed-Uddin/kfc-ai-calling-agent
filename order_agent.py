"""
KFC AI Live Voice Order Agent
-----------------------------
A real-time, bidirectional voice agent using the Gemini Live API (gemini-3.8-live)
and Playwright browser automation to order food from KFC over a live voice "phone call".

Usage:
    python order_agent.py                                # Full mode, default devices
    python order_agent.py --input 3 --output 2           # Pick mic & speaker
    python order_agent.py --test                         # Voice-only test (no browser)

Safety Rule: place_order() is hard-blocked unless the customer says yes to the
final question, and it only clicks Checkout. The order is finished in the browser.
"""

import argparse
import asyncio
import os
import re
import sys
import time
import traceback
from collections import deque
from difflib import SequenceMatcher

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
LIVE_MODEL = "gemini-3.8-live"
KFC_URL = os.getenv("KFC_URL", "https://www.kfcpakistan.com")
PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "browser_profile")

INPUT_SAMPLE_RATE = 16000   # Gemini Live expects 16kHz PCM input
OUTPUT_SAMPLE_RATE = 24000  # Gemini Live returns 24kHz PCM output
CHANNELS = 1
CHUNK_SIZE = 1024

# Playback is written 50ms at a time (24kHz, 16-bit mono). Bumping the
# generation tells the speaker loop to drop the chunk it is playing.
PLAYBACK_SLICE_BYTES = 2400
_playback_generation = 0

# Upper limit on clicks when emptying a leftover cart
MAX_CART_CLEAR_CLICKS = 60

# How close a spoken name must be to a menu or cart name (0 to 1) to use it without asking
MATCH_THRESHOLD = 0.8

# Spoken numbers, so "Midnight Deal two" matches "Midnight Deal 2"
NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}

# Give up quickly on an ADD button that can't be clicked (the agent can't talk while it waits)
ADD_CLICK_TIMEOUT_MS = 3000

# Words ignored by search_menu, and how many results it returns
SEARCH_STOP_WORDS = "deal deals the a an and with which that has have having include includes including all me some item items one ones kfc menu any what whats in of for or too also get got there their something anything like want show find contain contains"
MAX_SEARCH_RESULTS = 8

# The same add_to_cart request again within this many seconds is treated as an accidental repeat
DUPLICATE_ADD_WINDOW_S = 30

# While the agent is speaking (and for a moment after), mic sound that could be its own voice
# coming back is replaced with silence. The bar is the higher of BARGE_IN_RMS (headphone leak
# measured up to ~1600; direct speech reaches 3000-9000) and the learned echo level times
# ECHO_MARGIN, which is what lets it work through speakers too.
BARGE_IN_RMS = 2000
ECHO_WINDOW_S = 0.5
ECHO_MARGIN = 1.5
# Which percentile of measured echo ratios to trust: high enough to cover echo peaks,
# low enough that the customer talking over the agent doesn't count as echo
ECHO_PERCENTILE = 90
# Assume speakers until the agent's first words show how loud its echo really is
START_ECHO_RATIO = 1.0
MIN_ECHO_SAMPLES = 10

# Reconnecting after Gemini drops the connection; the handle lets it keep the conversation
MAX_RECONNECTS = 3
_resume_handle = None


def _sound_key(word: str) -> str:
    """Spells a word by sound, so 'Crunch' matches 'Krunch' and 'Extreme' matches 'Xtreme'."""
    word = word.rstrip("s")
    if word.startswith("ex"):
        word = "x" + word[2:]
    return word.replace("ph", "f").replace("c", "k")


def _sound_words(text: str) -> list[str]:
    """Lowercase sound keys for each word of a menu name, with spoken numbers as digits."""
    return [NUMBER_WORDS.get(word, _sound_key(word)) for word in re.findall(r"[a-z0-9]+", text.lower())]


def _match_score(name: str, title: str) -> float:
    """Scores 0 to 1: how closely every word of name matches some word of title."""
    name_words, title_words = _sound_words(name), _sound_words(title)
    if not name_words or not title_words:
        return 0.0
    return sum(
        max(SequenceMatcher(None, word, other).ratio() for other in title_words)
        for word in name_words
    ) / len(name_words)


def _best_matches(name: str, titles: list[str]) -> list[str]:
    """Titles most like name, best first. Ties go to the shortest (Burger before Burger + Drink)."""
    return sorted(titles, key=lambda title: (-_match_score(name, title), len(title)))


def _pick_title(name: str, titles: list[str]) -> tuple:
    """Returns (title, []) for one clear match, or (None, closest titles) to ask the customer."""
    ranked = _best_matches(name, titles)
    if not ranked or _match_score(name, ranked[0]) < MATCH_THRESHOLD:
        return None, ranked[:3]
    best_score = _match_score(name, ranked[0])
    tied = [title for title in ranked if _match_score(name, title) == best_score]
    exact = [title for title in tied if sorted(_sound_words(title)) == sorted(_sound_words(name))]
    if exact:
        return exact[0], []
    if len(tied) > 1:
        return None, tied[:5]
    return tied[0], []


def _rms(raw_bytes: bytes) -> float:
    """Loudness (RMS) of a 16-bit PCM chunk."""
    audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0


class EchoGate:
    """Keeps the agent from hearing its own voice, with headphones or speakers.

    It learns how loud the agent's voice comes back into the mic compared with how loud it
    was played (the echo ratio). Headphones leak little, so normal speech gets through;
    speakers echo a lot, so you need to speak clearly over the agent to interrupt it."""
    def __init__(self):
        self.played = deque(maxlen=40)   # (time, rms) of recently played 50ms slices
        self.ratios = deque(maxlen=200)  # mic rms / played rms while the agent talks
        self.reported = False
        self.latency = 0.0               # seconds between writing audio and hearing it

    def on_played(self, chunk: bytes):
        self.played.append((time.monotonic(), _rms(chunk)))

    def echo_ratio(self) -> float:
        if len(self.ratios) < MIN_ECHO_SAMPLES:
            return START_ECHO_RATIO
        # A high percentile covers echo peaks but ignores the few moments the customer talks over the agent
        return float(np.percentile(self.ratios, ECHO_PERCENTILE))

    def filter(self, raw: bytes, mic_rms: float) -> bytes:
        """Returns the mic chunk, or silence if it is probably the agent's echo."""
        now = time.monotonic()
        # A slice can be heard any time from when it's written until the speaker delay plus room echo has passed
        recent = [rms for played_at, rms in self.played if now - played_at < self.latency + ECHO_WINDOW_S]
        if not recent:
            return raw
        played_rms = max(recent)
        if played_rms > 300:
            self.ratios.append(mic_rms / played_rms)
            if not self.reported and len(self.ratios) >= MIN_ECHO_SAMPLES:
                self.reported = True
                ratio = self.echo_ratio()
                print(f"\n🔊 Echo ratio {ratio:.2f}: sounds like {'strong echo, speak up to interrupt' if ratio > 0.2 else 'low echo, normal speech can interrupt'}.")
        if mic_rms >= max(BARGE_IN_RMS, self.echo_ratio() * played_rms * ECHO_MARGIN):
            return raw
        return bytes(len(raw))


echo_gate = EchoGate()


def _print_mic_level(raw_bytes: bytes) -> float:
    """Compute RMS of a 16-bit PCM chunk, print a live meter bar, and return the RMS."""
    rms = _rms(raw_bytes)
    # Scale to 0-30 character bar (max int16 RMS ~32768, speech ~1000-5000)
    level = int(rms / 1500 * 30)
    level = min(level, 30)
    bar = "▇" * level + "░" * (30 - level)
    sys.stdout.write(f"\r🎤 Mic [{bar}] RMS:{int(rms):>5}  ")
    sys.stdout.flush()
    return rms


class OrderSessionState:
    """Tracks order progress and strictly enforces the place_order safety rule."""
    AFFIRMATIVE = re.compile(r"\b(yes|yeah|yep|sure|place it|place order|go ahead|do it)\b")
    NEGATIVE = re.compile(r"\b(no|not|nah|don'?t|wait|stop|cancel|change|hold on|unsure)\b")

    def __init__(self):
        self.total_read = False
        self.last_total = "N/A"
        self.final_question_asked = False
        self.user_said_yes_after_total = False
        self.order_placed = False
        self.agent_speech = ""

    def on_total_read(self, total: str):
        """Called when get_receipt_total is executed."""
        self.total_read = True
        self.last_total = total
        # Reset any prior yes: customer must confirm AFTER hearing this total
        self.final_question_asked = False
        self.user_said_yes_after_total = False
        print(f"\n[🛡️ Safety Guard] Total recorded as {total}. Waiting for customer confirmation.")

    def on_cart_changed(self):
        """Called when items are added: the old total and any yes no longer apply."""
        self.total_read = False
        self.final_question_asked = False
        self.user_said_yes_after_total = False

    def on_agent_transcript(self, text: str):
        """Watches for the final question, which arrives split across transcript fragments."""
        self.agent_speech = (self.agent_speech + " " + text.lower())[-120:]
        spoken = " ".join(self.agent_speech.split())
        if self.total_read and not self.final_question_asked and "place this order" in spoken:
            self.final_question_asked = True
            self.user_said_yes_after_total = False
            print("\n[🛡️ Safety Guard] Final question asked. Waiting for YES.")

    def on_user_transcript(self, text: str):
        """Monitors user's spoken words for explicit confirmation."""
        if not text or not self.final_question_asked:
            return
        clean = text.lower().strip()

        if self.NEGATIVE.search(clean):
            self.user_said_yes_after_total = False
            print("[🛡️ Safety Guard] Customer expressed hesitation or NO. place_order is blocked.")
        elif self.AFFIRMATIVE.search(clean):
            self.user_said_yes_after_total = True
            print(f"[🛡️ Safety Guard] Explicit YES confirmed after hearing total ({self.last_total}).")


state = OrderSessionState()


class KFCBrowserController:
    """Controls the visible Chrome browser instance for KFC ordering."""
    def __init__(self, kfc_url: str, profile_dir: str):
        self.kfc_url = kfc_url
        self.profile_dir = profile_dir
        self.playwright = None
        self.context = None
        self.page = None
        self.menu_titles = []
        self.menu_index = []
        self.last_add = None
        self.last_add_time = 0.0

    async def ensure_browser(self):
        """Opens visible Chrome browser if not already open."""
        if self.page is None:
            print("\n🌐 Opening Google Chrome (the agent's saved profile, so logins stay)...")
            self.playwright = await async_playwright().start()
            # Chrome won't let automation use your everyday profile, so the agent keeps its own
            # in browser_profile/. Log in to KFC there once and it stays logged in.
            options = dict(
                user_data_dir=self.profile_dir,
                headless=False,
                no_viewport=True,
                args=["--start-maximized"],
                ignore_default_args=["--enable-automation"],
            )
            try:
                self.context = await self.playwright.chromium.launch_persistent_context(channel="chrome", **options)
            except Exception as e:
                print(f"⚠️ Google Chrome didn't start ({type(e).__name__}), using Playwright's browser instead.")
                self.context = await self.playwright.chromium.launch_persistent_context(**options)
            self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()

    async def _open(self, path: str):
        """Opens a page of the KFC site and lets it load."""
        await self.page.goto(f"{self.kfc_url}/{path}", wait_until="domcontentloaded")
        await self.page.wait_for_timeout(3000)

    async def start_order(self):
        """Opens Chrome on an empty cart when the call starts."""
        await self.ensure_browser()
        await self.clear_cart()
        await self._open("menu")
        self.menu_titles = await self._search_menu("")
        self.menu_index = await self._read_menu_index()
        print(f"📋 Read {len(self.menu_titles)} items from the menu.")

    async def clear_cart(self):
        """Empties items left in the saved profile so they can't end up in this order."""
        await self._open("checkout")
        for _ in range(MAX_CART_CLEAR_CLICKS):
            if await self._cart_count() == 0:
                break
            # The donation box has one of these too, but cart rows come first
            await self.page.locator('button[aria-label="Decrease quantity"]').first.click()
            await self.page.wait_for_timeout(600)

    async def _cart_count(self) -> int:
        """Reads the item count from the '5 Items Rs 1,055 GO TO CART' or Checkout button."""
        for text in await self.page.locator('button:has-text("Item")').all_inner_texts():
            match = re.search(r"(\d+)\s*Items?\b", text)
            if match:
                return int(match.group(1))
        return 0

    def _cart_rows(self):
        """Cart rows on the checkout page (the donation box also has a Decrease button)."""
        decrease = self.page.locator('button[aria-label="Decrease quantity"]')
        details = self.page.locator("button", has_text="Details")
        return self.page.locator("div.p-5").filter(has=decrease).filter(has=details)

    async def _read_cart_rows(self) -> list[dict]:
        """Returns [{'item': 'Mayo Dip', 'quantity': 2}, ...] from the checkout page."""
        rows = []
        for text in await self._cart_rows().all_inner_texts():
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            rows.append({"item": lines[0], "quantity": int(lines[-1])})
        return rows

    async def read_cart(self) -> dict:
        """Opens the checkout page and reports what is really in the cart."""
        await self.ensure_browser()
        print("\n[Browser Action] Reading cart...")
        await self._open("checkout")
        return {"status": "success", "cart": await self._read_cart_rows()}

    async def update_cart_item(self, item: str, quantity: int) -> dict:
        """Sets an item already in the cart to quantity (0 removes it), then re-reads the cart."""
        await self.ensure_browser()
        quantity = int(quantity)
        print(f"\n[Browser Action] Setting '{item}' to quantity {quantity}...")
        await self._open("checkout")

        cart = await self._read_cart_rows()
        title, closest = _pick_title(item, [row["item"] for row in cart])
        if title is None:
            message = f"'{item}' matches several cart items: {closest}. Ask which one." if len(closest) > 1 and _match_score(item, closest[0]) >= MATCH_THRESHOLD else f"'{item}' is not in the cart."
            return {"status": "failed", "message": message, "cart": cart}
        target = next(row for row in cart if row["item"] == title)

        state.on_cart_changed()
        row = self._cart_rows().filter(has_text=target["item"]).first
        label = "Increase quantity" if quantity > target["quantity"] else "Decrease quantity"
        for _ in range(abs(quantity - target["quantity"])):
            await row.locator(f'button[aria-label="{label}"]').click()
            await self.page.wait_for_timeout(800)

        cart = await self._read_cart_rows()
        actual = next((row["quantity"] for row in cart if row["item"] == target["item"]), 0)
        return {
            "status": "success" if actual == quantity else "failed",
            "item": target["item"],
            "quantity_now": actual,
            "cart": cart,
            "instruction": "Tell the customer what the cart holds now, using this result. If you already read a total, it is out of date."
        }

    def _menu_cards(self):
        """Menu item cards currently shown."""
        return self.page.locator("div.bg-cardbg").filter(has=self.page.locator("button", has_text="ADD"))

    async def _find_menu_card(self, title: str):
        """Returns the shown menu card with exactly this title, or None."""
        cards = self._menu_cards()
        for i in range(await cards.count()):
            card = cards.nth(i)
            if (await card.inner_text()).strip().split("\n")[0].strip() == title:
                return card
        return None

    async def _section_start_time(self, card) -> str | None:
        """Returns '12 AM' when the card's menu section is titled like 'MIDNIGHT (START AT 12 AM)'."""
        section = card.locator('xpath=ancestor::div[contains(@class, "scroll-mt")][1]')
        if await section.count() == 0:
            return None
        heading = (await section.inner_text()).strip().split("\n")[0]
        match = re.search(r"START AT ([^)]+)", heading, re.IGNORECASE)
        return match.group(1).strip() if match else None

    async def _search_menu(self, term: str) -> list[str]:
        """Filters the menu by term and returns the titles shown."""
        search_input = self.page.locator('input[placeholder*="Search"]').first
        await search_input.fill("")
        await search_input.fill(term)
        await self.page.wait_for_timeout(2000)
        return [text.strip().split("\n")[0].strip() for text in await self._menu_cards().all_inner_texts()]

    async def _read_menu_index(self) -> list[dict]:
        """Name, price, section and description of every menu item (the description is the picture's alt text)."""
        return await self.page.evaluate("""() =>
            [...document.querySelectorAll('div[class*="scroll-mt"]')].flatMap(section => {
                const heading = section.innerText.trim().split("\\n")[0];
                return [...section.querySelectorAll("div.bg-cardbg")].map(card => {
                    const lines = card.innerText.split("\\n").map(line => line.trim()).filter(Boolean);
                    const alts = [...card.querySelectorAll("img[alt]")].map(img => img.alt).filter(alt => alt && alt !== lines[0]);
                    return {
                        item: lines[0],
                        price: (lines.find(line => /^RS/i.test(line)) || "").replace(/^RS\\s*/i, "") + " rupees",
                        section: heading,
                        about: alts.sort((a, b) => b.length - a.length)[0] || "",
                    };
                });
            })""")

    async def search_menu(self, query: str) -> dict:
        """Finds menu items whose name, section or description mention the query's words."""
        await self.ensure_browser()
        print(f"\n[Browser Action] Searching the menu for: '{query}'")
        if not self.menu_index:
            if "/menu" not in self.page.url:
                await self._open("menu")
            await self._search_menu("")
            self.menu_index = await self._read_menu_index()

        stop_words = set(_sound_words(SEARCH_STOP_WORDS))
        words = [word for word in _sound_words(query) if word not in stop_words]
        if not words:
            return {"status": "failed", "message": "Say what to look for, like 'zinger strips' or 'drinks'."}

        def hits(entry):
            text = _sound_words(f"{entry['item']} {entry['section']} {entry['about']}")
            return sum(any(SequenceMatcher(None, word, other).ratio() >= MATCH_THRESHOLD for other in text) for word in words)

        scored = [(hits(entry), entry) for entry in self.menu_index]
        best = max(score for score, _ in scored)
        if best == 0:
            return {"status": "failed", "message": f"Nothing on the menu mentions '{query}'."}
        found = [entry for score, entry in scored if score == best][:MAX_SEARCH_RESULTS]
        return {
            "status": "success",
            "matched_all_words": best == len(words),
            "items": found,
            "instruction": "Tell the customer the matching items with prices, briefly saying what's in each. If matched_all_words is false, say these are the closest matches. Do not add anything until they choose."
        }

    async def show_menu(self, section: str = "") -> dict:
        """Opens the full menu in Chrome, scrolls to a section, and returns what that section shows."""
        await self.ensure_browser()
        print(f"\n[Browser Action] Showing menu section: '{section or 'top'}'")
        if "/menu" not in self.page.url:
            await self._open("menu")
        await self._search_menu("")

        sections = self.page.locator('div[class*="scroll-mt"]')
        headings = [text.strip().split("\n")[0] for text in await sections.all_inner_texts()]
        if not section:
            await self.page.evaluate("window.scrollTo(0, 0)")
            return {"status": "success", "sections": headings, "instruction": "Tell the customer the menu is open in the browser and read out the section names."}

        letters = lambda text: re.sub(r"[^a-z]", "", text.lower())
        wanted = letters(section)
        index = next((i for i, h in enumerate(headings) if wanted and (wanted in letters(h) or letters(h) in wanted)), None)
        if index is None:
            return {"status": "failed", "message": f"No menu section called '{section}'.", "sections": headings}

        await sections.nth(index).scroll_into_view_if_needed()
        items = []
        for text in await sections.nth(index).locator("div.bg-cardbg").all_inner_texts():
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            price = next((line for line in lines if line.upper().startswith("RS")), "")
            items.append(f"{lines[0]} ({price[2:].strip()} rupees)" if price else lines[0])
        return {
            "status": "success",
            "section": headings[index],
            "items": items,
            "instruction": "Tell the customer this section is showing in the browser. Read out a few items with prices, not the whole list, unless they ask. Do not add anything until they order."
        }

    async def add_to_cart(self, items: list[str]) -> dict:
        """Adds items matched by name, checking the cart count after each one."""
        await self.ensure_browser()
        items = [items] if isinstance(items, str) else items
        print(f"\n[Browser Action] Adding items to cart: {items}")

        if "/menu" not in self.page.url:
            await self._open("menu")

        if not self.menu_titles:
            self.menu_titles = await self._search_menu("")

        # The model has re-sent the same add when the customer only said "can you hear me?"
        request = sorted(str(item).strip().lower() for item in items)
        if request == self.last_add and time.monotonic() - self.last_add_time < DUPLICATE_ADD_WINDOW_S:
            print(f"⚠️ Blocked a repeat of the add just done: {items}")
            self.last_add_time = 0.0
            return {
                "status": "blocked",
                "message": "This exact add was done a moment ago, so nothing was added again.",
                "instruction": "Do not say anything was added. If the customer clearly wants more of it, call add_to_cart again or use update_cart_item with the new total."
            }

        state.on_cart_changed()
        before = {row["item"]: row["quantity"] for row in await self._read_cart_rows()}
        added, problems = [], []
        for raw_item in items:
            # Parse quantity and item name: e.g. "2 Zinger Burgers" -> qty=2, name="Zinger Burger"
            qty = 1
            name = str(raw_item).strip()
            match = re.match(r"^(\d+)\s*[xX]?\s+(.*)$", name)
            if match:
                qty = int(match.group(1))
                name = match.group(2).strip()

            title, closest = _pick_title(name, self.menu_titles)
            if title is None:
                if closest and _match_score(name, closest[0]) >= MATCH_THRESHOLD:
                    print(f"⚠️ '{name}' matches several items: {closest}")
                    problems.append(f"'{name}' matches several menu items: {closest}. Nothing was added. Ask which one they want.")
                else:
                    print(f"⚠️ Could not find '{name}' on the menu.")
                    problems.append(f"'{name}' was not found. Closest real menu items: {closest}")
                continue

            print(f"🍔 Heard '{name}', menu item is '{title}' (Qty: {qty})...")
            await self._search_menu(title)
            card = await self._find_menu_card(title)
            if card is None:
                print(f"⚠️ '{title}' did not appear in the menu search.")
                problems.append(f"'{title}' could not be opened on the menu page.")
                continue

            count_before = await self._cart_count()
            clickable = True
            for _ in range(qty):
                try:
                    await card.locator("button", has_text="ADD").first.click(timeout=ADD_CLICK_TIMEOUT_MS)
                except PlaywrightTimeoutError:
                    clickable = False
                    break
                await self.page.wait_for_timeout(1200)

                # Decline the "You May Also Like" popup: its ADD TO BUCKET charges for an extra item
                no_thanks = self.page.locator('[role="dialog"] button:has-text("NO THANKS")').first
                if await no_thanks.count() > 0 and await no_thanks.is_visible():
                    await no_thanks.click()
                    await self.page.wait_for_timeout(800)

            gained = await self._cart_count() - count_before
            if gained == qty:
                added.append((title, qty))
            elif not clickable:
                print(f"⚠️ '{title}' can't be added right now.")
                start_time = await self._section_start_time(card)
                reason = f"It is only available starting at {start_time}." if start_time else "Its ADD button doesn't work."
                problems.append(f"{title} can't be added right now. {reason} The cart changed by {gained}.")
            else:
                print(f"⚠️ Wanted {qty}x {title}, cart changed by {gained}.")
                problems.append(f"{title}: wanted {qty} but the cart changed by {gained}. It may need options chosen in the browser.")
                # A dialog left open would block the next item
                await self.page.keyboard.press("Escape")
                await self.page.wait_for_timeout(500)

        # The item counter has disagreed with the cart before, so confirm against the cart rows
        expected = {}
        for title, qty in added:
            expected[title] = expected.get(title, 0) + qty
        await self.page.wait_for_timeout(1000)
        after = {row["item"]: row["quantity"] for row in await self._read_cart_rows()}
        added = []
        for title, qty in expected.items():
            got = after.get(title, 0) - before.get(title, 0)
            if got == qty:
                added.append(f"{qty}x {title}")
            else:
                print(f"⚠️ Cart shows {got}x {title} added, expected {qty}.")
                problems.append(f"{title}: the cart shows {got} added instead of {qty}.")
        if added:
            self.last_add, self.last_add_time = request, time.monotonic()
        # Leave the full menu showing instead of the last search result
        await self.page.locator('input[placeholder*="Search"]').first.fill("")

        return {
            "status": "success" if added and not problems else "partial" if added else "failed",
            "added_items": added,
            "problems": problems,
            "instruction": "Tell the customer exactly what was added and any problems, then ask if they want anything else. Do NOT go to checkout yet."
        }

    async def get_receipt_total(self) -> dict:
        """Navigates to checkout and reads the receipt subtotal, delivery fee, and total."""
        await self.ensure_browser()
        print("\n[Browser Action] Checking cart and reading receipt total...")

        await self._open("checkout")

        text = await self.page.locator('body').inner_text()
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        labels = {"sub total": "subtotal", "delivery": "delivery_fee", "total": "total"}
        receipt = dict.fromkeys(labels.values(), "N/A")
        for line, next_line in zip(lines, lines[1:]):
            if line.lower() in labels:
                receipt[labels[line.lower()]] = next_line

        state.on_total_read(receipt["total"])
        print(f"🧾 Subtotal: {receipt['subtotal']} | Delivery: {receipt['delivery_fee']} | Total: {receipt['total']}")

        # The voice has read "Rs" as dollars, so spell the currency out
        spoken = {key: re.sub(r"^Rs\.?\s*", "", value, flags=re.IGNORECASE) + " rupees" if value != "N/A" else value
                  for key, value in receipt.items()}
        return {
            "cart": await self._read_cart_rows(),
            **spoken,
            "instruction": "Read the total clearly to the customer in rupees (never dollars), then continue with the payment and instructions questions."
        }

    async def set_payment_method(self, cash_on_delivery: bool = True) -> dict:
        """Sets payment preference (Cash on Delivery vs Card)."""
        method = "Cash on Delivery" if cash_on_delivery else "Card / Online"
        print(f"\n[Browser Action] Setting payment method to: {method}")
        return {
            "status": "success",
            "payment_method": method,
            "message": f"Payment method noted as {method}."
        }

    async def set_instructions(self, text: str) -> dict:
        """Enters cooking or delivery instructions into checkout."""
        await self.ensure_browser()
        print(f"\n[Browser Action] Setting special instructions: '{text}'")
        field = self.page.locator('input[placeholder*="Instructions"], textarea[placeholder*="Instructions"]').first
        if await field.count() == 0 or not await field.is_visible():
            return {"status": "failed", "message": "The instructions box was not found on this page."}
        await field.fill(text)
        return {
            "status": "success",
            "instructions": text,
            "message": "Special instructions saved."
        }

    async def place_order(self) -> dict:
        """Places the final order (Strictly blocked if safety guard not satisfied)."""
        await self.ensure_browser()
        print("\n[Browser Action] place_order requested by model. Checking safety guard...")

        if state.order_placed:
            return {"status": "blocked", "error": "SAFETY BLOCK: Checkout was already started for this order."}

        if not state.total_read or not re.search(r"[1-9]", state.last_total):
            print("🛑 [SAFETY RULE BLOCKED] Total has not been checked or cart is empty.")
            return {
                "status": "blocked",
                "error": "SAFETY BLOCK: The receipt total has not been read to the customer yet. Call get_receipt_total first."
            }

        if not state.user_said_yes_after_total:
            print("🛑 [SAFETY RULE BLOCKED] Customer has not confirmed YES after total was read.")
            return {
                "status": "blocked",
                "error": "SAFETY BLOCK: Customer has NOT clearly said 'yes' to place the order after hearing the total. You must ask: 'Should I place this order?' and only call place_order when they say YES."
            }

        print("\n🚀 [SAFETY CHECK PASSED] Going to checkout in browser...")
        try:
            checkout_btn = self.page.locator('button:has-text("Checkout")').first
            if await checkout_btn.is_visible():
                await checkout_btn.click()
                state.order_placed = True
                print("✅ Checkout clicked. Finish the order in the browser window.")
                return {"status": "checkout_clicked", "message": "Checkout was clicked. The order is NOT confirmed. Tell the customer to finish it in the browser window."}
            else:
                return {"status": "manual_click_needed", "message": "The checkout button was not found. Tell the customer to finish in the browser window."}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    async def close(self):
        """Closes Playwright browser cleanly."""
        if self.context:
            await self.context.close()
        if self.playwright:
            await self.playwright.stop()


browser_ctrl = KFCBrowserController(KFC_URL, PROFILE_DIR)


def add_to_cart(items: list[str]) -> dict:
    """Adds a list of items to the KFC cart. Example items: ['1 Zinger Burger', '2 Fries']."""

def update_cart_item(item: str, quantity: int) -> dict:
    """Sets an item already in the cart to a new total quantity. Use quantity 0 to remove it. Example: to keep one Mayo Dip, use item 'Mayo Dip' and quantity 1."""

def read_cart() -> dict:
    """Returns what is really in the cart right now."""

def search_menu(query: str) -> dict:
    """Searches every menu item's name, section and description (what's inside it) for these words. Example queries: 'zinger strips', 'wings', 'coleslaw', 'family'. Returns matching items with prices and descriptions."""

def show_menu(section: str) -> dict:
    """Shows the KFC menu in the customer's browser and returns the items and prices in that section. Sections: Promotion, Everyday Value, Ala-Carte & Combos, Signature Boxes, Sharing, Condiments (dips and sauces), Snacks & Beverages (drinks, fries, desserts), Midnight. Use an empty section to show the top of the menu and list the sections."""

def get_receipt_total() -> dict:
    """Goes to the KFC checkout page and returns the subtotal, delivery fee, and total. Only call this once the customer says they are done adding items."""

def set_payment_method(cash_on_delivery: bool) -> dict:
    """Sets whether the customer is paying with Cash on Delivery (True) or Card/Online (False)."""

def set_instructions(text: str) -> dict:
    """Sets any special cooking or delivery instructions for the order."""

def place_order() -> dict:
    """Clicks Checkout so the customer can finish in the browser. WARNING: Only call this if the customer says YES to "Should I place this order?"."""

ORDERING_TOOLS = [add_to_cart, update_cart_item, read_cart, search_menu, show_menu, get_receipt_total, set_payment_method, set_instructions, place_order]
TOOL_NAMES = {tool.__name__ for tool in ORDERING_TOOLS}


SYSTEM_PROMPT = """You are a friendly, natural voice food-ordering assistant for KFC.
You are talking with the customer over a live telephone call.
Keep your spoken responses short, friendly, and conversational (like a drive-thru or phone agent).
Only say what the customer should hear. Never say your plans, reasoning, or notes about the customer out loud.
This is KFC Pakistan: every price is in Pakistani rupees. Always say "rupees", never "dollars".

Your order workflow:
1. Greet the customer warmly and ask what they would like to order.
2. As soon as the customer names items, call the `add_to_cart` tool with the items and quantities.
3. Say exactly what was added, using the names in the tool result. If something failed or was not found, say so and offer only the closest real menu items from the result. Then ask if they want anything else.
4. Repeat steps 2 and 3 for as long as they keep adding items. Never call `get_receipt_total` while they are still ordering.
5. When they say they are done, call `get_receipt_total` and clearly read the total price to the customer.
6. Ask if they prefer Cash on Delivery or Card, and call `set_payment_method`.
7. Ask if they have any special cooking or delivery instructions, and call `set_instructions` if they do.
8. Finally, explicitly ask: "Should I place this order?"
9. STRICT RULE: ONLY call `place_order` if the customer clearly says YES to that question. If they say no, hesitate, or ask a question, do NOT call place_order.
10. `place_order` only clicks Checkout. Never say the order is placed. Tell the customer to finish it in the browser window.

Rules for the whole call:
- Changing the cart: when the customer wants to remove an item or change a quantity, at any step, call `update_cart_item` with the total quantity they want. If you already read a total, call `get_receipt_total` again afterwards and read the new total.
- Never claim the cart changed unless a tool result in this conversation shows it. If there is no tool for what they asked, say you can't do it.
- Menu names: you do not know the menu, so never say an item is missing and never suggest an item unless a tool result shows it. When the customer corrects a name or its spelling, call `add_to_cart` again with the corrected name, every time, even if the last attempt failed.
- Keep numbers in item names: "Deal 2" is a different item from "Deal 1". Only for numbered items like "Deal 2" or "Chicky Meal 2", the word "to" or "too" right after the name may be a misheard "2". Otherwise "too" means "also": "a Pepsi too" is one Pepsi. If you are not sure of the number, ask before adding. If a tool result says a name matches several items, read those items out and ask which one.
- If an item can't be added right now, say that plainly and tell the customer the reason from the tool result, including any start time. Do not make up a reason.
- Check before guessing: if you did not clearly catch an item, ask the customer to repeat it. Vague or odd phrases like "all kinds of mayo" or "some sauces" are not an order: ask exactly which item and how many before calling `add_to_cart`. Never invent a quantity, except that "a Pepsi" or just "Pepsi" means one.
- Finding items: when the customer asks what's in a deal, which items include something ("deals with strips and Zinger"), or describes an item without its name, call `search_menu` with the key food words. Do not guess a section instead.
- Showing the menu: when the customer wants to see the menu, a section, or what's available ("open the menu", "show me the drinks", "go to deals"), call `show_menu` with the closest section. You can show the menu; never say you can't.
- "No" is not always "done": treat them as done only when they clearly say they want nothing more or want to check out. A "no" that answers something else, or that shows they are annoyed ("no, man", "no, open the menu"), is not done: do what they asked instead. "No, keep one only" is a cart change. If you are not sure which they mean, ask.
- Follow the customer: if they ask for something else while you are on the payment or instructions questions, do that first, then come back. Do not repeat the payment question over their request.
- Only add when they order: "can you hear me?", "hello?", or "I did it" is not an order, so never call `add_to_cart` again for it. Answer them instead.
- Noise: if you hear a stray word, a fragment, or words that repeat what you just said, it is probably background noise or your own voice. Do not greet again and do not act on it. Briefly ask "Sorry, what was that?"
"""


async def mic_stream_loop(session, input_queue: asyncio.Queue, stop_event: asyncio.Event):
    """Streams microphone audio chunks from input_queue to Gemini Live WebSocket."""
    while not stop_event.is_set():
        try:
            chunk = await asyncio.wait_for(input_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        # A send failure means the connection dropped, so let it raise and trigger a reconnect
        await session.send_realtime_input(audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000"))


async def speaker_playback_loop(output_stream: sd.RawOutputStream, output_queue: asyncio.Queue, stop_event: asyncio.Event):
    """Plays received 24kHz PCM audio chunks through speakers."""
    while not stop_event.is_set():
        try:
            audio_data = await asyncio.wait_for(output_queue.get(), timeout=0.05)
            generation = _playback_generation
            # Write in short slices so an interruption takes effect almost immediately
            for offset in range(0, len(audio_data or b""), PLAYBACK_SLICE_BYTES):
                if generation != _playback_generation:
                    break
                chunk = audio_data[offset:offset + PLAYBACK_SLICE_BYTES]
                await asyncio.to_thread(output_stream.write, chunk)
                echo_gate.on_played(chunk)
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            if not stop_event.is_set():
                print(f"\n❌ Speaker Playback Error [{type(e).__name__}]: {e}")
                traceback.print_exc()
            break


async def receive_server_loop(session, output_queue: asyncio.Queue, stop_event: asyncio.Event, enable_tools: bool):
    """Keeps receiving turn after turn (session.receive() ends after each model turn)."""
    while not stop_event.is_set():
        got_messages = await receive_turn(session, output_queue, stop_event, enable_tools)
        if not got_messages:
            raise ConnectionError("Gemini closed the connection")


async def receive_turn(session, output_queue: asyncio.Queue, stop_event: asyncio.Event, enable_tools: bool) -> bool:
    """Receives one model turn: audio, transcripts, and tool calls from Gemini."""
    global _playback_generation, _resume_handle
    got_messages = False
    async for message in session.receive():
        got_messages = True
        if stop_event.is_set():
            break

        if message.session_resumption_update and message.session_resumption_update.new_handle:
            _resume_handle = message.session_resumption_update.new_handle

        # 1. Handle Server Content (Audio, Interruption, Transcripts)
        if message.server_content:
            sc = message.server_content

            # Handle Interruption (barge-in): Flush speaker queue immediately
            if sc.interrupted:
                print("\n[⚡ Interrupted by user]")
                while not output_queue.empty():
                    output_queue.get_nowait()
                # Don't abort the stream here: it crashes a write running in the playback thread
                _playback_generation += 1

            if sc.input_transcription and sc.input_transcription.text:
                user_text = sc.input_transcription.text.strip()
                if user_text:
                    print(f"\n🗣️ [You]: {user_text}")
                    state.on_user_transcript(user_text)

            if sc.output_transcription and sc.output_transcription.text:
                agent_text = sc.output_transcription.text.strip()
                if agent_text:
                    print(f"🤖 [KFC Agent]: {agent_text}")
                    state.on_agent_transcript(agent_text)

            if sc.model_turn and sc.model_turn.parts:
                for part in sc.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        output_queue.put_nowait(part.inline_data.data)

        # 2. Handle Tool Calls
        if message.tool_call and enable_tools:
            for fc in message.tool_call.function_calls:
                print(f"\n🔧 [Tool Invoked]: {fc.name}({fc.args})")

                try:
                    if fc.name not in TOOL_NAMES:
                        raise ValueError(f"Unknown tool: {fc.name}")
                    result = await getattr(browser_ctrl, fc.name)(**fc.args)
                except Exception as e:
                    print(f"\n❌ Tool execution error [{type(e).__name__}]: {e}")
                    traceback.print_exc()
                    result = {"error": str(e)}

                print(f"📦 [Tool Result]: {result}")
                await session.send_tool_response(
                    function_responses=[types.FunctionResponse(name=fc.name, id=fc.id, response=result)]
                )

    return got_messages


async def run_connection(session, input_queue: asyncio.Queue, output_queue: asyncio.Queue, stop_event: asyncio.Event, enable_tools: bool):
    """Runs the mic and receive loops for one connection. Raises if the connection drops."""
    tasks = [
        asyncio.create_task(mic_stream_loop(session, input_queue, stop_event)),
        asyncio.create_task(receive_server_loop(session, output_queue, stop_event, enable_tools)),
    ]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    # Collect the other loop's outcome too, so its error isn't reported as "never retrieved"
    await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        task.result()


async def _reconnect_note(enable_tools: bool) -> str:
    """Tells the model the line dropped, with the real cart in case it lost the conversation."""
    cart = (await browser_ctrl.read_cart())["cart"] if enable_tools else []
    return (
        "(System note, not the customer: the call connection dropped for a moment and is back. "
        f"The real cart right now is: {cart}. Do not greet again. Briefly say the line dropped, "
        "and continue from where you were. If you don't remember, read back the cart and ask what they'd like next.)"
    )


async def run_live_voice_session(enable_tools: bool = True, input_device=None, output_device=None):
    """Sets up sounddevice streams and runs the Gemini Live session."""
    global _resume_handle
    if not API_KEY or API_KEY.strip() == "" or API_KEY == "your_gemini_api_key_here":
        print("\n❌ Error: GEMINI_API_KEY is not set in your .env file.")
        return

    print("🍗 KFC Real-Time Voice Agent")
    print(f"   Model: {LIVE_MODEL}")
    print(f"   Mode:  {'Full (KFC browser)' if enable_tools else 'Voice test (no browser)'}")
    print("🛑 Press Ctrl+C at any time to hang up.\n")

    client = genai.Client(api_key=API_KEY)

    # Configure the Multimodal Live Session
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        # Without language codes the transcript sometimes came out in Spanish, Portuguese or Hindi
        # (Urdu speech still showed in Hindi script with "ur-PK" added, so it is English only).
        # (Menu names as custom_vocabulary were tried too, but they turned "a Pepsi too" into "Pepsi Regular".)
        input_audio_transcription=types.AudioTranscriptionConfig(language_codes=["en-US"]),
        # Its planning was once spoken aloud ("I will greet them and ask..."), so keep thoughts out of replies
        thinking_config=types.ThinkingConfig(include_thoughts=False),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        # Low sensitivity so background mic noise doesn't count as the customer interrupting
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                prefix_padding_ms=200,
            )
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Puck"  # Friendly, natural voice
                )
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=SYSTEM_PROMPT if enable_tools else "You are a friendly voice assistant. Greet the user and chat with them.")]
        ),
        tools=ORDERING_TOOLS if enable_tools else None,
    )

    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    # Runs in the audio thread, so it must never block
    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"\n⚠️ Mic status: {status}")
        raw = bytes(indata)
        raw = echo_gate.filter(raw, _print_mic_level(raw))
        loop.call_soon_threadsafe(input_queue.put_nowait, raw)

    try:
        if enable_tools:
            await browser_ctrl.start_order()

        # A device of None means the system default
        stream_args = {"channels": CHANNELS, "dtype": "int16", "blocksize": CHUNK_SIZE}
        input_stream = sd.RawInputStream(samplerate=INPUT_SAMPLE_RATE, device=input_device, callback=mic_callback, **stream_args)
        output_stream = sd.RawOutputStream(samplerate=OUTPUT_SAMPLE_RATE, device=output_device, **stream_args)
        echo_gate.latency = output_stream.latency

        with input_stream, output_stream:
            speaker = asyncio.create_task(speaker_playback_loop(output_stream, output_queue, stop_event))
            for attempt in range(MAX_RECONNECTS + 1):
                config.session_resumption = types.SessionResumptionConfig(handle=_resume_handle)
                try:
                    async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
                        if attempt == 0:
                            print("[✅ Gemini Live session connected]\n")
                            await session.send_realtime_input(text="Hello! I am calling to order food from KFC.")
                        else:
                            print("[✅ Reconnected]\n")
                            await session.send_realtime_input(text=await _reconnect_note(enable_tools))
                        await run_connection(session, input_queue, output_queue, stop_event, enable_tools)
                    break
                except Exception as e:
                    if attempt == MAX_RECONNECTS:
                        raise
                    print(f"\n⚠️ Lost connection to Gemini ({type(e).__name__}: {e}). Reconnecting...")
                    # After a sudden drop Gemini may still hold the old session and refuse to
                    # resume it, so start a fresh one (the reconnect note carries the cart over)
                    if "already connected" in str(e):
                        _resume_handle = None
                    while not input_queue.empty():
                        input_queue.get_nowait()
                    await asyncio.sleep(2 ** attempt)
            speaker.cancel()

    except Exception as e:
        if not stop_event.is_set():
            print(f"\n❌ Error [{type(e).__name__}]: {e}")
            traceback.print_exc()
    finally:
        stop_event.set()
        if enable_tools:
            await browser_ctrl.close()


async def login_once():
    """Opens the agent's Chrome on KFC and waits until the window is closed; the login is saved."""
    await browser_ctrl.ensure_browser()
    await browser_ctrl.page.goto(KFC_URL)
    print("🔑 Log in to KFC and set your delivery address, then close the Chrome window.")
    # On a Mac, Chrome keeps running after its last window closes, so watch the tabs instead
    while browser_ctrl.context.pages:
        await asyncio.sleep(1)
    await browser_ctrl.close()
    print("✅ Saved. The agent will stay logged in from now on.")


def _print_device(label: str, device, default_index: int):
    """Prints the name of the chosen audio device, or the system default."""
    if device is None:
        device = sd.default.device[default_index]
    try:
        print(f"  {label} #{device}: {sd.query_devices(device)['name']}")
    except Exception as e:
        print(f"  ❌ {label} device #{device}: {e}")


def main():
    """CLI entry point with --input, --output, and --test flags."""
    parser = argparse.ArgumentParser(description="KFC Live Voice Ordering Agent")
    parser.add_argument("--input", type=int, default=None,
                        help="Sounddevice input (mic) device ID (e.g., 3)")
    parser.add_argument("--output", type=int, default=None,
                        help="Sounddevice output (speaker) device ID (e.g., 2)")
    parser.add_argument("--test", action="store_true",
                        help="Voice-only test mode (no browser/tools)")
    parser.add_argument("--login", action="store_true",
                        help="Open the agent's Chrome on KFC to log in once, no call")

    args, _ = parser.parse_known_args()

    if args.login:
        asyncio.run(login_once())
        return

    print()
    _print_device("Input ", args.input, 0)
    _print_device("Output", args.output, 1)
    print()

    try:
        asyncio.run(run_live_voice_session(
            enable_tools=not args.test,
            input_device=args.input,
            output_device=args.output,
        ))
    except KeyboardInterrupt:
        print("\n\n📞 Call ended. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

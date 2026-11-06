# KFC AI Calling Agent

Order KFC Pakistan by talking to an AI. You speak, it listens, and it adds items to your cart on the real KFC website in Chrome, then hands checkout over to you.

Built with Python, the Gemini Live API (real-time voice), and Playwright (browser automation).

> Personal project. Not affiliated with or endorsed by KFC.

## What it does

- **Live voice conversation**: talk naturally, and interrupt it mid-sentence like a real person.
- **Adds to the real cart**: finds the item on kfcpakistan.com and clicks ADD, then checks the cart to confirm it actually went in.
- **Understands sloppy speech**: "crunch burger" finds *Krunch Burger*, "midnight deal two" finds *Midnight Deal 2*, and if a name matches several items it asks which one.
- **Menu search**: "which deals have strips and Zinger?" searches every item's description.
- **Cart edits**: "remove the fries", "make it one Pepsi".
- **Reads the bill**, asks cash or card and any special instructions.
- **Never orders on its own**: it only clicks Checkout after you clearly say "yes" to "Should I place this order?", and you press KFC's final order button yourself.
- **Works on speakers or headphones**: an echo filter keeps it from hearing its own voice.
- **Survives dropped connections**: it reconnects and remembers the cart.

## How it works

```
Your mic ──► Gemini Live (listens, decides, talks back) ──► your speakers
                    │
                    │ calls tools: add_to_cart, search_menu, get_receipt_total, ...
                    ▼
             Python + Playwright ──► Google Chrome on kfcpakistan.com
```

Gemini doesn't see the screen. The program gives it a set of tools, Gemini decides which one to call, and Python carries it out in Chrome by reading the page's HTML and clicking buttons.

## Setup

Requires Python 3.11+, Google Chrome, and a mic.

```bash
git clone https://github.com/Sameed-Uddin/kfc-ai-calling-agent.git
cd kfc-ai-calling-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium   # fallback browser if Chrome isn't found

cp .env.example .env          # then put your Gemini API key in .env
```

Get a free Gemini API key at https://aistudio.google.com/app/apikey.

## Run

```bash
python order_agent.py --login   # once: log in to KFC and set your address, then close the window
python order_agent.py           # start a voice call
```

Other options:

- `--test`: voice only, no browser
- `--input N` / `--output N`: choose the mic or speaker by device number

Press **Ctrl+C** to hang up.

## Notes

- The agent keeps its own Chrome profile in `browser_profile/`. Chrome doesn't allow automation of your everyday profile.
- It depends on the KFC website's layout, so a site redesign can break the button-finding code.
- The final "yes" check understands English only.

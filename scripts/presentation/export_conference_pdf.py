#!/usr/bin/env python3
"""
Экспорт conference_presentation.html в PDF.

Reveal.js умеет раскладывать слайды для печати при добавлении ?print-pdf.
Playwright открывает эту версию и сохраняет в PDF.

Установка:
    pip install playwright
    playwright install chromium

Запуск:
    python scripts/presentation/export_conference_pdf.py
"""
import asyncio
import sys
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Установите playwright: pip install playwright")
    print("Затем: playwright install chromium")
    sys.exit(1)

SLIDE_WIDTH = 1024
SLIDE_HEIGHT = 768


async def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    pres_dir = repo_root / "docs" / "presentation"
    html_path = pres_dir / "conference_presentation.html"
    pdf_path = pres_dir / "conference_presentation.pdf"

    if not html_path.exists():
        print(f"Файл не найден: {html_path}")
        sys.exit(1)

    url = f"file://{html_path.absolute()}?print-pdf"

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": SLIDE_WIDTH, "height": SLIDE_HEIGHT}
        )
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_timeout(2000)

        await page.pdf(
            path=str(pdf_path),
            width=f"{SLIDE_WIDTH}px",
            height=f"{SLIDE_HEIGHT}px",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )
        await browser.close()

    print(f"PDF готов: {pdf_path}")
    print(f"Чтобы открыть в Keynote: откройте PDF → File → Export to → PowerPoint / Keynote")


if __name__ == "__main__":
    asyncio.run(main())

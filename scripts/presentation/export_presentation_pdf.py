#!/usr/bin/env python3
"""
Экспорт presentation.html в PDF.
Требуется: pip install playwright && playwright install chromium
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


async def main():
    repo_root = Path(__file__).resolve().parent.parent
    pres_dir = repo_root / "docs" / "presentation"
    html_path = pres_dir / "presentation.html"
    pdf_path = pres_dir / "presentation.pdf"

    if not html_path.exists():
        print(f"Файл не найден: {html_path}")
        sys.exit(1)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1400, "height": 900})
        await page.goto(f"file://{html_path.absolute()}", wait_until="networkidle")
        await page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={"top": "10mm", "right": "10mm", "bottom": "10mm", "left": "10mm"},
        )
        await browser.close()

    print(f"Готово: {pdf_path}")


if __name__ == "__main__":
    asyncio.run(main())

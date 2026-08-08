"""
scrape_prices.py — Dedicated Price Target Oracle Scraper
Uses Playwright to bypass Cloudflare and scrape exact FPL price change targets.
"""
import json
import asyncio
import requests
import difflib
from playwright.async_api import async_playwright

async def run_scraper():
    targets = {}
    print("Initiating FPL Price Target Oracle...")
    
    try:
        # 1. Fetch FPL mapping data to convert scraped names to Player IDs
        print("Fetching official FPL player mapping...")
        resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10)
        resp.raise_for_status()
        elements = resp.json().get('elements', [])
        
        name_to_id = {str(p['web_name']).lower().strip(): p['id'] for p in elements}
        full_name_to_id = {f"{p['first_name']} {p['second_name']}".lower().strip(): p['id'] for p in elements}
        name_pool = list(name_to_id.keys())
        
        # 2. Spin up Headless Chromium
        print("Launching headless Chromium to bypass Cloudflare...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            })
            
            print("Navigating to FPLStatistics...")
            
            # 1. Intercept and block heavy media/ads to drastically speed up page load
            await page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "media", "font"] else route.continue_())
            
            # 2. Use HTTPS and change the wait state to 'domcontentloaded' so it ignores hung ad-trackers
            await page.goto("https://www.fplstatistics.co.uk/", wait_until="domcontentloaded", timeout=60000)
            
            # 3. Wait for the JavaScript DataTables DOM to render
            await page.wait_for_selector("#myDataTable tbody tr", timeout=30000)
            
            # Expand table to show all entries (bypassing pagination)
            try:
                await page.select_option("select[name='myDataTable_length']", value="-1")
                await page.wait_for_timeout(2000)
            except Exception:
                print("Could not expand pagination, scraping visible top 50 rows only.")
            
            rows = await page.query_selector_all("#myDataTable tbody tr")
            print(f"Detected {len(rows)} player rows. Extracting targets...")
            
            for row in rows:
                cells = await row.query_selector_all("td")
                if len(cells) >= 9:
                    name_text = await cells[0].inner_text()
                    target_text = await cells[-1].inner_text()
                    
                    name = name_text.lower().strip()
                    try:
                        target_val = float(target_text.strip())
                        if name in name_to_id:
                            targets[name_to_id[name]] = target_val
                        elif name in full_name_to_id:
                            targets[full_name_to_id[name]] = target_val
                        else:
                            fuzzy = difflib.get_close_matches(name, name_pool, n=1, cutoff=0.8)
                            if fuzzy:
                                targets[name_to_id[fuzzy[0]]] = target_val
                    except Exception:
                        pass
                        
            await browser.close()
            print(f"Successfully mapped price targets for {len(targets)} players.")
            
    except Exception as e:
        print(f"WARNING: Price Oracle Scraping Failed ({e}). Defaulting to safe 0.0 baseline.")
        # Fallback is handled automatically because 'targets' remains empty
    
    # 3. Save to local JSON for the ML Pipeline and MILP Solver
    with open("fpl_price_targets.json", "w") as f:
        json.dump(targets, f, indent=4)
    print("Locked price targets to fpl_price_targets.json")

if __name__ == "__main__":
    asyncio.run(run_scraper())
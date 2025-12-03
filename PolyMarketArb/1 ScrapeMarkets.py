# Save as: step_1_markets.py (or PMbetValsV4.py)
# This version has a new main_async loop that pages through all results.

import aiohttp
import asyncio
import sqlite3
import os
import json
from datetime import datetime, timezone

DB_FILE = "PMactiveBets.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS markets(
            id TEXT PRIMARY KEY,
            question TEXT,
            yes_price REAL,
            no_price REAL,
            yes_token_id TEXT, 
            no_token_id TEXT,   
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def cleanup_expired_markets(active_ids):
    if not active_ids:
        return
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    q_marks = ','.join(['?']*len(active_ids))
    cur.execute(f"""
        DELETE FROM markets
                WHERE id NOT IN ({q_marks})
                """, active_ids)
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    print(f"Cleaned up {deleted} expired/delisted markets.")

async def fetch_markets_page(session, offset):
    """Fetches one page of events from the API."""
    params = {'closed': 'false', 'limit': 1000, 'offset': offset}
    url = "https://gamma-api.polymarket.com/events"
    print(f"Fetching page... offset={offset}")
    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data

async def process_markets(events):
    """Processes a list of events and extracts active binary markets."""
    now = datetime.now(timezone.utc)
    active = []
    for event in events:
        if isinstance(event, dict):
            for market in event.get("markets", []):
                end_date_str = market.get("endDate")
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                        if end_date >= now:
                            active.append(market)
                    except ValueError:
                        continue # Skip markets with bad date formats
    return active

def save_market(conn, bet_id, question, yes_price, no_price):
    """Saves a single market to the DB using an existing connection."""
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO markets (id, question, yes_price, no_price, last_updated)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET
            question = excluded.question,
            yes_price = excluded.yes_price,
            no_price = excluded.no_price,
            last_updated = CURRENT_TIMESTAMP
    """, (bet_id, question, yes_price, no_price))

async def main_async():
    init_db()
    all_markets = []
    
    conn = sqlite3.connect(DB_FILE)
    
    async with aiohttp.ClientSession() as session:
        offset = 0
        while True:
            # Fetch one page
            events_page = await fetch_markets_page(session, offset)
            
            # If the page is empty, we're done
            if not events_page:
                print("No more pages found. Stopping scrape.")
                break
                
            # Process the events to get markets
            markets_on_page = await process_markets(events_page)
            if not markets_on_page:
                # This can happen if a page has events but no valid markets
                print(f"No valid markets on page offset={offset}. Checking next page.")
            
            all_markets.extend(markets_on_page)
            
            # Move to the next page
            offset += 1000
            
            # Add a small delay to be nice to the API
            await asyncio.sleep(0.5)

    print(f"\nFetched {len(all_markets)} active markets from all pages.")

    saved_count = 0
    active_ids = []

    # Loop and save markets
    for market in all_markets:
        try:
            outcomes_raw = market.get("outcomes")
            prices_raw = market.get("outcomePrices")

            if isinstance(outcomes_raw, str):
                outcomes = json.loads(outcomes_raw)
            else:
                outcomes = outcomes_raw or []

            if isinstance(prices_raw, str):
                prices = json.loads(prices_raw)
            else:
                prices = prices_raw or []

            if (not outcomes or not prices) and "contracts" in market:
                outcomes, prices = [], []
                for c in market["contracts"]:
                    outcomes.append(c.get("outcome"))
                    prices.append(float(c.get("price", 0)))

            # This filter is what finds the binary markets
            if len(outcomes) == 2 and len(prices) == 2:
                bet_id = str(market.get("id") or market.get("_id"))
                question = market.get("question", "N/A")
                yes_price = float(prices[0] or 0)
                no_price = float(prices[1] or 0)
                
                # Pass the DB connection to save
                save_market(conn, bet_id, question, yes_price, no_price)
                saved_count += 1
                active_ids.append(bet_id)

        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            continue

    # Commit all changes at once
    conn.commit()
    conn.close()
    
    print(f"Saved {saved_count} active binary (Yes/No) markets to {DB_FILE}")

    # Clean up old markets
    cleanup_expired_markets(active_ids)

    
if __name__ == "__main__":
    print("Running Step 1: Market Sync (Full Pagination)")
    # It's good practice to delete the DB if you change the schema,
    # but since we're just adding/updating rows, it's fine to leave it.
    asyncio.run(main_async())
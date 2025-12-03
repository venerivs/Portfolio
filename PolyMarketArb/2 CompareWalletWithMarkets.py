#Compares User Wallet With All Active Bets

import sqlite3
import requests
import time
import json
from typing import Iterator, Dict, Optional, Tuple
import os

BASE = "https://data-api.polymarket.com"
DB_FILE = "PMactiveBets.db"

def ensure_user_table(user_addr:str):
    table_name = f"user_{user_addr.lower()}"
    table_name = table_name.replace("-","_")

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            market_id TEXT PRIMARY KEY,
            outcome TEXT,
            size REAL,
            avg_price REAL,
            current_price REAL
        );
    """)
    conn.commit()
    conn.close()
    return table_name

def sync_user_table(user_addr: str, active_positions:list):
    table_name = ensure_user_table(user_addr)
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(f"SELECT market_id FROM {table_name}")
    existing_ids = {row[0] for row in cur.fetchall()}
    active_ids = {p['market_id'] for p in active_positions}
    to_remove = existing_ids - active_ids
    for mid in to_remove:
        cur.execute(f"DELETE FROM {table_name} WHERE market_id = ?", (mid,))
    for pos in active_positions:
        cur.execute(f"""
            INSERT INTO {table_name} (market_id, outcome, size, avg_price, current_price)
            VALUES (?, ?, ? , ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
                outcome = excluded.outcome,
                size = excluded.size,
                avg_price = excluded.avg_price,
                current_price = excluded.current_price
            """, (
                pos["market_id"],
                pos.get("outcome"),
                pos.get("size"),
                pos.get("avgPrice"),
                pos.get("currentPrice")
            ))
        
    conn.commit()
    conn.close()

def load_markets_from_db() -> dict:
    """Load markets from local DB into a dictionary keyed by lowercase question text."""
    if not os.path.exists(DB_FILE):
        print(f"❌ Database not found: {os.path.abspath(DB_FILE)}")
        return {}

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT id, question, yes_price, no_price FROM markets")
    markets = cur.fetchall()
    conn.close()
    print(f"✅ Loaded {len(markets)} markets from DB.")
    return {question.lower(): {"id": str(mid), "yes_price": yes, "no_price": no}
            for mid, question, yes, no in markets}

def fetch_positions(user_addr: str, limit: int = 100, offset: int = 0) -> Dict:
    """Fetch active positions for a given wallet address."""
    params = {"user": user_addr, "limit": limit, "offset": offset}
    resp = requests.get(f"{BASE}/positions", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

def iter_all_positions(user_addr: str, page_size: int = 100) -> Iterator[Dict]:
    """Yield all positions for the user, handling pagination."""
    offset = 0
    while True:
        data = fetch_positions(user_addr, limit=page_size, offset=offset)
        if not data:
            break
        for pos in data:
            yield pos
        if len(data) < page_size:
            break
        offset += page_size
        time.sleep(0.1)

# -----------------------------------------------------------------
# Returns the full market data
# -----------------------------------------------------------------

def fetch_market_details_from_gamma(market_id: str) -> Optional[Dict]:
    """
    Fetches the full market data blob, which includes prices and contracts.
    """
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        print(f"Error fetching market details for {market_id}: {e}")
        return None

# -----------------------------------------------------------------
# Helper to update the main 'markets' table
# -----------------------------------------------------------------
def update_market_token_ids(market_id: str, yes_token_id: str, no_token_id: str):
    """
    Updates the main 'markets' table with the token IDs.
    """
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        UPDATE markets
        SET yes_token_id = ?, no_token_id = ?
        WHERE id = ? AND (yes_token_id IS NULL OR no_token_id IS NULL)
    """, (yes_token_id, no_token_id, market_id))
    updated_count = cur.rowcount
    conn.commit()
    conn.close()
    if updated_count > 0:
        print(f"    -> Enriched {market_id} with token IDs.")

# -----------------------------------------------------------------
# Updates DB
# -----------------------------------------------------------------

def update_user_market_prices(user_addr: str):
    """
    Updates user's table with prices AND updates the main 'markets'
    table with token IDs.
    """
    table_name = f"user_{user_addr.lower()}".replace("-", "_")

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(f"SELECT market_id, outcome FROM {table_name}")
    user_positions = cur.fetchall()
    conn.close()

    updated = 0
    for market_id, outcome in user_positions:
        # 1. Fetch all market details
        data = fetch_market_details_from_gamma(market_id)
        if not data:
            continue

        # --- Handle Price Parsing ---
        prices_str = data.get("outcomePrices")
        prices = []
        if isinstance(prices_str, str):
            try:
                prices = json.loads(prices_str)
            except json.JSONDecodeError:
                prices = []

        # --- Handles Token ID Parsing ---
        outcomes_str = data.get("outcomes")
        token_ids_str = data.get("clobTokenIds")
        yes_token_id = None
        no_token_id = None

        if isinstance(outcomes_str, str) and isinstance(token_ids_str, str):
            try:
                outcomes_list = json.loads(outcomes_str)
                token_ids_list = json.loads(token_ids_str)
                
                if isinstance(outcomes_list, list) and len(outcomes_list) >= 2 and \
                   isinstance(token_ids_list, list) and len(token_ids_list) >= 2:

                    # -----------------------------------------------------------------
                    # The API returns outcomes in [Yes, No] order,
                    # but clobTokenIds in [No_Token_ID, Yes_Token_ID] order.
                    # -----------------------------------------------------------------
                    if outcomes_list[0] == "Yes":
                        # This is the normal case
                        yes_token_id = str(token_ids_list[0]) # <--- This was wrong
                        no_token_id = str(token_ids_list[1])  # <--- This was wrong
                        
                        # Correct mapping (based on your logs):
                        yes_token_id = str(token_ids_list[1])
                        no_token_id = str(token_ids_list[0])
                        
                    elif outcomes_list[0] == "No":
                        # This is the swapped case
                        yes_token_id = str(token_ids_list[0])
                        no_token_id = str(token_ids_list[1])

            except json.JSONDecodeError:
                print(f"    -> Skipping token sync for {market_id}: Failed to parse JSON strings.")
        
        # 3. Update 'markets' table IF we found tokens
        if yes_token_id and no_token_id:
            update_market_token_ids(market_id, yes_token_id, no_token_id)
        else:
            print(f"    -> Skipping token sync for {market_id}: Could not find or map token IDs.")

        # 4. Update user's current price
        if isinstance(prices, list) and len(prices) == 2:
            new_price = None
            if outcome == "Yes" and prices[0] is not None:
                new_price = float(prices[0])
            elif outcome == "No" and prices[1] is not None:
                new_price = float(prices[1])
            
            if new_price is not None:
                conn_user = sqlite3.connect(DB_FILE)
                cur_user = conn_user.cursor()
                cur_user.execute(
                    f"UPDATE {table_name} SET current_price = ? WHERE market_id = ?",
                    (new_price, market_id)
                )
                conn_user.commit()
                conn_user.close()
                updated += 1
        
        time.sleep(0.05)
    
    print(f"🔄 Updated {updated} user market prices.")


def print_positions_with_db_prices(user_addr: str):
    """Compare wallet positions with local DB markets by question and print combined details."""
    market_map = load_markets_from_db()
    if not market_map:
        print("❌ No markets found in local database.")
        return

    positions = list(iter_all_positions(user_addr))
    if not positions:
        print("❌ No active positions found.")
        return

    total_value = sum(float(pos.get("size", 0)) * float(pos.get("avgPrice", 0)) for pos in positions if float(pos.get("avgPrice", 0)) > 0)

    print(f"\n🔹 Active Polymarket positions for {user_addr}\n")
    print(f"{'Market ID':<15} | {'Question':<70} | {'DB Yes':>6} | {'DB No':>6} | "
          f"{'User Side':>9} | {'Size':>7} | {'Entry Price':>12} | {'Current Price':>14} | Portfolio %")
    print("-" * 150)

    matches_found = 0
    active_for_db = []

    for pos in positions:
        question = (pos.get("title") or pos.get("market_title") or pos.get("name") or "").strip()
        if not question:
            continue
        q_key = question.lower()

        db_entry = market_map.get(q_key)
        
        # gives us the market_id directly.
        api_market_id = pos.get("market_id")
        
        # If question-based match fails, try finding in DB by ID
        if not db_entry:
            # Re-load map by ID instead of question
            id_map = {v['id']: v for v in market_map.values()}
            db_entry = id_map.get(api_market_id)

        if not db_entry:
            # print(f"Skipping position, no match for {api_market_id} or '{question}'")
            continue # Skip unmatched questions

        # Extract wallet position details
        user_side = pos.get("outcome") or pos.get("outcomeName") or ""
        size = float(pos.get("size", 0))
        avg_price = float(pos.get("avgPrice", 0))
        current_price = float(pos.get("currentPrice", 0))

        # DB data
        market_id = db_entry["id"]
        yes_price = db_entry["yes_price"]
        no_price = db_entry["no_price"]

        position_value = size * avg_price
        percent_of_portfolio = (position_value/total_value * 100) if total_value > 0 else 0


        if avg_price < 0.001:
            continue

        if yes_price >= 0.995 or yes_price <= 0.005 or no_price >= 0.995 or no_price <= 0.005:
            continue

        active_for_db.append({
            "market_id": market_id, 
            "outcome": user_side,
            "size": size,
            "avgPrice": avg_price,
            "currentPrice": current_price
        })

        print(f"{market_id:<15} | {question[:70]:<70} | {yes_price:>6.3f} | {no_price:>6.3f} | "
              f"{user_side:>9} | {size:>7.2f} | {avg_price:>12.3f} | "
              f"{(yes_price if user_side == 'Yes' else no_price):>14.3f} | {percent_of_portfolio:>11.2f}%")
        matches_found+=1

    if active_for_db:
        sync_user_table(user_addr, active_for_db)
        print(f"\n Synced {len(active_for_db)} active positions into table 'user_{user_addr.lower()}'.")
        
        # Back-fills token IDs
        update_user_market_prices(user_addr)

    if matches_found == 0:
        print("⚠️ No matching questions found between user positions and DB.")
    else:
        print(f"\n✅ Found {matches_found} matching active markets.")

# -----------------------------------------------------------------
# 👇 NEW FUNCTION: This is the function the main bot will call
# -----------------------------------------------------------------
def run_step2_sync(user_addr: str):
    """
    This function replaces the original 'main' function.
    It runs the sync process for a given user address.
    """
    print(f"--- Running Step 2: Syncing positions for {user_addr} ---")
    print_positions_with_db_prices(user_addr)
    print(f"--- Step 2 Complete ---")


if __name__ == "__main__":
    user_addr = "0x34e5acef2fa9e5a125a6ff690e0ec70723cfd9d0"

    # Runs file directly for testing
    print(f"--- Running Step 2 Sync (Directly) ---")
    run_step2_sync(user_addr)
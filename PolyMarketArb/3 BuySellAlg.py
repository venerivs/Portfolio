# Run Code separately with --snapshot-- for initialization
# example: "python BuySellAlg.py --snapshot"

import sqlite3
import os
import time
from datetime import datetime, timezone
import sys

# --- CLOB Client Imports ---
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

# --- Config Import ---
try:
    import config
except ImportError:
    print("❌ Error: config.py not found.")
    print("Please create it with YOUR_PRIVATE_KEY and YOUR_FUNDER_ADDRESS.")
    sys.exit(1)

# --- Step 2 Import ---
# We need the functions from Step 2 to get table names and run the sync
try:
    from PMwalletCompare2_2 import ensure_user_table, run_step2_sync
except ImportError:
    print("❌ Error: Could not import from 'PMwalletCompare2.py'")
    sys.exit(1)

# --- Global Settings & Client Initialization ---
DB_FILE = "PMactiveBets.db"
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

print("🔐 Initializing CLOB Client...")
try:
    client = ClobClient(
        host=HOST,
        key=config.YOUR_PRIVATE_KEY,
        chain_id=CHAIN_ID,
        signature_type=config.LOGIN_SIGNATURE_TYPE,
        funder=config.YOUR_FUNDER_ADDRESS
    )
    client.set_api_creds(client.create_or_derive_api_creds())
    print("✅ CLOB Client initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize CLOB client: {e}")
    print("Please check your settings in config.py.")
    sys.exit(1)


def ensure_my_bets_table(leader_addr: str) -> str:
    """
    Creates a table to store *my* copied bets for a specific leader.
    Returns the name of the table.
    """
    safe_addr = leader_addr.lower().replace("-", "_")
    table_name = f"mybets_{safe_addr}"
    
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            market_id TEXT PRIMARY KEY,
            outcome TEXT NOT NULL,
            my_size REAL NOT NULL,
            my_entry_price REAL NOT NULL,
            leader_trade_size REAL NOT NULL,
            copied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()
    return table_name

def get_my_copied_trades(my_table_name: str) -> dict:
    """
    Fetches all trades I have already copied (or snapshotted)
    from the 'mybets' table.
    Returns a dict: {market_id: {my_size, outcome, ...}}
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT * FROM {my_table_name}")
        rows = cur.fetchall()
        return {row['market_id']: dict(row) for row in rows}
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()

def get_leader_positions_and_value(leader_table_name: str) -> (dict, float):
    """
    Fetches the leader's synced positions and their total active portfolio value
    from the 'user_{addr}' table (populated by Step 2).
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    leader_positions = {}
    total_value = 0.0

    try:
        cur.execute(f"SELECT * FROM {leader_table_name}")
        rows = cur.fetchall()
        for row in rows:
            pos = dict(row)
            market_id = pos.get('market_id')
            if not market_id:
                continue

            # This handles Market Makers holding both sides
            # If already stored the 'Yes' side, skip the 'No' side.
            if market_id in leader_positions:
                print(f"  - Warning: Leader is holding both sides of {market_id}. Skipping duplicate.")
                continue
                
            leader_positions[market_id] = pos
            size = pos.get('size') or 0.0
            avg_price = pos.get('avg_price') or 0.0
            if size > 0 and avg_price > 0:
                total_value += (size * avg_price)
                
    except sqlite3.OperationalError:
        print(f"Error reading leader table {leader_table_name}.")
    
    conn.close()
    return leader_positions, total_value

def get_market_token_ids(market_id: str) -> dict:
    """
    Fetches the Yes and No token IDs for a given market ID
    from our main 'markets' table.
    """
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "SELECT yes_token_id, no_token_id FROM markets WHERE id = ?", 
        (market_id,)
    )
    row = cur.fetchone()
    conn.close()
    if row and row[0] and row[1]:
        return {"Yes": row[0], "No": row[1]}
    
    # This can happen if Step 2 hasn't enriched this market yet
    print(f"  - Warning: Token IDs for {market_id} not found in DB.")
    return None


def execute_my_trade(market_id: str, outcome: str, my_share_size: float, current_price: float) -> dict:
    """
    Executes a real trade using the CLOB client with hybrid logic.
    """
    print("\n" + "="*30)
    print("🚀 EXECUTING LIVE TRADE 🚀")
    
    # 1. Get the correct token_id to trade
    token_ids = get_market_token_ids(market_id)
    if not token_ids:
        print(f"  - ❌ FAILED: Could not find token IDs for market {market_id}")
        return {'success': False, 'error': 'Token IDs not found in DB'}
        
    token_id_to_buy = token_ids.get(outcome)
    if not token_id_to_buy:
        print(f"  - ❌ FAILED: Invalid outcome '{outcome}' for market {market_id}")
        return {'success': False, 'error': 'Invalid outcome'}

    # 2. Set trade parameters
    side_to_trade = BUY 
    size_to_trade = round(my_share_size, 2)

    # 3. Get live order book prices
    try:
        book = client.get_order_book(token_id_to_buy)
        # We must convert the price (which is a string) to a float
        best_ask = float(book.asks[0].price) if book.asks else 1.0 # Price to BUY at
    except Exception:
        print("  - Warning: Could not fetch order book. Using leader's price as fallback.")
        best_ask = current_price # This is already a float

    # 4. Implement Hybrid Logic
    LOW_PRICE_THRESHOLD = 0.08
    HIGH_PRICE_THRESHOLD = 0.92
    
    # This comparison will now work
    if best_ask <= LOW_PRICE_THRESHOLD or best_ask >= HIGH_PRICE_THRESHOLD:
        # --- STRATEGY 1: PASSIVE (Extreme Prices) ---
        print(f"  -> Extreme price ({best_ask:.3f}) detected. Placing PASSIVE limit order.")
        limit_price = best_ask # Buy at the exact best ask
    else:
        # --- STRATEGY 2: AGGRESSIVE (Mid-Range Prices) ---
        print(f"  -> Mid-range price ({best_ask:.3f}). Placing AGGRESSIVE limit order.")
        limit_price = min(best_ask + config.SLIPPAGE, 0.99)

    limit_price = round(limit_price, 2)

    # 5. Check against the Polymarket API minimum order size (5 shares)
    API_MINIMUM_SHARES = 5.0
    if size_to_trade < API_MINIMUM_SHARES:
        print(f"  - ❌ FAILED: Calculated size {size_to_trade} is less than API minimum of {API_MINIMUM_SHARES}.")
        return {
            'success': True, 
            'avg_price': 0, 
            'size': 0, 
            'order_response': 'SKIPPED - BELOW MINIMUM'
        }

    print(f"    Market ID: {market_id}")
    print(f"    Token ID:  {token_id_to_buy}")
    print(f"    Outcome:   {outcome}")
    print(f"    Side:      {side_to_trade}")
    print(f"    Size:      {size_to_trade:.2f} shares")
    print(f"    Price:     {limit_price:.2f} (Best Ask: {best_ask:.3f})")
    print(f"    Cost:      ~${(size_to_trade * limit_price):.2f}")
    
    try:
        # 6. Create and sign the order
        order_args = OrderArgs(
            price=limit_price,
            size=size_to_trade,
            side=side_to_trade,
            token_id=token_id_to_buy
        )
        signed_order = client.create_order(order_args)
        
        # 7. Post the GTC (Good-Till-Cancelled) order
        resp = client.post_order(signed_order, OrderType.GTC)
        print(f"    API Response: {resp}")

        if resp.get('success', False) and resp.get('status') in ['matched', 'live']:
            return {
                'success': True,
                'avg_price': limit_price, 
                'size': size_to_trade,
                'order_response': resp
            }
        else:
            api_error = resp.get('errorMsg', 'API did not confirm success')
            print(f"  - ❌ FAILED TO EXECUTE: API reported failure: {api_error}")
            return {'success': False, 'error': api_error}

    except Exception as e:
        print(f"  - ❌ FAILED TO EXECUTE: {e}")
        return {'success': False, 'error': str(e)}
    finally:
        print("="*30 + "\n")

# In PMbuysell.py (Step 3)

def save_my_copied_trade(my_table_name: str, market_id: str, outcome: str, my_size: float, my_entry_price: float, leader_size: float):
    """Saves the record of a trade *I* just made into my 'mybets' table."""
    
    # If we skipped the trade (size 0), save it as a 0-size "snapshot"
    if my_size == 0:
        print(f"✅ Skipping {market_id} (below min size) and saving as 'seen'.")
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute(f"""
            INSERT INTO {my_table_name} (market_id, outcome, my_size, my_entry_price, leader_trade_size)
            VALUES (?, ?, 0, 0, 0)
            ON CONFLICT(market_id) DO NOTHING
        """, (market_id, outcome))
        conn.commit()
        conn.close()
        return

    # Otherwise, save the real trade
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(f"""
        INSERT INTO {my_table_name} (market_id, outcome, my_size, my_entry_price, leader_trade_size, copied_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(market_id) DO UPDATE SET
            outcome = excluded.outcome,
            my_size = excluded.my_size,
            my_entry_price = excluded.my_entry_price,
            leader_trade_size = excluded.leader_trade_size,
            copied_at = excluded.copied_at
    """, (market_id, outcome, my_size, my_entry_price, leader_size, datetime.now(timezone.utc)))
    conn.commit()
    conn.close()
    print(f"✅ Saved my trade for {market_id} to {my_table_name}.")

def take_initial_snapshot(leader_addr: str):
    """
    One-time function to 'mark' all of the leader's current positions
    as 'seen' so they are not copied.
    """
    print(f"--- Taking snapshot of {leader_addr}'s current positions ---")
    
    leader_table_name = ensure_user_table(leader_addr)
    my_table_name = ensure_my_bets_table(leader_addr)
    leader_positions, _ = get_leader_positions_and_value(leader_table_name)
    
    if not leader_positions:
        print("❌ Leader positions table is empty. Please run Step 2 first.")
        return

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    
    snapshot_count = 0
    for market_id, pos_data in leader_positions.items():
        # Insert a 0-size "placeholder" bet into mybets table
        cur.execute(f"""
            INSERT INTO {my_table_name} (market_id, outcome, my_size, my_entry_price, leader_trade_size)
            VALUES (?, ?, 0, 0, 0)
            ON CONFLICT(market_id) DO NOTHING
        """, (
            market_id,
            pos_data.get('outcome', 'N/A')
        ))
        if cur.rowcount > 0:
            snapshot_count += 1
            
    conn.commit()
    conn.close()
    
    print(f"✅ Snapshot complete. Ignored {snapshot_count} new positions.")
    print(f"  (Total positions seen: {len(leader_positions)})")

def check_for_new_trades_and_copy(leader_addr: str, my_total_portfolio_value: float):
    """
    Main logic for Step 3.
    Compares leader's current positions with my copied positions and executes new trades.
    """
    print(f"--- Running Step 3: Checking for new trades to copy from {leader_addr} ---")
    
    leader_table_name = ensure_user_table(leader_addr)
    my_table_name = ensure_my_bets_table(leader_addr)
    
    # 1. Get trades I've already copied (this includes 0-size snapshots)
    my_copied_trades = get_my_copied_trades(my_table_name)
    print(f"Found {len(my_copied_trades)} trades I have 'seen' (copied or snapshotted).")
    
    # 2. Get leader's current positions and total active value
    leader_positions, leader_total_value = get_leader_positions_and_value(leader_table_name)
    
    if not leader_positions:
        print("Leader's position table is empty. Did Step 2 run successfully? Exiting Step 3.")
        return
    if leader_total_value <= 0:
        print("Leader's total active portfolio value is $0. Cannot calculate ratios. Exiting.")
        return

    print(f"Leader's total active value (sum of positions): ${leader_total_value:,.2f}")
    print(f"My total portfolio value: ${my_total_portfolio_value:,.2f}")
    
    new_trades_found = 0
    
    # 3. Loop through all leader positions and find new ones
    for market_id, pos in leader_positions.items():
        
        # 4. (CORE LOGIC) Check if this is a new trade
        if market_id not in my_copied_trades:
            new_trades_found += 1
            print(f"\n🔎 Found new position to copy: Market {market_id}")
            
            leader_trade_size = pos.get('size', 0.0)
            leader_avg_price = pos.get('avg_price', 0.0)
            leader_trade_value = leader_trade_size * leader_avg_price
            
            current_price = pos.get('current_price') 
            outcome_to_buy = pos.get('outcome')

            # --- Sanity Checks ---
            if leader_trade_value <= 0:
                print(f"  - Skipping: Leader's trade value is ${leader_trade_value:.2f}.")
                continue
            if not current_price or current_price <= 0.01 or current_price >= 0.99:
                print(f"  - Skipping: Market price ${current_price} is invalid or market is near resolution.")
                continue
            if not outcome_to_buy:
                print("  - Skipping: Leader position has no 'outcome'.")
                continue

            # 5. Calculate proportional trade size
            leader_percent_allocation = leader_trade_value / leader_total_value
            my_trade_value_to_allocate = my_total_portfolio_value * leader_percent_allocation
            my_share_size_to_buy = my_trade_value_to_allocate / current_price
            
            print(f"  - Leader Allocation: {leader_percent_allocation:.2%} (${leader_trade_value:,.2f})")
            print(f"  - My Allocation:     {leader_percent_allocation:.2%} (${my_trade_value_to_allocate:,.2f})")

            # 6. Execute the trade (LIVE)
            trade_receipt = execute_my_trade(market_id, outcome_to_buy, my_share_size_to_buy, current_price)
            
            # 7. Save trade to 'mybets' DB
            if trade_receipt and trade_receipt.get('success'):
                filled_price = trade_receipt.get('avg_price', current_price)
                filled_size = trade_receipt.get('size', my_share_size_to_buy)
                
                save_my_copied_trade(
                    my_table_name=my_table_name,
                    market_id=market_id,
                    outcome=outcome_to_buy,
                    my_size=filled_size,
                    my_entry_price=filled_price,
                    leader_size=leader_trade_size
                )
            else:
                print(f"  - ❌ FAILED to execute or save trade for {market_id}.")
                print(f"  - Error: {trade_receipt.get('error')}")

    if new_trades_found == 0:
        print("\n✅ No new positions found to copy. My portfolio is in sync.")
    
    print(f"--- Step 3 Complete ---")


if __name__ == "__main__":
    if not os.path.exists(DB_FILE):
        print(f"Error: Database file not found at {DB_FILE}")
        print("Please run Step 1 (markets) and Step 2 (user sync) first.")
        sys.exit(1)
        
    # --- CONFIGURATION ---
    LEADER_ADDR = "0x34e5acef2fa9e5a125a6ff690e0ec70723cfd9d0" # Example good leader
    
    print("Fetching your wallet balance...")
    try:
        # Returns the balance as a large integer (e.g., 100000000 for $100.00)
        usdc_balance_wei = client.get_collateral_balance(config.YOUR_FUNDER_ADDRESS)
        
        # Convert from 6-decimal integer to a float
        MY_TOTAL_PORTFOLIO_VALUE = usdc_balance_wei / 1_000_000.0
        
        print(f"✅ Your wallet balance: ${MY_TOTAL_PORTFOLIO_VALUE:,.2f} USDC")
    except Exception as e:
        print(f"❌ Error fetching wallet balance: {e}")
        print("Defaulting to $100.00. Please check your config and balance.")
        MY_TOTAL_PORTFOLIO_VALUE = 100.0

    # This block handles the one-time snapshot setup
    if len(sys.argv) > 1 and sys.argv[1] == '--snapshot':
        print(f"--- Running ONE-TIME SNAPSHOT for {LEADER_ADDR} ---")
        print("This will mark all current positions as 'seen' without copying.")
        
        # We must run Step 2 first to have data to snapshot
        try:
            print("Running Step 2 sync first...")
            run_step2_sync(LEADER_ADDR)
            print("Step 2 sync complete. Now taking snapshot...")
            take_initial_snapshot(LEADER_ADDR)
            print("--- Snapshot complete. You can now start the main bot. ---")
        except Exception as e:
            print(f"Failed to run snapshot: {e}")
            sys.exit(1)
    
    # This is the normal operation for your main bot loop
    else:
        print("--- Running Step 3: Check for New Trades (Live Mode) ---")
        # The script now passes the dynamically fetched balance
        check_for_new_trades_and_copy(LEADER_ADDR, MY_TOTAL_PORTFOLIO_VALUE)
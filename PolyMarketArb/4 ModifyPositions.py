# Sync my current positions with the user being copied.

import sqlite3
import os
import time
from datetime import datetime, timezone
import sys

# --- CLOB Client Imports ---
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL # Now importing SELL

# --- Config Import ---
try:
    import config
except ImportError:
    print("❌ Error: config.py not found.")
    sys.exit(1)

# --- Step 2/3 Imports ---
try:
    # Need these functions to get the correct table names
    from PMwalletCompare2_2 import ensure_user_table
    from PMbuysell_3 import ensure_my_bets_table, get_market_token_ids, get_leader_positions_and_value
except ImportError:
    print("❌ Error: Could not import from 'step_2_user_sync.py' or 'step_3_copy_trades.py'")
    sys.exit(1)

# --- Global Settings & Client Initialization ---
DB_FILE = "PMactiveBets.db"
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

# Initialize client (same as Step 3)
try:
    client = ClobClient(
        host=HOST,
        key=config.YOUR_PRIVATE_KEY,
        chain_id=CHAIN_ID,
        signature_type=config.LOGIN_SIGNATURE_TYPE,
        funder=config.YOUR_FUNDER_ADDRESS
    )
    client.set_api_creds(client.create_or_derive_api_creds())
except Exception as e:
    print(f"❌ Failed to initialize CLOB client: {e}")
    sys.exit(1)


def get_my_active_positions(my_table_name: str) -> dict:
    """
    Fetches all *active* trades I have copied (size > 0).
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # Only select trades where we actually hold shares
        cur.execute(f"SELECT * FROM {my_table_name} WHERE my_size > 0")
        rows = cur.fetchall()
        return {row['market_id']: dict(row) for row in rows}
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()

# In PMsyncpositions_4.py
# REPLACE this entire function

def execute_position_adjustment(market_id: str, outcome: str, size: float, action: str) -> dict:
    """
    Executes a BUY or SELL order to adjust a position using hybrid logic.
    """
    print("\n" + "="*30)
    print(f"🚀 EXECUTING {action} ADJUSTMENT 🚀")

    # 1. Get the correct token_id to trade
    token_ids = get_market_token_ids(market_id)
    if not token_ids:
        return {'success': False, 'error': 'Token IDs not found in DB'}
        
    token_id_to_trade = token_ids.get(outcome)
    if not token_id_to_trade:
        return {'success': False, 'error': 'Invalid outcome'}

    # 2. Get live order book prices
    try:
        book = client.get_order_book(token_id_to_trade)
        # We must convert the prices (which are strings) to floats
        best_ask = float(book.asks[0].price) if book.asks else 1.0 # Price to BUY
        best_bid = float(book.bids[0].price) if book.bids else 0.0 # Price to SELL
    except Exception:
        print("  - Warning: Could not fetch order book. Using fallback prices.")
        best_ask = 0.99
        best_bid = 0.01

    # 3. Define price boundaries and action
    side_to_trade = BUY if action == "BUY" else SELL
    LOW_PRICE_THRESHOLD = 0.08
    HIGH_PRICE_THRESHOLD = 0.92
    
    # 4. Implement Hybrid Logic
    if side_to_trade == BUY:
        current_market_price = best_ask
        # This comparison will now work
        if current_market_price <= LOW_PRICE_THRESHOLD or current_market_price >= HIGH_PRICE_THRESHOLD:
            # --- STRATEGY 1: PASSIVE (Extreme Prices) ---
            print(f"  -> Extreme price ({current_market_price:.3f}) detected. Placing PASSIVE limit order.")
            limit_price = current_market_price # Buy at the exact best ask
        else:
            # --- STRATEGY 2: AGGRESSIVE (Mid-Range Prices) ---
            print(f"  -> Mid-range price ({current_market_price:.3f}). Placing AGGRESSIVE limit order.")
            limit_price = min(current_market_price + config.SLIPPAGE, 0.99)
    
    else: # SELL
        current_market_price = best_bid
        # This comparison will now work
        if current_market_price <= LOW_PRICE_THRESHOLD or current_market_price >= HIGH_PRICE_THRESHOLD:
            # --- STRATEGY 1: PASSIVE (Extreme Prices) ---
            print(f"  -> Extreme price ({current_market_price:.3f}) detected. Placing PASSIVE limit order.")
            limit_price = current_market_price # Sell at the exact best bid
        else:
            # --- STRATEGY 2: AGGRESSIVE (Mid-Range Prices) ---
            print(f"  -> Mid-range price ({current_market_price:.3f}). Placing AGGRESSIVE limit order.")
            limit_price = max(current_market_price - config.SLIPPAGE, 0.01)

    limit_price = round(limit_price, 2)
    size_to_trade = round(size, 2)

    # 5. Check minimum size
    API_MINIMUM_SHARES = 5.0
    if size_to_trade < API_MINIMUM_SHARES:
        print(f"  - ❌ SKIPPING: Calculated size {size_to_trade} is less than API minimum of {API_MINIMUM_SHARES}.")
        return {'success': True, 'filled_size': 0, 'filled_price': 0, 'error': 'SKIPPED - BELOW MINIMUM'}

    print(f"    Market ID: {market_id}")
    print(f"    Token ID:  {token_id_to_trade}")
    print(f"    Action:    {action}")
    print(f"    Size:      {size_to_trade:.2f} shares")
    print(f"    Price:     {limit_price:.2f} (Market: {current_market_price:.3f})")

    # 6. Create, sign, and post the order
    try:
        order_args = OrderArgs(
            price=limit_price,
            size=size_to_trade,
            side=side_to_trade,
            token_id=token_id_to_trade
        )
        signed_order = client.create_order(order_args)
        resp = client.post_order(signed_order, OrderType.GTC)
        print(f"    API Response: {resp}")

        if resp.get('success', False) and resp.get('status') in ['matched', 'live']:
            return {'success': True, 'filled_size': size_to_trade, 'filled_price': limit_price}
        else:
            api_error = resp.get('errorMsg', 'API did not confirm success')
            print(f"  - ❌ FAILED TO EXECUTE: API reported failure: {api_error}")
            return {'success': False, 'error': api_error}

    except Exception as e:
        print(f"  - ❌ FAILED TO EXECUTE: {e}")
        return {'success': False, 'error': str(e)}
    finally:
        print("="*30 + "\n")

def update_my_position_in_db(my_table_name: str, market_id: str, new_my_size: float, new_leader_size: float):
    """Updates the 'mybets' table with new sizes."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(f"""
        UPDATE {my_table_name}
        SET my_size = ?, leader_trade_size = ?
        WHERE market_id = ?
    """, (new_my_size, new_leader_size, market_id))
    conn.commit()
    conn.close()

def remove_my_position_from_db(my_table_name: str, market_id: str):
    """Deletes a closed position from the 'mybets' table."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {my_table_name} WHERE market_id = ?", (market_id,))
    conn.commit()
    conn.close()

def check_and_sync_positions(leader_addr: str):
    """
    Main logic for Step 4.
    Compares my active positions to the leader's and syncs.
    """
    print(f"--- Running Step 4: Syncing existing positions with {leader_addr} ---")
    
    leader_table_name = ensure_user_table(leader_addr)
    my_table_name = ensure_my_bets_table(leader_addr)

    # 1. Get my *active* positions (where my_size > 0)
    my_positions = get_my_active_positions(my_table_name)
    if not my_positions:
        print("✅ No active positions to sync.")
        print(f"--- Step 4 Complete ---")
        return

    # 2. Get leader's *current* positions
    leader_positions, _ = get_leader_positions_and_value(leader_table_name)
    
    print(f"Checking {len(my_positions)} active copied positions...")
    
    for market_id, my_pos in my_positions.items():
        my_current_size = my_pos['my_size']
        leader_original_size = my_pos['leader_trade_size'] # Size when we copied
        
        # 3. Check if leader still holds this position
        leader_current_pos = leader_positions.get(market_id)
        
        if not leader_current_pos:
            # --- CASE 1: Leader has fully CLOSED their position ---
            print(f"🔎 Leader has closed position {market_id}. Selling my {my_current_size} shares.")
            
            receipt = execute_position_adjustment(
                market_id=market_id,
                outcome=my_pos['outcome'],
                size=my_current_size,
                action="SELL"
            )
            
            if receipt.get('success'):
                # Delete from DB so we don't try to sell again
                remove_my_position_from_db(my_table_name, market_id)
                print(f"✅ Sold and removed {market_id} from my bets.")
            else:
                print(f"❌ FAILED to sell {market_id}. Will retry next cycle.")
            
        else:
            # --- CASE 2: Leader has CHANGED their position size ---
            leader_new_size = leader_current_pos.get('size', 0.0)
            
            if leader_new_size == leader_original_size:
                # No change, do nothing
                continue
            
            if leader_new_size == 0 or leader_original_size == 0:
                print(f"Skipping sync for {market_id}: division by zero error.")
                continue

            # Calculate the new proportional size I should have
            size_change_ratio = leader_new_size / leader_original_size
            my_new_target_size = my_current_size * size_change_ratio
            
            size_diff = my_new_target_size - my_current_size
            
            if size_diff > 0.01:
                # --- Leader ADDED to position ---
                print(f"🔎 Leader added to {market_id}. Buying {size_diff:.2f} more shares.")
                
                receipt = execute_position_adjustment(
                    market_id=market_id,
                    outcome=my_pos['outcome'],
                    size=size_diff,
                    action="BUY"
                )
                
                if receipt.get('success'):
                    new_total_size = my_current_size + receipt.get('filled_size', size_diff)
                    update_my_position_in_db(my_table_name, market_id, new_total_size, leader_new_size)
                    print(f"✅ Bought more {market_id}. New size: {new_total_size:.2f}")

            elif size_diff < -0.01:
                # --- Leader REDUCED position ---
                shares_to_sell = abs(size_diff)
                print(f"🔎 Leader reduced {market_id}. Selling {shares_to_sell:.2f} shares.")

                receipt = execute_position_adjustment(
                    market_id=market_id,
                    outcome=my_pos['outcome'],
                    size=shares_to_sell,
                    action="SELL"
                )
                
                if receipt.get('success'):
                    new_total_size = my_current_size - receipt.get('filled_size', shares_to_sell)
                    update_my_position_in_db(my_table_name, market_id, new_total_size, leader_new_size)
                    print(f"✅ Sold some {market_id}. New size: {new_total_size:.2f}")
    
    print(f"--- Step 4 Complete ---")


if __name__ == "__main__":
    if not os.path.exists(DB_FILE):
        print(f"Error: Database file not found at {DB_FILE}")
    else:
        # --- CONFIGURATION ---
        LEADER_ADDR = "0x34e5acef2fa9e5a125a6ff690e0ec70723cfd9d0"
        
        print("--- Running Step 4: Sync Positions (Directly) ---")
        check_and_sync_positions(LEADER_ADDR)
# Save this file as: PMtraderbot1.py

import asyncio
import time
from datetime import datetime
import sys
import os
from decimal import Decimal 

DB_FILE = "PMactiveBets.db"

# --- CLOB Client Imports ---
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import AssetType, BalanceAllowanceParams 

try:
    import config
except ImportError:
    print("❌ Error: config.py not found.")
    sys.exit(1)


# --- Import functions from your other files ---
try:
    from PMbetValsV4_1 import main_async as run_step1_markets
    print("✅ Loaded Step 1: Market Sync")
except ImportError:
    print("❌ Error: Could not import from 'PMbetValsV4_1.py'")
    sys.exit(1)

try:
    from PMwalletCompare2_2 import run_step2_sync
    print("✅ Loaded Step 2: User Sync")
except ImportError:
    print("❌ Error: Could not import from 'PMwalletCompare2_2.py'")
    sys.exit(1)
    
try:
    from PMbuysell_3 import check_for_new_trades_and_copy
    print("✅ Loaded Step 3: Copy New Trades")
except ImportError:
    print("❌ Error: Could not import from 'PMbuysell_3.py'")
    sys.exit(1)

try:
    from PMsyncpositions_4 import check_and_sync_positions
    print("✅ Loaded Step 4: Sync Existing Positions")
except ImportError:
    print("❌ Error: Could not import from 'PMsyncpositions_4.py'")
    sys.exit(1)

# --- CONFIGURATION ---
LEADER_ADDR = "0x34e5acef2fa9e5a125a6ff690e0ec70723cfd9d0" 
LOOP_DELAY_SECONDS = 10
# --- END CONFIGURATION ---


async def main_loop():
    print(f"\n🚀 Starting Polymarket Copy Trader Bot 🚀")
    print(f"    Watching: {LEADER_ADDR}")
    print(f"    Loop Delay: {LOOP_DELAY_SECONDS}s")
    
    # --- Initialize Client Once ---
    print("🔐 Initializing CLOB Client for balance checks...")
    try:
        balance_client = ClobClient(
            host="https://clob.polymarket.com",
            key=config.YOUR_PRIVATE_KEY,
            chain_id=137,
            signature_type=config.LOGIN_SIGNATURE_TYPE,
            funder=config.YOUR_FUNDER_ADDRESS
        )
        

        balance_client.set_api_creds(balance_client.create_or_derive_api_creds())
        
        print("✅ CLOB Client initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize CLOB client: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
 
    
    while True:
        print("\n" + "="*50)
        print(f"Starting new cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        
        try:
            # --- STEP 0: Fetch current wallet balance ---
            print("\n[STEP 0] Fetching current wallet balance...")
            try:
                # This function is SYNCHRONOUS
                balance_data = balance_client.get_balance_allowance(
                    params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                )
                
                # Access the balance using dictionary key ['balance']
                my_current_balance = float(balance_data['balance']) / 1_000_000.0
                
                print(f"✅ Current balance: ${my_current_balance:,.2f} USDC")
            except Exception as e:
                print(f"❌ Error fetching wallet balance: {e}")
                print("Defaulting to $50.00 for this cycle to be safe.")
                my_current_balance = 50.0 
            # -----------------------------

            # --- STEP 1: Update all market data ---
            print("\n[STEP 1] Syncing all active markets...")
            await run_step1_markets() # This is ASYNC
            print("[STEP 1] Market sync complete.")
            
            # --- STEP 2: Sync leader's current positions & get token IDs ---
            print("\n[STEP 2] Syncing leader's positions...")
            run_step2_sync(LEADER_ADDR) # This is SYNC
            print("[STEP 2] Leader sync complete.")
            
            # --- STEP 3: Check for new trades and copy ---
            print("\n[STEP 3] Checking for NEW trades to copy...")
            check_for_new_trades_and_copy(LEADER_ADDR, my_current_balance) # This is SYNC
            print("[STEP 3] New trade check complete.")
            
            # --- STEP 4: Check for sales/adjustments ---
            print("\n[STEP 4] Syncing EXISTING position adjustments...")
            check_and_sync_positions(LEADER_ADDR) # This is SYNC
            print("[STEP 4] Position sync complete.")

        except Exception as e:
            print(f"❌❌❌ AN ERROR OCCURRED IN THE MAIN LOOP: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*50)
        print(f"Cycle complete. Sleeping for {LOOP_DELAY_SECONDS} seconds...")
        await asyncio.sleep(LOOP_DELAY_SECONDS) # This is ASYNC

if __name__ == "__main__":
    if not os.path.exists(DB_FILE):
        print("❌ Error: Database file 'PMactiveBets.db' not found.")
        print("Please run the 3-step setup first:")
        print(f"1. py PMbetValsV4_1.py")
        print(f"2. py PMwalletCompare2_2.py")
        print(f"3. py PMbuysell_3.py --snapshot")
        sys.exit(1)
        
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nBot stopped manually.")
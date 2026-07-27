import os
import sys
import requests

FPL_EMAIL = os.environ.get("FPL_EMAIL")
FPL_PASSWORD = os.environ.get("FPL_PASSWORD")
FPL_TEAM_ID = os.environ.get("FPL_TEAM_ID")

if not all([FPL_EMAIL, FPL_PASSWORD, FPL_TEAM_ID]):
    print("CRITICAL ERROR: Missing FPL_EMAIL, FPL_PASSWORD, or FPL_TEAM_ID secrets.")
    sys.exit(1)

def test_fpl_authentication():
    session = requests.Session()
    
    # 1. Correct Active Authentication Endpoint
    login_url = "https://users.auth.premierleague.com/accounts/login/"
    
    # Standard headers to mimic browser authentication
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://fantasy.premierleague.com/"
    })
    
    payload = {
        "login": FPL_EMAIL,
        "password": FPL_PASSWORD,
        "redirect_uri": "https://fantasy.premierleague.com/a/login",
        "app": "plfpl-web"
    }
    
    print("Attempting programmatic authentication with Premier League servers...")
    
    try:
        login_response = session.post(login_url, data=payload)
        
        # Check for active pl_profile session token
        cookies = session.cookies.get_dict()
        if 'pl_profile' not in cookies:
            print(f"AUTH FAILED: Status {login_response.status_code}. 'pl_profile' cookie missing.")
            sys.exit(1)
            
        print("AUTH SUCCESS: Active 'pl_profile' session cookie acquired.")
        
        # 2. Access Private My-Team Endpoint
        my_team_url = f"https://fantasy.premierleague.com/api/my-team/{FPL_TEAM_ID}/"
        team_response = session.get(my_team_url)
        
        if team_response.status_code != 200:
            print(f"API BLOCKED: Received Status Code {team_response.status_code}")
            sys.exit(1)
            
        team_data = team_response.json()
        picks = team_data.get("picks", [])
        
        if not picks:
            print("ERROR: Authentication succeeded, but no squad picks were returned.")
            sys.exit(1)
            
        print("\n==================================================")
        print("         HIDDEN SQUAD ECONOMICS EXTRACTED         ")
        print("==================================================")
        print(f"{'Player ID':<10} | {'Purchase Price':<15} | {'Selling Price':<15}")
        print("--------------------------------------------------")
        
        for pick in picks:
            player_id = pick["element"]
            purchase = pick["purchase_price"] / 10.0
            selling = pick["selling_price"] / 10.0
            print(f"{player_id:<10} | £{purchase:<14.1f}m | £{selling:<14.1f}m")
            
        print("--------------------------------------------------")
        bank_balance = team_data.get('transfers', {}).get('bank', 0) / 10.0
        print(f"Current Bank Balance: £{bank_balance}m")
        print("==================================================")
        print("SUCCESS: Authenticated session operates flawlessly on GitHub Actions.")
        
    except Exception as e:
        print(f"CRITICAL EXCEPTION: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_fpl_authentication()

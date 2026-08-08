import requests

def inspect_fpl_api_for_price_predictor():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    headers = {"User-Agent": "FPL-API-Inspector/1.0"}
    
    print("Fetching FPL Bootstrap API...")
    resp = requests.get(url, headers=headers)
    
    if resp.status_code == 200:
        data = resp.json()
        elements = data.get("elements", [])
        
        if elements:
            player = elements[0]
            print(f"\n--- Scanning Keys for {player['first_name']} {player['second_name']} ---\n")
            
            # Look for obvious new keywords
            keywords = ["price", "change", "progress", "predict", "likely", "trend"]
            
            for key, value in player.items():
                if any(kw in key.lower() for kw in keywords):
                    print(f"🎯 POTENTIAL MATCH -> '{key}': {value}")
                else:
                    # Print everything else just in case it's named obscurely
                    print(f"'{key}': {value}")
    else:
        print(f"Failed to connect. HTTP {resp.status_code}")

if __name__ == "__main__":
    inspect_fpl_api_for_price_predictor()
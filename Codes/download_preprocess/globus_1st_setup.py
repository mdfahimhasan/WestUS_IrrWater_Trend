# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu


'''
This script is for the first-time setup of Globus authentication. Not needed for MAC as Globus can be automatically configured from the download.py script.

However, particularly for linux servers, where jobs have to submitted, the authentication process needs to be done separately.

Run this script only once for the first-time setup of Globus authenticatiom. Follow the steps outlined below. 

Steps:
------------------------------------------
> Connect to the linux server using ssh >

> Browse (cd) to the directory where this .py file is located

> Run this script without submitting a job 

> copy paste the generated link to MAC or windows internet browser

> copy the code from the browser and paste it on linux terminal


**** After the setup is complete, a 'json' file will be created in the same directory where this .py file is located. 
This file contains the authentication tokens and other information. 
The download.py script will use this file to authenticate and download data from Globus.
'''


import sys
import globus_sdk
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Codes.download_preprocess.download import globus_save_tokens, globus_on_refresh


CLIENT_ID      = 'e344b16b-0e18-456b-bf32-449ab168aa35'

 # ── First-time only ──────────────────────────────────────────────
print("[auth] First-time setup — completing OAuth flow...")

# creates a Globus app identity using the Client ID
auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)   

auth_client.oauth2_start_flow(
    requested_scopes="urn:globus:auth:scope:transfer.api.globus.org:all",
    refresh_tokens=True,             # <-- key: request long-lived refresh token
)

url = auth_client.oauth2_get_authorize_url()
print(f"\nOpen this URL in your browser:\n{url}\n")
auth_code = input("Paste the auth code: ").strip()

token_response = auth_client.oauth2_exchange_code_for_tokens(auth_code)
t = token_response.by_resource_server["transfer.api.globus.org"]

globus_save_tokens({
    "access_token"     : t["access_token"],
    "refresh_token"    : t["refresh_token"],
    "expires_at_seconds": t["expires_at_seconds"],
})

authorizer = globus_sdk.RefreshTokenAuthorizer(
    t["refresh_token"],
    auth_client,
    access_token=t["access_token"],
    expires_at=t["expires_at_seconds"],
    on_refresh=globus_on_refresh,
)

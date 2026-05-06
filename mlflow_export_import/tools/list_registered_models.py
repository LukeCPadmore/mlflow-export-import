""" 
Lists all registered models.
"""

import json
from mlflow_export_import.client.client_utils import create_http_client, create_mlflow_client

def main():
    client = create_http_client(create_mlflow_client())
    print("HTTP client:",client)
    rsp = client._get("registered-models/search")
    dct = json.loads(rsp.text)
    print(json.dumps(dct,indent=2)+"\n")

if __name__ == "__main__":
    main()

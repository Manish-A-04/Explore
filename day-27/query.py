import requests

API_URL = "http://localhost:8000/rag/invoke"

payload = {"input": {"query": "What are the projects done by OpenAI in the last 5 years?"}}

response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    result = response.json()
    print("Answer:", result.get("output", "No response"))
    print("Source Documents:", result.get("source_documents", []))
else:
    print(f"Error {response.status_code}: {response.text}")

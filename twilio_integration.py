from twilio.rest import Client
import requests
import time

# === Twilio credentials ===
account_sid = "AC539a12aa0037c7ec3cf00e9c52a97bec"
auth_token = "ec7c39a3a43abeec8652d3f9ab99e75a"
client = Client(account_sid, auth_token)

# === Numbers ===
to_number = "+918208272663"     # your verified number
from_number = "+17819922715"    # your Twilio number

# === Flask backend (your running app) ===
backend_url = "http://127.0.0.1:5001/analyze_recording"

# === 1️⃣ Make a call and record it ===
print("📞 Initiating call...")
call = client.calls.create(
    to=to_number,
    from_=from_number,
    record=True,
    recording_channels="mono",
    recording_status_callback=backend_url,  # Send recording URL to backend
    recording_status_callback_method="POST",
    url="https://handler.twilio.com/twiml/EHcb11fa0470bb215e2fbd1b1922cc0dd9"  # simple TwiML that says “start speaking”
)

print(f"✅ Call started! SID: {call.sid}")
print("🕒 Wait for Twilio to call you and record your voice.")

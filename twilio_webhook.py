from flask import Flask, request, Response

app = Flask(__name__)

@app.route("/twiml", methods=["POST"])
def twiml():
    twiml_response = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://morainal-kelly-bedazzlingly.ngrok-free.dev/media-stream" />
    </Connect>
</Response>"""
    return Response(twiml_response, mimetype="text/xml")

if __name__ == "__main__":
    app.run(port=5000, debug=True)

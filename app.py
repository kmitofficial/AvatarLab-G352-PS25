from flask import Flask, request, jsonify, send_file
import os
import uuid
import torch
import torchaudio

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

app = Flask(__name__)
OUTPUT_FOLDER = 'outputs'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

@app.route("/")
def index():
    return "Zonos TTS API is up and running 🚀"

@app.route('/api/generate_speech', methods=['POST'])
def generate_speech():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    if 'speaker_audio_path' not in data:
        return jsonify({'error': 'No speaker audio path provided'}), 400

    text = data['text']
    speaker_audio_path = data['speaker_audio_path']
    language = data.get('language', 'en-us')

    try:
        wav, sampling_rate = torchaudio.load(speaker_audio_path)
        speaker = model.make_speaker_embedding(wav, sampling_rate)

        cond_dict = make_cond_dict(text=text, speaker=speaker, language=language)
        conditioning = model.prepare_conditioning(cond_dict)

        codes = model.generate(conditioning)
        wavs = model.autoencoder.decode(codes).cpu()

        output_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4()}.wav")
        torchaudio.save(output_path, wavs[0], model.autoencoder.sampling_rate)

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

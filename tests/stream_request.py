""" test VAD service """
# python3 -m grpc_tools.protoc -I . --python_out=./tests --pyi_out=./tests --grpc_python_out=./tests proto/vad.proto

import logging
import numpy as np
import soundfile as sf
import pyaudio
from tempfile import NamedTemporaryFile
import grpc

from proto import vad_pb2, vad_pb2_grpc

HOST = "localhost"
PORT = 60053

CHUNK_DURATION_MS = 100
SAMPLE_RATE = 16000
FRAMES_PER_BUFFER=int(SAMPLE_RATE/1000 * CHUNK_DURATION_MS)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s | %(asctime)s | %(message)s",
        level=logging.INFO
    )

    with grpc.insecure_channel(f"{HOST}:{PORT}") as channel:
        stub = vad_pb2_grpc.VoiceActivityDetectorStub(channel)
        
        audio = pyaudio.PyAudio()
        
        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER
        )
        
        logging.info("Started Recording...")
        
        while True:
            audio_chunk = stream.read(FRAMES_PER_BUFFER)
            
            data = np.frombuffer(audio_chunk, np.float32)
            with NamedTemporaryFile(suffix=".wav") as f:
                sf.write(
                    file=f,
                    data=data,
                    samplerate=SAMPLE_RATE, 
                    format='WAV', 
                    subtype='PCM_16'
                )
                f.seek(0)
                audio_data = f.read()
                
            request = vad_pb2.VoiceActivityDetectorRequest(audio_data=audio_data)
            response = stub.detect_voice_activity(request)
            
            logging.info("Voice Detected Confidence: %0.2f", response.confidence)

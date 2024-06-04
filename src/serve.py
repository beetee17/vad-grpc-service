""" MT Service """
import os
import logging
from time import perf_counter
from concurrent import futures

from tempfile import NamedTemporaryFile
import librosa
import torch
import numpy as np

import grpc

from proto import vad_pb2, vad_pb2_grpc

MODEL_DIR = os.environ["MODEL_DIR"]
SAMPLE_RATE = 16000

class VoiceActivityDetectorService(vad_pb2_grpc.VoiceActivityDetectorServicer):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Running on device: %s", self.device)

        logging.info("Loading model...")
        
        model, utils = torch.hub.load(
            repo_or_dir=MODEL_DIR, 
            source='local', 
            model='silero_vad'
        )
        
        self.model = model.to(self.device)

    def detect_voice_activity(self, request, context):
        infer_start = perf_counter()
        
        with NamedTemporaryFile(suffix=".wav") as f:
            f.write(request.audio_data)
            f.seek(0)
            audio_float32, _ = librosa.load(
                path=f.name,
                sr=SAMPLE_RATE,
                dtype=np.float32
            )
            
            tensor = torch.from_numpy(audio_float32).to(self.device)
            
            confidence = self.model(tensor, SAMPLE_RATE)
            
            infer_end = perf_counter()
            
            logging.info("Voice Detected Confidence: %0.2f", confidence)
            
            logging.info("Inference elapsed time: %s", infer_end - infer_start)
            
            return vad_pb2.VoiceActivityDetectorResponse(confidence=confidence.float())

def serve():
    """Serves VAD Model"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    vad_pb2_grpc.add_VoiceActivityDetectorServicer_to_server(VoiceActivityDetectorService(), server)
    
    server.add_insecure_port(f"[::]:{os.environ['GRPC_PORT']}")
    server.start()

    logging.info("Server started, listening on port %s", os.environ["GRPC_PORT"])
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(
        format="%(levelname)s | %(asctime)s | %(message)s",
        level=logging.INFO
    )
    serve()

#!/usr/bin/env python3
import runpod
import os
import torch
import logging
import sys
import time
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import glob

# Configure logging
log_level = os.environ.get("RUNPOD_DEBUG_LEVEL", "info").upper()
if log_level == "DEBUG":
    level = logging.DEBUG
elif log_level == "WARNING":
    level = logging.WARNING
elif log_level == "ERROR":
    level = logging.ERROR
else:
    level = logging.INFO

logging.basicConfig(
    level=level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info("------ Starting RunPod Serverless Translation Handler ------")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"CUDA current device: {torch.cuda.current_device()}")
    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Get environment variables
DEFAULT_MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_ID)
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Check if the model path exists and contains model files
def is_valid_model_path(path):
    if not os.path.exists(path):
        logger.warning(f"Model path {path} does not exist")
        return False
    
    # Check for typical model files
    config_file = os.path.join(path, "config.json")
    if not os.path.exists(config_file):
        logger.warning(f"No config.json found in {path}")
        return False
    
    logger.info(f"Found valid model directory at {path}")
    return True

# Determine the actual model path to use
if MODEL_PATH != DEFAULT_MODEL_ID and not is_valid_model_path(MODEL_PATH):
    logger.warning(f"Invalid model path {MODEL_PATH}, falling back to Hugging Face model {DEFAULT_MODEL_ID}")
    MODEL_PATH = DEFAULT_MODEL_ID

logger.info(f"Initializing with model: {MODEL_PATH} on device: {DEVICE}")

# Make sure our main module is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from main.py
try:
    from main import get_lang_info, detect_language
    logger.info("Successfully imported functions from main.py")
except ImportError as e:
    logger.error(f"Failed to import from main.py: {e}")
    raise

# Load models at module startup time (not in the handler)
# This ensures the model is loaded only once when the serverless function starts
try:
    start_time = time.time()
    logger.info("Loading tokenizer and model...")
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_PATH)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    load_time = time.time() - start_time
    logger.info(f"Model loaded successfully on {DEVICE} in {load_time:.2f} seconds")
    
    # Optional: Optimize model for inference if using CUDA
    if DEVICE == "cuda":
        model = model.eval()  # Set to evaluation mode
        # Use mixed precision for faster inference if using newer GPUs
        if torch.cuda.get_device_capability()[0] >= 7:
            logger.info("Using mixed precision for inference")
            model = model.half()  # Convert to FP16 for faster inference on Volta+ GPUs
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

def handler(event):
    """
    This is the RunPod serverless handler function that processes translation requests.
    
    Args:
        event (dict): The event object containing the request data
            - 'input' (dict): Contains the request parameters
                - 'text' (str): The text to translate
                - 'target_lang' (str): The target language code
                - 'source_lang' (str, optional): The source language code
    
    Returns:
        dict: The response object containing the translation result
    """
    try:
        start_time = time.time()
        
        # Get input data
        input_data = event.get("input", {})
        
        # Extract parameters
        text = input_data.get("text")
        target_lang = input_data.get("target_lang")
        source_lang = input_data.get("source_lang")
        
        # Validate required parameters
        if not text:
            return {"error": "Missing required parameter: text"}
        if not target_lang:
            return {"error": "Missing required parameter: target_lang"}
        
        logger.info(f"Processing translation request: {target_lang=}, text length: {len(text)}, {source_lang=}")
        
        # Auto-detect source language if not provided
        detection_time = 0
        is_detected = False
        if not source_lang:
            detection_start = time.time()
            try:
                source_lang = detect_language(text)
                is_detected = True
                detection_time = time.time() - detection_start
                logger.info(f"Auto-detected language: {source_lang} in {detection_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Language detection failed: {e}")
                return {"error": f"Language detection failed: {str(e)}"}
        
        # Get language info objects
        try:
            source_lang_info = get_lang_info(source_lang)
            target_lang_info = get_lang_info(target_lang)
        except Exception as e:
            logger.error(f"Invalid language code: {e}")
            return {"error": str(e)}
        
        # Get full language codes
        full_source_lang = source_lang_info.full_code_with_suffix
        full_target_lang = target_lang_info.full_code_with_suffix
        
        # Log to help with troubleshooting
        logger.info(f"Using source language: {source_lang} → {full_source_lang}")
        logger.info(f"Using target language: {target_lang} → {full_target_lang}")
        
        # Verify both language codes are in tokenizer's vocabulary
        if full_source_lang not in tokenizer.lang_code_to_id:
            logger.error(f"Source language code {full_source_lang} not found in tokenizer vocabulary")
            return {"error": f"Source language '{source_lang}' is not supported by the translation model"}
            
        if full_target_lang not in tokenizer.lang_code_to_id:
            logger.error(f"Target language code {full_target_lang} not found in tokenizer vocabulary")
            return {"error": f"Target language '{target_lang}' is not supported by the translation model"}
        
        # Tokenize with safety limit on input text length
        max_length = 512  # Safe maximum for most models
        input_text = text[:max_length] if len(text) > max_length else text
        
        # Set source language - this automatically handles the language token
        tokenizer.src_lang = full_source_lang
        
        tokenize_start = time.time()
        encoded_text = tokenizer(input_text, return_tensors="pt")
        tokenize_time = time.time() - tokenize_start
        
        forced_bos_token_id = tokenizer.lang_code_to_id.get(full_target_lang)
        
        if forced_bos_token_id is None:
            logger.error(f"Could not get forced_bos_token_id for target language {full_target_lang}")
            return {"error": f"Invalid target language code: {target_lang}"}
        
        logger.info(f"Using forced_bos_token_id={forced_bos_token_id} for target language {full_target_lang}")
        
        # Move tensors to GPU if using CUDA
        if DEVICE == "cuda":
            encoded_text = {key: value.to(DEVICE) for key, value in encoded_text.items()}
        
        # Generate translation using torch.no_grad() for memory efficiency
        generate_start = time.time()
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded_text,
                forced_bos_token_id=forced_bos_token_id,
                decoder_start_token_id=tokenizer.eos_token_id,  # Use eos_token_id as the decoder_start_token_id
                max_length=1024,  # Safe maximum for output length
                num_beams=4,      # Beam search for better quality
            )
        generate_time = time.time() - generate_start
        
        # Decode the translation
        decode_start = time.time()
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        decode_time = time.time() - decode_start
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Log timing details
        logger.info(f"Translation statistics: source={source_lang}, target={target_lang}, chars={len(text)}")
        logger.info(f"Timing: detection={detection_time:.2f}s, tokenize={tokenize_time:.2f}s, " + 
                   f"generate={generate_time:.2f}s, decode={decode_time:.2f}s, total={total_time:.2f}s")
        
        # Return the result
        return {
            "source_lang": source_lang,
            "detected": is_detected,
            "target_lang": target_lang,
            "text": text,
            "translated_text": translated_text,
            "stats": {
                "total_time": round(total_time, 3),
                "detection_time": round(detection_time, 3),
                "translation_time": round(generate_time, 3)
            }
        }
    
    except Exception as e:
        logger.error(f"Error in handler: {e}", exc_info=True)
        return {"error": str(e)}

# Log that we're ready to start handling requests
logger.info("RunPod handler initialization complete, ready to process requests")

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler}) 


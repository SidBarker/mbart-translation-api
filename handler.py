#!/usr/bin/env python3
import runpod
import os
import torch
import logging
import sys
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", "facebook/mbart-large-50-many-to-many-mmt")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

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
    logger.info("Loading tokenizer and model...")
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_PATH)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    logger.info(f"Model loaded successfully on {DEVICE}")
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
        is_detected = False
        if not source_lang:
            try:
                source_lang = detect_language(text)
                is_detected = True
                logger.info(f"Auto-detected language: {source_lang}")
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
        
        # Prepare for translation
        tokenizer.src_lang = full_source_lang
        
        # Tokenize with safety limit on input text length
        max_length = 512  # Safe maximum for most models
        input_text = text[:max_length] if len(text) > max_length else text
        
        encoded_text = tokenizer(input_text, return_tensors="pt")
        forced_bos_token_id = tokenizer.lang_code_to_id.get(full_target_lang)
        
        if forced_bos_token_id is None:
            return {"error": f"Invalid target language code: {target_lang}"}
        
        # Move tensors to GPU if using CUDA
        if DEVICE == "cuda":
            encoded_text = {key: value.to(DEVICE) for key, value in encoded_text.items()}
        
        # Generate translation using torch.no_grad() for memory efficiency
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded_text,
                forced_bos_token_id=forced_bos_token_id,
                max_length=1024,  # Safe maximum for output length
                num_beams=4,      # Beam search for better quality
            )
        
        # Decode the translation
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        # Return the result
        return {
            "source_lang": source_lang,
            "detected": is_detected,
            "target_lang": target_lang,
            "text": text,
            "translated_text": translated_text
        }
    
    except Exception as e:
        logger.error(f"Error in handler: {e}", exc_info=True)
        return {"error": str(e)}

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler}) 
import os
from fastapi import FastAPI, HTTPException
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline
from uvicorn import run
from pydantic import BaseModel, Field
import torch
from config import Settings
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG", "false").lower() == "true" else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SETTINGS = Settings()

DEFAULT_MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"
MODEL_PATH = SETTINGS.model_path

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

logger.info(f'model_path: {MODEL_PATH}')

# Check if DEVICE is set and not empty
DEVICE = getattr(SETTINGS, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'DEVICE: {DEVICE}')

# Load the models
model = MBartForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_PATH)

class TranslationRequest(BaseModel):
    source_lang: str = Field(None, description="Source language code (optional - will be auto-detected if not provided)")
    target_lang: str = Field(..., description="Target language code")
    text: str = Field(..., description="Text to translate")

class LanguageInfo:
    def __init__(self, full_code_with_suffix: str, full_name: str):
        self.full_code_with_suffix = full_code_with_suffix
        self.full_name = full_name
        
# Mapping of language prefixes to LanguageInfo objects
lang_suffix_map = {
    "ar": LanguageInfo("ar_AR", "Arabic"),
    "cs": LanguageInfo("cs_CZ", "Czech"),
    "de": LanguageInfo("de_DE", "German"),
    "en": LanguageInfo("en_XX", "English"),
    "es": LanguageInfo("es_XX", "Spanish"),
    "et": LanguageInfo("et_EE", "Estonian"),
    "fi": LanguageInfo("fi_FI", "Finnish"),
    "fr": LanguageInfo("fr_XX", "French"),
    "gu": LanguageInfo("gu_IN", "Gujarati"),
    "hi": LanguageInfo("hi_IN", "Hindi"),
    "it": LanguageInfo("it_IT", "Italian"),
    "ja": LanguageInfo("ja_XX", "Japanese"),
    "kk": LanguageInfo("kk_KZ", "Kazakh"),
    "ko": LanguageInfo("ko_KR", "Korean"),
    "lt": LanguageInfo("lt_LT", "Lithuanian"),
    "lv": LanguageInfo("lv_LV", "Latvian"),
    "my": LanguageInfo("my_MM", "Burmese"),
    "ne": LanguageInfo("ne_NP", "Nepali"),
    "nl": LanguageInfo("nl_XX", "Dutch"),
    "ro": LanguageInfo("ro_RO", "Romanian"),
    "ru": LanguageInfo("ru_RU", "Russian"),
    "si": LanguageInfo("si_LK", "Sinhala"),
    "tr": LanguageInfo("tr_TR", "Turkish"),
    "vi": LanguageInfo("vi_VN", "Vietnamese"),
    "zh": LanguageInfo("zh_CN", "Chinese"),
    "af": LanguageInfo("af_ZA", "Afrikaans"),
    "az": LanguageInfo("az_AZ", "Azerbaijani"),
    "bn": LanguageInfo("bn_IN", "Bengali"),
    "fa": LanguageInfo("fa_IR", "Persian"),
    "he": LanguageInfo("he_IL", "Hebrew"),
    "hr": LanguageInfo("hr_HR", "Croatian"),
    "id": LanguageInfo("id_ID", "Indonesian"),
    "ka": LanguageInfo("ka_GE", "Georgian"),
    "km": LanguageInfo("km_KH", "Khmer"),
    "mk": LanguageInfo("mk_MK", "Macedonian"),
    "ml": LanguageInfo("ml_IN", "Malayalam"),
    "mn": LanguageInfo("mn_MN", "Mongolian"),
    "mr": LanguageInfo("mr_IN", "Marathi"),
    "pl": LanguageInfo("pl_PL", "Polish"),
    "ps": LanguageInfo("ps_AF", "Pashto"),
    "pt": LanguageInfo("pt_XX", "Portuguese"),
    "sv": LanguageInfo("sv_SE", "Swedish"),
    "sw": LanguageInfo("sw_KE", "Swahili"),
    "ta": LanguageInfo("ta_IN", "Tamil"),
    "te": LanguageInfo("te_IN", "Telugu"),
    "th": LanguageInfo("th_TH", "Thai"),
    "tl": LanguageInfo("tl_XX", "Tagalog"),
    "uk": LanguageInfo("uk_UA", "Ukrainian"),
    "ur": LanguageInfo("ur_PK", "Urdu"),
    "xh": LanguageInfo("xh_ZA", "Xhosa"),
    "gl": LanguageInfo("gl_ES", "Galician"),
    "sl": LanguageInfo("sl_SI", "Slovene"),
}

# Map from full code to language code for detection
full_code_to_lang = {info.full_code_with_suffix: code for code, info in lang_suffix_map.items()}

# Initialize language detection pipeline
try:
    from transformers import pipeline
    # Use CamemBERT-based language identification model
    lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    logger.info("Language detection pipeline initialized successfully")
except Exception as e:
    logger.error(f"Error initializing language detection: {e}")
    lang_detector = None

def detect_language(text):
    """Detect the language of the input text"""
    if not lang_detector:
        raise HTTPException(status_code=500, detail="Language detection model not available")
    
    try:
        # Map language detector output to mBART language codes
        lang_mapping = {
            "arabic": "ar",
            "czech": "cs", 
            "german": "de",
            "english": "en",
            "spanish": "es",
            "estonian": "et",
            "finnish": "fi",
            "french": "fr",
            "gujarati": "gu",
            "hindi": "hi",
            "italian": "it",
            "japanese": "ja",
            "kazakh": "kk",
            "korean": "ko",
            "lithuanian": "lt",
            "latvian": "lv",
            "burmese": "my",
            "nepali": "ne",
            "dutch": "nl",
            "romanian": "ro",
            "russian": "ru",
            "sinhala": "si",
            "turkish": "tr",
            "vietnamese": "vi",
            "chinese": "zh",
            "afrikaans": "af",
            "azerbaijani": "az",
            "bengali": "bn",
            "persian": "fa",
            "hebrew": "he",
            "croatian": "hr",
            "indonesian": "id",
            "georgian": "ka",
            "khmer": "km",
            "macedonian": "mk",
            "malayalam": "ml",
            "mongolian": "mn",
            "marathi": "mr",
            "polish": "pl",
            "pashto": "ps",
            "portuguese": "pt",
            "swedish": "sv",
            "swahili": "sw",
            "tamil": "ta",
            "telugu": "te",
            "thai": "th",
            "tagalog": "tl",
            "ukrainian": "uk",
            "urdu": "ur",
            "xhosa": "xh",
            "galician": "gl",
            "slovenian": "sl",
        }
        
        # Truncate text if too long for the model
        sample_text = text[:512] if len(text) > 512 else text
        
        # Detect language
        result = lang_detector(sample_text)[0]
        detected_lang = result["label"].lower()
        confidence = result["score"]
        
        logger.debug(f"Detected language: {detected_lang} with confidence {confidence}")
        
        # Map to mBART language code
        mbart_lang_code = lang_mapping.get(detected_lang)
        if mbart_lang_code is None:
            logger.warning(f"Language {detected_lang} not mapped to an mBART language code, falling back to English")
            return "en"
        
        # Verify the mapped code exists in our supported languages
        if mbart_lang_code not in lang_suffix_map:
            logger.warning(f"Mapped language code {mbart_lang_code} not found in supported mBART languages, falling back to English")
            return "en"
            
        return mbart_lang_code
    except Exception as e:
        logger.error(f"Error in language detection: {e}")
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")

def get_lang_info(language_code: str):
    lang_info = lang_suffix_map.get(language_code)
    if lang_info is None:
        raise HTTPException(status_code=404, detail=f"Language code {language_code} not found")
    return lang_info


app = FastAPI(title='Translation API Service')

@app.get("/v1/lang/support", summary='Get supported languages')
async def get_languages(lang_code: str = None):
    if lang_code is None:
        # Return information about all languages as an array
        languages_info = [
            {"code": code, "full_code": info.full_code_with_suffix, "name": info.full_name}
            for code, info in lang_suffix_map.items()
        ]
        return languages_info
    else:
       # Return information about a specific language
        lang_info = get_lang_info(lang_code)
        if lang_info is None:
            raise HTTPException(status_code=404, detail=f"Language code {lang_code} not found")
        return {
            "code": lang_code,
            "full_code": lang_info.full_code_with_suffix,
            "name": lang_info.full_name,
        }
 
@app.post("/v1/lang/translate", summary='Translate text')
async def translate_text(request: TranslationRequest):
    text = request.text
    target_lang = request.target_lang
    
    # Auto-detect source language if not provided
    source_lang = request.source_lang
    if source_lang is None:
        source_lang = detect_language(text)
        logger.info(f"Auto-detected language: {source_lang}")
    
    # Get the LanguageInfo object based on the language prefix
    source_lang_info = get_lang_info(source_lang)
    target_lang_info = get_lang_info(target_lang)

    # Access the properties of the LanguageInfo object
    full_source_lang = source_lang_info.full_code_with_suffix
    full_target_lang = target_lang_info.full_code_with_suffix

    logger.debug(f"Translating from {full_source_lang} to {full_target_lang}")
    
    tokenizer.src_lang = full_source_lang
    encoded_text = tokenizer(text, return_tensors="pt")
    forced_bos_token_id = tokenizer.lang_code_to_id.get(full_target_lang)
    if forced_bos_token_id is None:
        raise HTTPException(status_code=400, detail=f"Invalid target language code: {target_lang}")
    
    # Generate translation
    if DEVICE == 'cuda':
        # If using CUDA, move tensors to the GPU
        encoded_text = {key: value.to('cuda') for key, value in encoded_text.items()}

    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=forced_bos_token_id)
    
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return {
        "source_lang": source_lang, 
        "detected": request.source_lang is None,
        "target_lang": target_lang, 
        "text": text, 
        "translated_text": translated_text
    }

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=3000)

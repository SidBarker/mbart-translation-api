# mBART Translation API

This project provides a translation API service using the mBART-large-50 model. It supports translation between 50 languages with automatic language detection.

## Model Information

The solution uses the [facebook/mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) model from Hugging Face, which enables translation between any pair of 50 languages. It was introduced in the paper "Multilingual Translation with Extensible Multilingual Pretraining and Finetuning".

## Features

- Automatic language detection of source text
- Translation between any pair of 50 supported languages  
- FastAPI-based REST API
- GPU acceleration support
- Docker containerization for easy deployment on RunPod.io
- Configurable debugging via environment variables

## Supported Languages

Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)

## Installation and Usage

### Local Development

```bash
git clone https://github.com/your-username/mbart-translation-api
cd mbart-translation-api
pip install -r requirements.txt
# Set environment variables
export MODEL_PATH=/path/to/model
export DEVICE=cuda  # or cpu if no GPU available
export DEBUG=true   # optional for debug logging
uvicorn main:app --host 0.0.0.0 --port 23129
```

### Docker Build and Run

```bash
docker build -t mbart-translation-api .
docker run -p 23129:23129 -v /path/to/models:/data/models mbart-translation-api
```

### For GPU support

```bash
docker run --gpus all -p 23129:23129 -v /path/to/models:/data/models mbart-translation-api
```

## API Endpoints

The API is accessible at http://localhost:23129/docs which provides Swagger UI documentation.

### Main Endpoints:

1. `POST /v1/lang/translate` - Translate text between languages
   - Auto-detects source language if not specified
   - Requires target language and text

2. `GET /v1/lang/support` - Get information about supported languages

## Example Usage

See `api_request.py` for example API usage:

```python
import requests

# Auto-detect source language
response = requests.post("http://localhost:23129/v1/lang/translate", 
                         json={"target_lang": "en", "text": "Bonjour le monde"})
print(response.json())

# Specify source language
response = requests.post("http://localhost:23129/v1/lang/translate", 
                         json={"source_lang": "fr", "target_lang": "es", "text": "Bonjour le monde"})
print(response.json())
```
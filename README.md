# mBART Translation API for RunPod Serverless

This project provides a translation API service optimized for RunPod.io serverless deployment using the mBART-large-50 model. It supports translation between 50 languages with automatic language detection.

## Model Information

The solution uses the [facebook/mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) model from Hugging Face, which enables translation between any pair of 50 languages. It was introduced in the paper "Multilingual Translation with Extensible Multilingual Pretraining and Finetuning".

## Features

- Automatic language detection of source text
- Translation between any pair of 50 supported languages  
- GPU acceleration optimized for RunPod.io
- Performance metrics and detailed logging
- Configurable debugging via environment variables
- Automatic model downloading if local files aren't found
- FP16 mixed precision on compatible GPUs

## Supported Languages

Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)

## Model Loading Behavior

The serverless handler can load the mBART model in two ways:

1. **Local Model**: If you have the model files saved at the path specified by `MODEL_PATH` environment variable, it will load them from there.
2. **Automatic Download**: If local model files aren't found or are incomplete, the application will automatically fall back to downloading the model from Hugging Face.

For best performance, it's recommended to pre-download the model to a persistent RunPod volume.

## Docker Build

```bash
docker build -t yourusername/mbart-translation-api:latest .
docker push yourusername/mbart-translation-api:latest
```

## Pre-downloading Model Files

To pre-download the model files to your RunPod volume before deploying:

```bash
# Create a directory for the model
mkdir -p /path/to/models

# Download the model files (can run this inside the container too)
python -c "from transformers import MBartForConditionalGeneration, MBart50TokenizerFast; \
  model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt'); \
  tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt'); \
  model.save_pretrained('/path/to/models'); \
  tokenizer.save_pretrained('/path/to/models')"
```

## RunPod.io Serverless Deployment

To deploy to RunPod:

1. Build and push your Docker image to a registry (Docker Hub, GHCR, etc.)
   ```bash
   docker build -t yourusername/mbart-translation-api:latest .
   docker push yourusername/mbart-translation-api:latest
   ```

2. In the RunPod.io serverless UI:
   - Create a new serverless template
   - Enter your container image URL
   - Set appropriate GPU type based on your needs (recommend at least 8GB VRAM)
   - Set environment variables if needed (MODEL_PATH is set to /runpod-volume/models by default)
   - Create a volume to store models and attach it to /runpod-volume

3. Deploy your serverless container
   - When your template is created, deploy it as a serverless function
   - RunPod will handle scaling based on traffic

### Important Notes for RunPod Deployment

The first request may take longer if the model needs to be downloaded. For optimal performance:

1. Create a RunPod volume and mount it at `/runpod-volume`
2. Pre-download the model files to this volume
3. Make sure the volume persists between deployments

If you don't pre-download the model, it will be downloaded automatically on first use and cached in the volume (if one is attached), or in the container's filesystem (less optimal).

### Testing the RunPod Handler Locally

Before deploying to RunPod, you can test the handler locally:

```bash
python runpod_test.py
```

## RunPod API Request Format

When using the RunPod API endpoint, your requests should follow this format:

```json
{
  "input": {
    "text": "Hello world, how are you today?",
    "target_lang": "fr",
    "source_lang": "en"  // Optional - will auto-detect if not provided
  }
}
```

The response will be in this format:

```json
{
  "output": {
    "source_lang": "en",
    "detected": false,
    "target_lang": "fr",
    "text": "Hello world, how are you today?",
    "translated_text": "Bonjour le monde, comment allez-vous aujourd'hui?",
    "stats": {
      "total_time": 0.342,
      "detection_time": 0,
      "translation_time": 0.289
    }
  }
}
```

## Environment Variables

The serverless handler supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| MODEL_PATH | Path to the model files | /runpod-volume/models |
| DEVICE | Device to use for inference | cuda (if available) |
| RUNPOD_DEBUG_LEVEL | Logging level (debug, info, warning, error) | info |
| HF_HOME | Path to the Hugging Face cache | /runpod-volume/huggingface |
| TRANSFORMERS_CACHE | Path to the Transformers cache | /runpod-volume/cache |
| TORCH_HOME | Path to the PyTorch models | /runpod-volume/torch |
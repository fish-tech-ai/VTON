from typing import Union, List, Optional
import base64

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
from diffusers.image_processor import VaeImageProcessor
from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from vton_utils import resize_and_crop, resize_and_padding, init_weight_dtype
from google.cloud import storage
from google.cloud import secretmanager_v1
from Crypto.Cipher import AES
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI(title="CatVTON Virtual Try-On API")

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PRECISION = "fp16"
USE_TF32 = True
MODELS_BUCKET_NAME = "style-me-models"
IMAGES_BUCKET_NAME = "styleme-images-storage"
SECRET_PROJECT_ID = "870021534924"
SECRET_ID = "AES_256"
LOCAL_MODEL_DIR = "/tmp/models"
SMALL_IMAGE_WIDTH = 300

torch.backends.cuda.matmul.allow_tf32 = USE_TF32
torch.backends.cudnn.allow_tf32 = USE_TF32
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)

storage_client = storage.Client()
secret_client = secretmanager_v1.SecretManagerServiceClient()

pipeline: Optional[CatVTONPipeline] = None
automasker: Optional[AutoMasker] = None
vae_processor = None
mask_processor = None
AES_KEY = None


def get_secret(secret_id: str, project_id: str) -> bytes:
    try:
        secret_name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = secret_client.access_secret_version(request={"name": secret_name})
        secret_value = response.payload.data.decode("UTF-8").strip()
        aes_key = bytes.fromhex(secret_value)
        if len(aes_key) != 32:
            raise ValueError(f"AES key must be 32 bytes, got {len(aes_key)} bytes")
        logger.info("Successfully retrieved AES key from Secret Manager")
        return aes_key
    except Exception as e:
        logger.error(f"Failed to retrieve secret: {str(e)}")
        raise


def download_from_gcs(bucket_name, source_path, local_path):
    bucket = storage_client.bucket(bucket_name)
    prefix = source_path + "/"
    blobs = bucket.list_blobs(prefix=prefix)
    os.makedirs(local_path, exist_ok=True)
    for blob in blobs:
        relative_path = blob.name[len(prefix):]
        local_file_path = os.path.join(local_path, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)
        logger.info(f"Downloaded {blob.name} to {local_file_path}")


@app.on_event("startup")
async def startup_event():
    global pipeline, automasker, vae_processor, mask_processor, AES_KEY
    AES_KEY = get_secret(SECRET_ID, SECRET_PROJECT_ID)
    sd_path = f"{LOCAL_MODEL_DIR}/stable-diffusion-inpainting"
    catvton_path = f"{LOCAL_MODEL_DIR}/catvton"
    models_exist = os.path.exists(sd_path) and os.path.exists(catvton_path)
    if not models_exist:
        logger.info("Models not found in /tmp/models, downloading from GCS...")
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        download_from_gcs(MODELS_BUCKET_NAME, "stable-diffusion-inpainting", sd_path)
        download_from_gcs(MODELS_BUCKET_NAME, "catvton", catvton_path)
    else:
        logger.info("Models already exist in /tmp/models, skipping download.")
    try:
        logger.info("Initializing CatVTON Pipeline...")
        pipeline = CatVTONPipeline(
            base_ckpt=sd_path,
            attn_ckpt=catvton_path,
            attn_ckpt_version="mix",
            weight_dtype=init_weight_dtype(MIXED_PRECISION),
            use_tf32=USE_TF32,
            device=DEVICE,
            skip_safety_check=True
        )
    except Exception as e:
        logger.exception(f"Failed to initialize CatVTON Pipeline: {str(e)}")
        raise
    logger.info("Initializing image processors...")
    vae_processor = VaeImageProcessor(vae_scale_factor=8)
    mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True,
                                       do_convert_grayscale=True)
    try:
        logger.info("Initializing AutoMasker...")
        automasker = AutoMasker(
            densepose_ckpt=f"{catvton_path}/DensePose",
            schp_ckpt=f"{catvton_path}/SCHP",
            device=DEVICE
        )
    except Exception as e:
        logger.exception(f"Failed to initialize AutoMasker: {str(e)}")
        raise
    logger.info("Startup completed successfully.")


def fetch_image_from_gcs(image_id: str) -> bytes:
    logger.info(f"Fetching image from GCS: {image_id}")
    bucket = storage_client.bucket(IMAGES_BUCKET_NAME)
    blob = bucket.blob(f"{image_id}.WEBP")
    data = blob.download_as_bytes()
    logger.info(f"Fetched data length: {len(data)}, first 32 bytes: {data[:32].hex()}")
    return data

def resize_image(image: Image.Image) -> Image.Image:
    aspect_ratio = image.height / image.width
    height = int(SMALL_IMAGE_WIDTH * aspect_ratio)
    return image.resize((SMALL_IMAGE_WIDTH, height))


def save_images_to_gcs(full_image: Image.Image, image_id: str, user_id: str):
    try:
        full_buffer = image_to_bytes(full_image)
        full_bytes = full_buffer.getvalue()
        encrypted_full_bytes = encode_image(full_bytes)

        bucket = storage_client.bucket(IMAGES_BUCKET_NAME)

        full_blob = bucket.blob(f"outfit-images-full/{user_id}/{image_id}.WEBP")
        full_blob.upload_from_string(encrypted_full_bytes, content_type="image/webp")
        logger.info(f"Saved full image to: outfit-images-full/{user_id}/{image_id}.WEBP")

        resized_image = resize_image(full_image)
        resized_buffer = image_to_bytes(resized_image)
        resized_bytes = resized_buffer.getvalue()
        encrypted_resized_bytes = encode_image(resized_bytes)

        resized_blob = bucket.blob(f"outfit-images/{user_id}/{image_id}.WEBP")
        resized_blob.upload_from_string(encrypted_resized_bytes, content_type="image/webp")
        logger.info(f"Saved resized image to: outfit-images/{user_id}/{image_id}.WEBP")

    except Exception as e:
        logger.error(f"Failed to save images to GCS: {str(e)}")
        raise


def decode_image(encrypted_data: bytes) -> bytes:
    nonce = encrypted_data[:16]
    tag = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]
    cipher = AES.new(AES_KEY, AES.MODE_GCM, nonce=nonce)
    try:
        decrypted_data = cipher.decrypt_and_verify(ciphertext, tag)
    except ValueError as e:
        raise Exception("Decryption failed: The data is corrupted or the tag is invalid.") from e
    return decrypted_data


def encode_image(image_data: bytes) -> bytes:
    cipher = AES.new(AES_KEY, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(image_data)
    encrypted_data = cipher.nonce + tag + ciphertext
    return encrypted_data


def preprocess_image(WIDTH, HEIGHT, img: Image.Image, is_cloth: bool = False) -> torch.Tensor:
    if is_cloth:
        img = resize_and_padding(img, (WIDTH, HEIGHT))
    else:
        img = resize_and_crop(img, (WIDTH, HEIGHT))
    return vae_processor.preprocess(img, HEIGHT, WIDTH)[0]


def preprocess_mask(WIDTH, HEIGHT, mask: Image.Image) -> torch.Tensor:
    mask = resize_and_crop(mask, (WIDTH, HEIGHT))
    return mask_processor.preprocess(mask, HEIGHT, WIDTH)[0]


def image_to_bytes(img: Image.Image) -> BytesIO:
    buffered = BytesIO()
    img.save(buffered, format="WEBP")
    buffered.seek(0)
    return buffered


def run_inference(person_img: Image.Image, cloth_img: Image.Image, cloth_type: str, inference_steps: int) -> Image.Image:
    torch.cuda.empty_cache()

    WIDTH = 600
    aspect_ratio = person_img.height / person_img.width
    HEIGHT = int(WIDTH * aspect_ratio)

    person_img_processed = resize_and_crop(person_img, (WIDTH, HEIGHT))
    cloth_img_processed = resize_and_padding(cloth_img, (WIDTH, HEIGHT))
    mask_result = automasker(person_img_processed, cloth_type)
    mask_img = mask_result["mask"]
    if mask_img is None:
        raise HTTPException(status_code=500, detail="Failed to generate mask")
    mask_img = mask_processor.blur(mask_img, blur_factor=9)
    person_tensor = preprocess_image(WIDTH, HEIGHT, person_img_processed).unsqueeze(0).to(DEVICE)
    cloth_tensor = preprocess_image(WIDTH, HEIGHT, cloth_img_processed, is_cloth=True).unsqueeze(0).to(DEVICE)
    mask_tensor = preprocess_mask(WIDTH, HEIGHT, mask_img).unsqueeze(0).to(DEVICE)
    logger.info(f"Input shapes: person={person_tensor.shape}, cloth={cloth_tensor.shape}, mask={mask_tensor.shape}")
    logger.info(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            result = pipeline(
                image=person_tensor,
                condition_image=cloth_tensor,
                mask=mask_tensor,
                num_inference_steps=inference_steps,
                guidance_scale=2.5,
                generator=generator,
                height=HEIGHT,
                width=WIDTH,
                skip_safety_check=True
            )[0]
    torch.cuda.empty_cache()
    return result


class PredictRequest(BaseModel):
    person_image_id: str
    cloth_upper_image_id: Union[str, None] = None
    cloth_lower_image_id: Union[str, None] = None
    cloth_overall_image_id: Union[str, None] = None
    generated_image_id: str
    user_id: str
    inference_steps: int = 20


class VertexRequest(BaseModel):
    instances: List[PredictRequest]
    parameters: Union[dict, None] = None

class PredictResponse(BaseModel):
    predictions: List[str]


@app.get("/health")
async def health():
    return {"status": "OK"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: VertexRequest):
    try:
        if not request.instances:
            raise HTTPException(status_code=400, detail="No instances provided")

        instance = request.instances[0]
        logger.info(
            f"Received request: person_image_id={instance.person_image_id}, cloth_upper_image_id={instance.cloth_upper_image_id}, "
            f"cloth_lower_image_id={instance.cloth_lower_image_id}, cloth_overall_image_id={instance.cloth_overall_image_id}")

        cloth_ids = [instance.cloth_upper_image_id, instance.cloth_lower_image_id, instance.cloth_overall_image_id]
        provided_cloth_ids = [cid for cid in cloth_ids if cid is not None]
        if len(provided_cloth_ids) == 0:
            raise HTTPException(status_code=400, detail="At least one cloth ID must be provided")
        if instance.cloth_overall_image_id and (instance.cloth_upper_image_id or instance.cloth_lower_image_id):
            raise HTTPException(status_code=400, detail="Overall cloth ID cannot be combined with upper or lower IDs")
        if pipeline is None or automasker is None:
            logger.error("Pipeline or automasker not initialized")
            raise HTTPException(status_code=503, detail="Service not ready: Models are still loading")

        logger.info(f"Fetching person image: {instance.person_image_id}")
        person_data_encrypted = fetch_image_from_gcs(instance.person_image_id)
        logger.info("Decoding person image")
        person_data = decode_image(person_data_encrypted)
        person_img = Image.open(BytesIO(person_data)).convert("RGB")
        result = person_img

        if instance.cloth_overall_image_id:
            logger.info(f"Fetching overall cloth image: {instance.cloth_overall_image_id}")
            cloth_data_encrypted = fetch_image_from_gcs(instance.cloth_overall_image_id)
            logger.info("Decoding overall cloth image")
            cloth_data = decode_image(cloth_data_encrypted)
            cloth_img = Image.open(BytesIO(cloth_data)).convert("RGB")
            logger.info("Running inference for overall cloth")
            result = run_inference(result, cloth_img, "overall", instance.inference_steps)
        elif instance.cloth_upper_image_id or instance.cloth_lower_image_id:
            if instance.cloth_upper_image_id and instance.cloth_lower_image_id:
                instance.inference_steps = 15
            if instance.cloth_upper_image_id:
                logger.info(f"Fetching upper cloth image: {instance.cloth_upper_image_id}")
                cloth_data_encrypted = fetch_image_from_gcs(instance.cloth_upper_image_id)
                logger.info("Decoding upper cloth image")
                cloth_data = decode_image(cloth_data_encrypted)
                cloth_img = Image.open(BytesIO(cloth_data)).convert("RGB")
                logger.info("Running inference for upper cloth")
                result = run_inference(result, cloth_img, "upper", instance.inference_steps)
            if instance.cloth_lower_image_id:
                logger.info(f"Fetching lower cloth image: {instance.cloth_lower_image_id}")
                cloth_data_encrypted = fetch_image_from_gcs(instance.cloth_lower_image_id)
                logger.info("Decoding lower cloth image")
                cloth_data = decode_image(cloth_data_encrypted)
                cloth_img = Image.open(BytesIO(cloth_data)).convert("RGB")
                logger.info("Running inference for lower cloth")
                result = run_inference(result, cloth_img, "lower", instance.inference_steps)

        logger.info(f"Saving generated image to GCS: {instance.generated_image_id}")
        save_images_to_gcs(result, instance.generated_image_id, instance.user_id)

        torch.cuda.empty_cache()

        return {"predictions": ["success"]}

    except Exception as e:
        torch.cuda.empty_cache()
        logger.exception(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
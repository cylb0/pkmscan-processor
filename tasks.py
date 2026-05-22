import os
import logging
from shared.messaging import ImageTask
from shared.domain import CardIdentity
from shared.constants import FOLDER_MEDIA
from shared.utils.storage import get_s3_img_key
from shared.aws import AWSClientManager, QueueAlias
from errors.processing import PokemonCardDetectionError
from pokemon_card_processor import PokemonCardProcessor
from utils.image_utils import save_as_webp

logger = logging.getLogger(__name__)


def process_image(
    task: ImageTask, client: AWSClientManager, processor: PokemonCardProcessor
):
    """
    Orchestrates the download, processing, and upload of a card image.

    This handler manages local temporary files and coordinates between S3 storage and the CV cropping logic.
    """
    original_extension = os.path.splitext(task.s3_key)[1].lower()
    card_data: CardIdentity = task.card
    local_raw_path = f"/tmp/{card_data.id}_raw{original_extension}"

    local_processed_path = f"/tmp/{card_data.id}_processed.webp"

    try:
        logger.info(f"Processing image for card {card_data.id}")
        print("KEY", task.s3_key)
        client.download_file(task.s3_key, local_raw_path)

        card_img = processor.process(local_raw_path)

        save_as_webp(card_img, local_processed_path, 90)

        processed_key = get_s3_img_key(
            card=card_data,
            folder=FOLDER_MEDIA,
            extension="webp"
        )
        client.upload_file(local_processed_path, processed_key)
        logger.info(f"Successfully uploaded processed image to {processed_key}")
        print("TYPE", type(task.s3_key), "VALUE", repr(task.s3_key))
        client.delete_file(task.s3_key)
        logger.info(f"Successfully deleted original S3 object: {task.s3_key}")

    except PokemonCardDetectionError as e:
        logger.warning(f"Detection failed for card {card_data.id}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing {card_data.id}: {e}")
    finally:
        for path in [local_raw_path, local_processed_path]:
            if os.path.exists(path):
                os.remove(path)


def process_messages(messages, client, processor: PokemonCardProcessor):
    for msg in messages:
        logger.info(f"Received message: {msg}")
        try:
            task: ImageTask = ImageTask.model_validate_json(msg["Body"])
            card_data: CardIdentity = task.card
            logger.info(f"Processing image for card {card_data.id}")
            process_image(task, client, processor)
            client.delete_message(msg["ReceiptHandle"], QueueAlias.RAW_IMAGES)
            logger.info(f"Image processed and deleted for card {card_data.id}")
        except Exception as e:
            logger.error(f"Failed to process {msg['MessageId']}: {e}")

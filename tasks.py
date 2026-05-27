import os
import logging
from shared.messaging import ImageTask, CardImageProcessedPayload, DBUpdateMessage, DBUpdateType, ImageProcessingStatus
from shared.domain import CardIdentity
from shared.constants import FOLDER_MEDIA
from shared.utils.storage import get_s3_img_key
from shared.aws import AWSClientManager, QueueAlias
from errors.processing import PokemonCardDetectionError
from pokemon_card_processor import PokemonCardProcessor
from utils.image_utils import save_as_webp
from typing import Optional

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

    status = ImageProcessingStatus.SUCCESS
    processed_key = None
    error_msg = None

    try:
        logger.info(f"Processing image for card {card_data.id}")
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
        
        try:
            client.delete_file(task.s3_key)
            logger.info(f"Successfully deleted original S3 object: {task.s3_key}")
        except Exception as e:
            logger.warning(f"Failed to delete original S3 object {task.s3_key}: {e}")

    except PokemonCardDetectionError as e:
        logger.warning(f"Detection failed for card {card_data.id}: {e}")
        status = ImageProcessingStatus.FAILED
        error_msg = str(e)

    except Exception as e:
        logger.error(f"Unexpected error processing {card_data.id}: {e}")
        status = ImageProcessingStatus.FAILED
        error_msg = f"Unexpected error {e}"

    finally:
        _publish_db_update(client, card_data.id, status, processed_key, error_msg)

        for path in [local_raw_path, local_processed_path]:
            if os.path.exists(path):
                os.remove(path)

def _publish_db_update(
        client: AWSClientManager,
        card_id: int,
        status: ImageProcessingStatus,
        processed_key: Optional[str] = None,
        error_msg: Optional[str] = None
):
    try:
        update_event = DBUpdateMessage(
            event_type=DBUpdateType.CARD_IMAGE_PROCESSED,
            payload=CardImageProcessedPayload(
                id=card_id,
                status=status,
                master_image_path=processed_key,
                error_message=error_msg
            )
        )
        client.trigger_database_update(update_event)
        logger.info(f"Published DB Update event ({status.value}) for card {card_id}")
    except Exception as sqs_err:
        logger.critical(f"Failed to send DB Update status to SQS for card {card_id}: {sqs_err}")

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

def process_message(msg: dict, client: AWSClientManager, processor: PokemonCardProcessor) -> None:
    """Parses and orchestrates a single SQS image task.

    If it raises an exception, generic start_sqs_worker will skip the deletion,
    allowing SQS to retry the message later.
    """
    logger.info(f"Received message: {msg}")
    try:
        task: ImageTask = ImageTask.model_validate_json(msg["Body"])
        card_data: CardIdentity = task.card

        process_image(task, client, processor)
        logger.info(f"Image processing complete for card {card_data.id}")
    except Exception as e:
        raise e
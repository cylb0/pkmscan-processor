import logging
import sys
from tasks import process_message
from shared.aws import QueueAlias, aws_client, start_sqs_worker
import os
from pokemon_card_processor import PokemonCardProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def run():
    logger.info("Starting image lab worker")

    model_path = os.getenv("YOLO_SEG_MODEL_LOCALPATH")
    processor = PokemonCardProcessor(model_path)

    start_sqs_worker(
        queue_alias=QueueAlias.RAW_IMAGES,
        message_handler=lambda msg: process_message(msg, aws_client, processor),
        max_messages=5,
        wait_time_seconds=10,
        max_empty_poll=5,
        client=aws_client
    )

if __name__ == "__main__":
    run()

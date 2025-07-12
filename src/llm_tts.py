import os
import aiohttp
import asyncio
import logging
from groq import Groq
from discord import VoiceClient
from src.stream import QueuedStreamingPCMAudio

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class GroqYandexTTS:
    def __init__(self, persona="You are a helpful assistant"):
        self.persona = persona
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.iam_token = os.getenv("YANDEX_IAM_TOKEN")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID", "")  # обязательно укажите в .env
        logger.debug(f"Initialized GroqYandexTTS with persona: {self.persona}")

    async def process_text(self, text: str, voice_client: VoiceClient):
        logger.debug(f"Received text to process: {text}")
        try:
            # 1. Получить ответ от Groq LLM
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": text},
                ],
                model="llama3-8b-8192",  # или другой доступный
            )
            reply = chat_completion.choices[0].message.content
            logger.debug(f"Groq LLM reply: {reply}")
        except Exception as e:
            logger.error(f"Error during Groq LLM completion: {e}", exc_info=True)
            return

        # 2. Преобразовать текст в речь через Yandex SpeechKit
        tts_url = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
        headers = {
            "Authorization": f"Bearer {self.iam_token}",
        }
        data = {
            "text": reply,
            "lang": "ru-RU",  # или "en-US"
            "voice": "alena",  # или другой голос
            "folderId": self.folder_id,
            "format": "lpcm",
            "sampleRateHertz": "48000",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(tts_url, headers=headers, data=data) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"TTS error: {error_text}")
                        return
                    audio_bytes = await resp.read()
                    logger.debug(f"Received {len(audio_bytes)} bytes of audio data from Yandex TTS")
        except Exception as e:
            logger.error(f"Exception during Yandex TTS request: {e}", exc_info=True)
            return

        # 3. Воспроизвести аудио
        try:
            audio_queue = asyncio.Queue()
            await audio_queue.put(audio_bytes)
            await audio_queue.put(None)
            audio_source = QueuedStreamingPCMAudio(audio_queue)
            voice_client.play(audio_source)
            logger.debug("Started playing audio in voice client")
        except Exception as e:
            logger.error(f"Error playing audio: {e}", exc_info=True)

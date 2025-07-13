import asyncio
import traceback
import discord
import wave
import logging
import os
import aiohttp
from typing import Optional
from src.llm_tts import GroqYandexTTS
from discord.ext import voice_recv, commands

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

WHISPER_API_KEY = os.getenv("WHISPER_API_KEY")
WHISPER_API_URL = os.getenv("WHISPER_API_URL")

async def transcribe_with_whisper(filename: str) -> str:
    logger.info("Transcribing audio with Whisper API...")
    try:
        headers = {
            "Authorization": f"Bearer {WHISPER_API_KEY}"
        }
        data = aiohttp.FormData()
        data.add_field('file', open(filename, 'rb'), filename=filename, content_type='audio/wav')

        async with aiohttp.ClientSession() as session:
            async with session.post(WHISPER_API_URL, headers=headers, data=data) as resp:
                if resp.status != 200:
                    logger.error(f"Whisper API error: {resp.status} {await resp.text()}")
                    return "service_error"
                result = await resp.json()
                # Для async-версии API результат может прийти не сразу, а по job_id. 
                # В этом случае нужно делать дополнительный GET-запрос. 
                # Но если API сразу возвращает результат, то:
                text = result.get("text") or result.get("transcription") or ""
                logger.info(f"Whisper transcript: {text}")
                return text.lower() if text else "could_not_understand"
    except Exception as e:
        logger.error(f"Error in Whisper API: {e}", exc_info=True)
        return "error"

class AudioProcessor(voice_recv.AudioSink):
    def __init__(self,
                 user: discord.User,
                 channel: discord.TextChannel,
                 bot: commands.Bot,
                 llm_tts: GroqYandexTTS) -> None:
        super().__init__()
        self.speaking_timeout_task = None
        self.buffer: bytes = b""
        self.target_user: discord.User = user
        self.known_ssrcs = set()
        self.recording_active: bool = False
        self.channel: discord.TextChannel = channel
        self.bot: commands.Bot = bot
        self.llm_tts: GroqYandexTTS = llm_tts

    def wants_opus(self) -> bool:
        return False

    def write(self, user, audio_data):
        if hasattr(audio_data, 'ssrc') and audio_data.ssrc not in self.known_ssrcs:
            self.known_ssrcs.add(audio_data.ssrc)
            logger.info(f"Registered new SSRC: {audio_data.ssrc} from user {user}")

        if self.recording_active and audio_data.pcm:
            if user == self.target_user:
                self.buffer += audio_data.pcm

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_start(self, member: discord.Member) -> None:
        logger.info(f"User {member} started speaking.")

        if member == self.target_user:
            if self.voice_client and self.voice_client.is_playing():
                self.voice_client.stop_playing()

            self.recording_active = True
            if self.speaking_timeout_task:
                self.speaking_timeout_task.cancel()
                self.speaking_timeout_task = None

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_stop(self, member: discord.Member) -> None:
        logger.info(f"User {member.name} stopped speaking.")
        if member == self.target_user:
            self.recording_active = False

            if self.speaking_timeout_task:
                self.speaking_timeout_task.cancel()

            loop = self.bot.loop
            self.speaking_timeout_task = loop.call_later(
                0.7, lambda: asyncio.ensure_future(self.process_recorded_audio())
            )

    async def process_recorded_audio(self):
        if not self.buffer:
            logger.info("No audio buffer to process.")
            return

        try:
            logger.info("Audio capture stopped")
            filename = f"recorded_{self.target_user.id}.wav"
            sample_rate = 48000
            sample_width = 2
            channels = 1

            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(self.buffer)

            logger.info(f"Saved recorded audio to {filename}")

            # Отправляем файл в канал для проверки (можете убрать если не нужно)
            future = asyncio.run_coroutine_threadsafe(
                self.channel.send(file=discord.File(filename)),
                self.bot.loop
            )
            try:
                future.result(timeout=5)
            except Exception as e:
                logger.error(f"Error sending audio file: {e}", exc_info=True)

            # Проверка на пустой/тишину
            audio_length = len(self.buffer) / (sample_rate * sample_width)
            if audio_length < 0.3:
                logger.warning("Audio too short - likely not a complete word")
                self.buffer = b""
                return

            self.buffer = b""

            # Распознавание речи через Whisper API
            transcript = await transcribe_with_whisper(filename)
            if transcript in ["could_not_understand", "service_error", "error"] or not transcript.strip():
                future = asyncio.run_coroutine_threadsafe(
                    self.channel.send("I couldn't understand you."),
                    self.bot.loop
                )
                try:
                    future.result(timeout=5)
                except Exception as e:
                    logger.error(f"Error sending message: {e}", exc_info=True)
                return

            logger.info(f"Text: {transcript}")

            asyncio.run_coroutine_threadsafe(self.llm_tts.process_text(transcript, self.voice_client), self.bot.loop)

        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            traceback.print_exc()

    def cleanup(self) -> None:
        logger.info("AudioSink cleanup complete.")

import asyncio
import traceback
import discord
import wave
import speech_recognition as sr
import logging
from typing import Optional
from src.llm_tts import GroqYandexTTS
from discord.ext import voice_recv, commands

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

recognizer: sr.Recognizer = sr.Recognizer()

def convert_audio_to_text_using_google_speech(audio: sr.AudioData) -> str:
    logger.info("Converting audio to text...")
    try:
        command_text: str = recognizer.recognize_google(audio)
        return command_text.lower()
    except sr.UnknownValueError:
        logger.warning("Speech recognition could not understand the audio")
        return "could_not_understand"
    except sr.RequestError as e:
        logger.error(f"Could not request results from speech recognition service; {e}")
        return "service_error"
    except Exception as e:
        logger.error(f"Error in speech recognition: {e}", exc_info=True)
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
        """Accumulate audio data only when recording is active."""
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
            # Если был запущен таймер завершения записи — отменяем его
            if self.speaking_timeout_task:
                self.speaking_timeout_task.cancel()
                self.speaking_timeout_task = None

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_stop(self, member: discord.Member) -> None:
        logger.info(f"User {member.name} stopped speaking.")
        if member == self.target_user:
            self.recording_active = False

            # Если уже был запущен таймер — отменяем
            if self.speaking_timeout_task:
                self.speaking_timeout_task.cancel()

            # Запускаем таймер: если пользователь не начал говорить снова за 0.7 сек — обрабатываем буфер
            loop = asyncio.get_event_loop()
            self.speaking_timeout_task = loop.call_later(
                0.7, lambda: asyncio.ensure_future(self._finalize_recording())
            )

    async def _finalize_recording(self):
        if not self.buffer:
            logger.info("No audio buffer to process.")
            return

        try:
            logger.info("Audio capture stopped")
            filename = f"recorded_{self.target_user.id}.wav"
            sample_rate = 48000  # Discord's sample rate
            sample_width = 2      # 16-bit audio
            channels = 1

            # Сохраняем буфер в WAV-файл
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

            audio_data = sr.AudioData(self.buffer, sample_rate, sample_width)
            logger.info("Audio capture done. Now convert it to text...")

            wav_data = audio_data.get_wav_data()

            # Проверка на пустой/тишину
            if not wav_data or not wav_data.strip():
                logger.warning("No words captured - audio appears to be silence")
                self.buffer = b""
                return

            # Минимальная длина аудио (0.3 сек)
            audio_length = len(self.buffer) / (sample_rate * sample_width)
            if audio_length < 0.3:
                logger.warning("Audio too short - likely not a complete word")
                self.buffer = b""
                return

            self.buffer = b""

            # Распознавание речи
            if audio_data.get_wav_data().strip():
                logger.info("Audio data is not empty")
                result = convert_audio_to_text_using_google_speech(audio_data)
                if result in ["could_not_understand", "service_error", "error"]:
                    if result == "could_not_understand":
                        future = asyncio.run_coroutine_threadsafe(
                            self.channel.send("I couldn't understand you."),
                            self.bot.loop
                        )
                    elif result == "service_error":
                        future = asyncio.run_coroutine_threadsafe(
                            self.channel.send("I'm having trouble connecting to the speech service. Please try again in a moment."),
                            self.bot.loop
                        )
                    else:
                        future = asyncio.run_coroutine_threadsafe(
                            self.channel.send("Something went wrong. I'm ready to listen again."),
                            self.bot.loop
                        )
                    try:
                        future.result(timeout=5)
                    except Exception as e:
                        logger.error(f"Error sending message: {e}", exc_info=True)
                    return

                logger.info(f"Text: {result}")

                asyncio.run_coroutine_threadsafe(self.llm_tts.process_text(result, self.voice_client), self.bot.loop)

        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            traceback.print_exc()

    def cleanup(self) -> None:
        logger.info("AudioSink cleanup complete.")

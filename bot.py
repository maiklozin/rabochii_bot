# -*- coding: utf-8 -*-
import asyncio
import base64
import io
import json
import logging
import os
import sqlite3
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import aiohttp
from aiogram import Bot, Dispatcher
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Загружаем переменные окружения из файла token.env
load_dotenv("token.env", override=True)

# Настраиваем логирование в терминал и в файл
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger("bot")
user_logger = logging.getLogger("user_messages")
user_logger.setLevel(logging.INFO)
user_logger.propagate = False
user_logger.addHandler(
    logging.FileHandler("logs/user_messages.log", encoding="utf-8")
)
DB_PATH = os.path.join("logs", "messages.db")
db_conn = sqlite3.connect(DB_PATH)
db_conn.execute(
    """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT,
        message_id INTEGER,
        message_text TEXT,
        message_type TEXT,
        timestamp TEXT,
        from_bot INTEGER,
        user_id INTEGER,
        username TEXT
    )
    """
)
def ensure_messages_schema(connection: sqlite3.Connection) -> None:
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(messages)")
    }
    if "is_bot" in columns:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT,
                message_id INTEGER,
                message_text TEXT,
                message_type TEXT,
                timestamp TEXT,
                from_bot INTEGER,
                user_id INTEGER,
                username TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO messages_new (
                id,
                full_name,
                message_id,
                message_text,
                message_type,
                timestamp,
                from_bot,
                user_id,
                username
            )
            SELECT
                id,
                full_name,
                message_id,
                message_text,
                message_type,
                timestamp,
                COALESCE(from_bot, CASE WHEN full_name = 'Bot' THEN 1 ELSE 0 END),
                user_id,
                username
            FROM messages
            """
        )
        connection.execute("DROP TABLE messages")
        connection.execute("ALTER TABLE messages_new RENAME TO messages")
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(messages)")
        }
    if "from_bot" not in columns:
        connection.execute("ALTER TABLE messages ADD COLUMN from_bot INTEGER")
    connection.execute(
        """
        UPDATE messages
        SET from_bot = CASE
            WHEN full_name = 'Bot' THEN 1
            ELSE 0
        END
        WHERE from_bot IS NULL
        """
    )

ensure_messages_schema(db_conn)
db_conn.execute(
    """
    CREATE INDEX IF NOT EXISTS idx_messages_user_id
    ON messages (user_id)
    """
)
db_conn.commit()

# Берем токены из переменных окружения
BOT_TOKEN = os.getenv("BOT_TOKEN")
NEUROAPI_API_KEY = os.getenv("NEUROAPI_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5-nano")
NEUROAPI_MODEL = os.getenv("NEUROAPI_MODEL", DEFAULT_MODEL)
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-3-pro-image-preview")
NEXARA_API_KEY = os.getenv("NEXARA_API_KEY")
NEXARA_BASE_URL = os.getenv("NEXARA_BASE_URL", "https://api.nexara.ru/api/v1")
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "20"))

# Если токен не найден, сразу сообщаем об ошибке
if not BOT_TOKEN:
    raise ValueError(
        "Не найден BOT_TOKEN! Откройте token.env и добавьте строку BOT_TOKEN=ВАШ_ТОКЕН"
    )
if not NEUROAPI_API_KEY:
    raise ValueError(
        "Не найден NEUROAPI_API_KEY! Откройте token.env и добавьте строку NEUROAPI_API_KEY=ВАШ_КЛЮЧ"
    )
if not NEXARA_API_KEY:
    raise ValueError(
        "Не найден NEXARA_API_KEY! Откройте token.env и добавьте строку NEXARA_API_KEY=ВАШ_КЛЮЧ"
    )

# Создаем объекты бота и диспетчера для обработки сообщений
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
ai_client = AsyncOpenAI(
    base_url="https://neuroapi.host/v1",
    api_key=NEUROAPI_API_KEY,
)

SYSTEM_PROMPT = (
    "Отвечай на том языке, на котором написан запрос пользователя. "
    "Отвечай кратко и по делу: не добавляй пояснения, примеры или код, "
    "если их не просили. Отвечай только на последний вопрос."
)

chat_histories: Dict[int, Deque[dict]] = {}
chatgpt_enabled_chats = set()
image_enabled_chats = set()
transcribe_enabled_chats = set()

def build_main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Диалог с ИИ")],
            [KeyboardButton(text="Генерация фото")],
            [KeyboardButton(text="Голос → текст")],
        ],
        resize_keyboard=True,
    )

def get_history(chat_id: int) -> Deque[dict]:
    if chat_id not in chat_histories:
        chat_histories[chat_id] = deque(maxlen=HISTORY_LIMIT)
    return chat_histories[chat_id]

def log_history(messages: List[dict]) -> None:
    logger.info("NeuroAPI request model=%s messages_count=%s", NEUROAPI_MODEL, len(messages))
    logger.info("FULL HISTORY MESSAGES:")
    for idx, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        logger.info("%s. [%s] (%s chars): %s", idx, role.upper(), len(content), content)


def build_message_summary(message: Message) -> str:
    if message.text:
        return message.text
    if message.caption:
        return message.caption
    if message.photo:
        return f"<photo {message.photo[-1].file_id}>"
    if message.document:
        name = message.document.file_name or message.document.file_id
        return f"<document {name}>"
    if message.audio:
        name = message.audio.file_name or message.audio.file_id
        return f"<audio {name}>"
    if message.voice:
        return f"<voice {message.voice.file_id}>"
    if message.video:
        return f"<video {message.video.file_id}>"
    if message.video_note:
        return f"<video_note {message.video_note.file_id}>"
    if message.animation:
        return f"<animation {message.animation.file_id}>"
    if message.sticker:
        return f"<sticker {message.sticker.emoji or ''}>"
    if message.contact:
        return f"<contact {message.contact.phone_number}>"
    if message.location:
        return "<location>"
    if message.venue:
        return "<venue>"
    if message.poll:
        return f"<poll {message.poll.question}>"
    return f"<{message.content_type}>"


def log_user_message(message: Message) -> None:
    user = message.from_user
    chat = message.chat
    content = build_message_summary(message)

    display_name = "none"
    if user:
        name_parts = [user.first_name or "", user.last_name or ""]
        display_name = " ".join(part for part in name_parts if part).strip() or "none"

    logger.info(
        "User message | user_id=%s username=%s name=%s chat_id=%s chat_type=%s content_type=%s content=%s",
        user.id if user else "unknown",
        f"@{user.username}" if user and user.username else "none",
        display_name,
        chat.id if chat else "unknown",
        chat.type if chat else "unknown",
        message.content_type,
        content,
    )
    user_logger.info(
        "user_id=%s username=%s name=%s chat_id=%s chat_type=%s content_type=%s content=%s",
        user.id if user else "unknown",
        f"@{user.username}" if user and user.username else "none",
        display_name,
        chat.id if chat else "unknown",
        chat.type if chat else "unknown",
        message.content_type,
        content,
    )
    db_conn.execute(
        """
        INSERT INTO messages (
            full_name,
            message_id,
            message_text,
            message_type,
            timestamp,
            from_bot,
            user_id,
            username
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            display_name,
            message.message_id,
            content,
            message.content_type,
            message.date.isoformat() if message.date else None,
            0,
            user.id if user else None,
            user.username if user else None,
        ),
    )
    db_conn.commit()


def log_bot_message(reply_text: str, message: Message, content_type: str = "text") -> None:
    user = message.from_user
    db_conn.execute(
        """
        INSERT INTO messages (
            full_name,
            message_id,
            message_text,
            message_type,
            timestamp,
            from_bot,
            user_id,
            username
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "Bot",
            None,
            reply_text,
            content_type,
            message.date.isoformat() if message.date else None,
            1,
            user.id if user else None,
            user.username if user else None,
        ),
    )
    db_conn.commit()

def build_greeting(message: Message) -> str:
    user = message.from_user
    if not user:
        return "Привет! Чтобы начать диалог с ИИ, сначала отправьте /chatgpt."
    name = user.first_name or ""
    username = f"@{user.username}" if user.username else ""
    label = " ".join(part for part in (name, username) if part).strip()
    if not label:
        label = "друг"
    return f"Привет, {label}! Чтобы начать диалог с ИИ, сначала отправьте /chatgpt."

async def fetch_ai_reply(chat_id: int, user_text: str) -> str:
    history = get_history(chat_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(list(history))
    messages.append({"role": "user", "content": user_text})
    log_history(messages)
    try:
        response = await ai_client.chat.completions.create(
            model=NEUROAPI_MODEL,
            messages=messages,
        )
    except Exception:
        logger.exception("AI request failed")
        return "Извините, я не смог получить ответ от ИИ. Попробуйте позже."

    content: Optional[str] = None
    if response.choices:
        content = response.choices[0].message.content
    reply_text = (content or "").strip() or "Извините, не удалось сформировать ответ."
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply_text})
    return reply_text

async def fetch_image(prompt: str) -> Tuple[bytes, str]:
    response = await ai_client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
    )
    if not response.data:
        raise ValueError("NeuroAPI returned no image data")

    item = response.data[0]
    if getattr(item, "b64_json", None):
        return base64.b64decode(item.b64_json), "image.png"
    if getattr(item, "url", None):
        async with aiohttp.ClientSession() as session:
            async with session.get(item.url) as resp:
                resp.raise_for_status()
                return await resp.read(), "image.png"
    raise ValueError("NeuroAPI returned unsupported image payload")

async def fetch_transcription(file_name: str, file_bytes: bytes) -> str:
    url = f"{NEXARA_BASE_URL}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {NEXARA_API_KEY}"}
    form = aiohttp.FormData()
    form.add_field("response_format", "text")
    form.add_field(
        "file",
        file_bytes,
        filename=file_name,
        content_type="application/octet-stream",
    )
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=form) as resp:
            resp.raise_for_status()
            body = await resp.text()
            try:
                data = json.loads(body)
                if isinstance(data, dict) and "text" in data:
                    return (data.get("text") or "").strip()
            except json.JSONDecodeError:
                pass
            return body.strip()

async def read_file_bytes(file_obj) -> bytes:
    if isinstance(file_obj, (bytes, bytearray)):
        return bytes(file_obj)
    if hasattr(file_obj, "read"):
        data = file_obj.read()
        if asyncio.iscoroutine(data):
            data = await data
        return data
    raise TypeError("Unsupported file object for download")

@dp.message()
async def log_handler(message: Message) -> None:
    log_user_message(message)
    raise SkipHandler()


# Обработчик команды /start
@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    # Берем имя пользователя, если его нет - используем нейтральное обращение
    user_name = message.from_user.first_name or "друг"

    reply_text = (
        f"Привет, {user_name}!\n\n"
        "Я бот на aiogram.\n"
        "Выберите режим в меню ниже.\n"
        "Команда /start показывает это приветствие!"
    )
    await message.answer(reply_text, reply_markup=build_main_menu())
    log_bot_message(reply_text, message)

@dp.message(Command("chatgpt"))
async def cmd_chatgpt(message: Message) -> None:
    chatgpt_enabled_chats.add(message.chat.id)
    image_enabled_chats.discard(message.chat.id)
    transcribe_enabled_chats.discard(message.chat.id)
    reply_text = "Режим ChatGPT включен. Напишите сообщение."
    await message.answer(reply_text)
    log_bot_message(reply_text, message)

@dp.message(Command("chatgpt_off"))
async def cmd_chatgpt_off(message: Message) -> None:
    chatgpt_enabled_chats.discard(message.chat.id)
    chat_histories.pop(message.chat.id, None)
    reply_text = "Режим ChatGPT выключен. История очищена."
    await message.answer(reply_text)
    log_bot_message(reply_text, message)

@dp.message(Command("menu"))
async def cmd_menu(message: Message) -> None:
    reply_text = "Выберите режим в меню."
    await message.answer(reply_text, reply_markup=build_main_menu())
    log_bot_message(reply_text, message)

@dp.message(lambda msg: msg.text == "Диалог с ИИ")
async def menu_chatgpt(message: Message) -> None:
    chatgpt_enabled_chats.add(message.chat.id)
    image_enabled_chats.discard(message.chat.id)
    transcribe_enabled_chats.discard(message.chat.id)
    reply_text = "Режим ChatGPT включен. Напишите сообщение."
    await message.answer(reply_text)
    log_bot_message(reply_text, message)

@dp.message(lambda msg: msg.text == "Генерация фото")
async def menu_image(message: Message) -> None:
    image_enabled_chats.add(message.chat.id)
    chatgpt_enabled_chats.discard(message.chat.id)
    transcribe_enabled_chats.discard(message.chat.id)
    reply_text = "Режим генерации фото включен. Отправьте описание картинки."
    await message.answer(reply_text)
    log_bot_message(reply_text, message)

@dp.message(lambda msg: msg.text == "Голос → текст")
async def menu_transcribe(message: Message) -> None:
    transcribe_enabled_chats.add(message.chat.id)
    chatgpt_enabled_chats.discard(message.chat.id)
    image_enabled_chats.discard(message.chat.id)
    reply_text = "Режим транскрибации включен. Отправьте аудио/voice."
    await message.answer(reply_text)
    log_bot_message(reply_text, message)

@dp.message(Command("image"))
async def cmd_image(message: Message) -> None:
    if not message.text or len(message.text.split(maxsplit=1)) < 2:
        reply_text = "Напишите /image и далее описание картинки."
        await message.answer(reply_text)
        log_bot_message(reply_text, message)
        return

    prompt = message.text.split(maxsplit=1)[1].strip()
    thinking_message = await message.answer("🎨 Генерирую изображение...")
    try:
        image_bytes, filename = await fetch_image(prompt)
        await message.answer_photo(BufferedInputFile(image_bytes, filename=filename))
        log_bot_message("image generated", message, "photo")
        image_enabled_chats.discard(message.chat.id)
        await message.answer("Выберите режим:", reply_markup=build_main_menu())
    except Exception:
        logger.exception("Image generation failed")
        reply_text = "Не удалось сгенерировать изображение. Попробуйте позже."
        await message.answer(reply_text)
        log_bot_message(reply_text, message)
    finally:
        try:
            await thinking_message.delete()
        except Exception:
            pass

@dp.message(Command("transcribe"))
async def cmd_transcribe(message: Message) -> None:
    audio = message.audio or message.voice or message.document
    if not audio:
        reply_text = "Отправьте аудио/voice или документ и подпишите /transcribe."
        await message.answer(reply_text)
        log_bot_message(reply_text, message)
        return

    file_info = await bot.get_file(audio.file_id)
    file_bytes = await bot.download_file(file_info.file_path)
    file_name = os.path.basename(file_info.file_path) or "audio"
    thinking_message = await message.answer("📝 Распознаю аудио...")
    try:
        data = await read_file_bytes(file_bytes)
        transcript = await fetch_transcription(file_name, data)
        reply_text = transcript or "Не удалось извлечь текст."
        await message.answer(reply_text)
        log_bot_message(reply_text, message)
    except Exception:
        logger.exception("Transcription failed")
        reply_text = "Не удалось распознать аудио. Попробуйте позже."
        await message.answer(reply_text)
        log_bot_message(reply_text, message)
    finally:
        try:
            await thinking_message.delete()
        except Exception:
            pass

# Обработчик всех остальных сообщений (эхо)
@dp.message(~Command(commands=["start"]))
async def echo_handler(message: Message) -> None:
    if message.chat.id in transcribe_enabled_chats:
        audio = message.audio or message.voice or message.document
        if not audio:
            reply_text = "Отправьте аудио/voice или документ для транскрибации."
            await message.answer(reply_text)
            log_bot_message(reply_text, message)
            return
        thinking_message = await message.answer("📝 Распознаю аудио...")
        try:
            file_info = await bot.get_file(audio.file_id)
            file_bytes = await bot.download_file(file_info.file_path)
            file_name = os.path.basename(file_info.file_path) or "audio"
            data = await read_file_bytes(file_bytes)
            transcript = await fetch_transcription(file_name, data)
            reply_text = transcript or "Не удалось извлечь текст."
            await message.answer(reply_text)
            log_bot_message(reply_text, message)
            transcribe_enabled_chats.discard(message.chat.id)
            await message.answer("Выберите режим:", reply_markup=build_main_menu())
        except Exception:
            logger.exception("Transcription failed")
            reply_text = "Не удалось распознать аудио. Попробуйте позже."
            await message.answer(reply_text)
            log_bot_message(reply_text, message)
        finally:
            try:
                await thinking_message.delete()
            except Exception:
                pass
        return

    if message.chat.id in image_enabled_chats:
        if not message.text:
            reply_text = "Отправьте текстовое описание для генерации картинки."
            await message.answer(reply_text)
            log_bot_message(reply_text, message)
            return
        thinking_message = await message.answer("🎨 Генерирую изображение...")
        try:
            image_bytes, filename = await fetch_image(message.text.strip())
            await message.answer_photo(
                BufferedInputFile(image_bytes, filename=filename)
            )
            log_bot_message("image generated", message, "photo")
            image_enabled_chats.discard(message.chat.id)
            await message.answer("Выберите режим:", reply_markup=build_main_menu())
        except Exception:
            logger.exception("Image generation failed")
            reply_text = "Не удалось сгенерировать изображение. Попробуйте позже."
            await message.answer(reply_text)
            log_bot_message(reply_text, message)
        finally:
            try:
                await thinking_message.delete()
            except Exception:
                pass
        return

    if message.chat.id not in chatgpt_enabled_chats:
        reply_text = build_greeting(message)
        await message.answer(reply_text, reply_markup=build_main_menu())
        log_bot_message(reply_text, message)
        return

    user_text = message.text or message.caption
    if not user_text:
        reply_text = "Пожалуйста, отправьте текстовое сообщение, и я отвечу."
        await message.answer(reply_text)
        log_bot_message(reply_text, message)
        return

    thinking_message = await message.answer("🤔 Думаю...")
    reply_text = await fetch_ai_reply(message.chat.id, user_text)
    try:
        await thinking_message.delete()
    except Exception:
        pass
    await message.answer(reply_text)
    log_bot_message(reply_text, message)


async def main() -> None:
    # Запускаем long-polling, бот будет работать до остановки процесса
    print("Бот запущен. Нажмите Ctrl+C для остановки.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

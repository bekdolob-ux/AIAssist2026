import os
import json
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from groq import Groq
from openai import OpenAI
from google import genai
from google.cloud import firestore
from google.oauth2 import service_account


# ============================================================
# SETTINGS
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")

FIRESTORE_DATABASE = "ai-studio-e7d6a6f3-8921-414c-887e-4b97383c7bed"

GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3.8-flash"
OPENAI_MODEL = "gpt-5-mini"

MAX_MESSAGE_LENGTH = 4000


# ============================================================
# CLIENTS
# ============================================================

groq_client = None
gemini_client = None
openai_client = None
db = None


if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ GROQ CLIENT READY")
    except Exception as e:
        print("❌ GROQ INIT ERROR:", repr(e))


if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ GEMINI CLIENT READY")
    except Exception as e:
        print("❌ GEMINI INIT ERROR:", repr(e))


if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OPENAI CLIENT READY")
    except Exception as e:
        print("❌ OPENAI INIT ERROR:", repr(e))


# ============================================================
# FIRESTORE
# ============================================================

try:
    if not GOOGLE_CREDENTIALS_JSON:
        raise Exception("GOOGLE_CREDENTIALS_JSON жок")

    credentials_info = json.loads(GOOGLE_CREDENTIALS_JSON)

    credentials = service_account.Credentials.from_service_account_info(
        credentials_info
    )

    db = firestore.Client(
        project=credentials_info["project_id"],
        credentials=credentials,
        database=FIRESTORE_DATABASE,
    )

    print("✅ FIRESTORE CLIENT READY")

except Exception as e:
    print("❌ FIRESTORE INIT ERROR:", repr(e))
    db = None


# ============================================================
# FIRESTORE HELPERS
# ============================================================

def get_user_ref(user_id):
    if db is None:
        return None

    return db.collection("users").document(str(user_id))


def load_user_data(user_id):
    try:
        ref = get_user_ref(user_id)

        if ref is None:
            return {}

        doc = ref.get()

        if not doc.exists:
            return {}

        return doc.to_dict() or {}

    except Exception as e:
        print("❌ USER DATA LOAD ERROR:", repr(e))
        return {}


# ============================================================
# NORMAL MEMORY
# ============================================================

def load_memory(user_id):
    data = load_user_data(user_id)

    history = data.get("history", [])

    if not isinstance(history, list):
        return []

    return history


def save_memory(user_id, user_text, ai_text):
    try:
        ref = get_user_ref(user_id)

        if ref is None:
            return False

        data = load_user_data(user_id)

        history = data.get("history", [])

        if not isinstance(history, list):
            history = []

        history.append(
            {
                "user": str(user_text),
                "assistant": str(ai_text),
            }
        )

        history = history[-50:]

        ref.set(
            {
                "telegram_user_id": str(user_id),
                "history": history,
                "updated": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

        print(
            f"🧠 MEMORY SAVED: user={user_id}, "
            f"dialogs={len(history)}"
        )

        return True

    except Exception as e:
        print("❌ MEMORY SAVE ERROR:", repr(e))
        return False


# ============================================================
# IMPORTANT MEMORY
# ============================================================

def load_important_memory(user_id):
    data = load_user_data(user_id)

    memory = data.get("important_memory", [])

    if not isinstance(memory, list):
        return []

    return memory


def save_important_memory(user_id, memories):
    try:
        if not memories:
            return False

        ref = get_user_ref(user_id)

        if ref is None:
            return False

        old_memory = load_important_memory(user_id)

        for item in memories:
            item = str(item).strip()

            if item and item not in old_memory:
                old_memory.append(item)

        old_memory = old_memory[-100:]

        ref.set(
            {
                "telegram_user_id": str(user_id),
                "important_memory": old_memory,
                "updated": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

        print(
            f"⭐ IMPORTANT MEMORY SAVED: "
            f"{len(old_memory)} items"
        )

        return True

    except Exception as e:
        print(
            "❌ IMPORTANT MEMORY SAVE ERROR:",
            repr(e),
        )
        return False


def clear_memory(user_id):
    try:
        ref = get_user_ref(user_id)

        if ref is None:
            return False

        ref.set(
            {
                "telegram_user_id": str(user_id),
                "history": [],
                "important_memory": [],
                "updated": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

        print(f"🧹 MEMORY CLEARED: {user_id}")

        return True

    except Exception as e:
        print("❌ MEMORY CLEAR ERROR:", repr(e))
        return False


# ============================================================
# MEMORY EXTRACTION
# ============================================================

def extract_important_memory(user_text, ai_text):
    prompt = f"""
Сен жеке AI жардамчынын Memory системасысың.

Сүйлөшүүдөн келечекте пайдалуу боло турган
туруктуу маалыматтарды гана танда.

Сакталуучу маалыматтар:
- узак мөөнөттүү максаттар
- киреше жана карьера максаттары
- долбоорлор
- үйрөнүп жаткан нерселер
- туруктуу кызыгуулар
- жеке артыкчылыктар
- AI жардамчыга болгон туруктуу талаптар

САКТАБА:
- API key
- Telegram token
- пароль
- сыр
- бир жолку суроо
- убактылуу маалымат
- AI өзү сунуштаган нерселер

ТЕК JSON массив кайтар.

Мисал:
["Максат: айына 100 000 сом киреше",
 "Долбоор: AIAssist2026"]

Маанилүү маалымат жок болсо:
[]

Колдонуучу:
{user_text}

AI:
{ai_text}
"""

    try:
        if not groq_client:
            return []

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Сен Memory extractorсиң. "
                        "Ар дайым valid JSON array гана кайтар."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        result = response.choices[0].message.content

        if not result:
            return []

        result = result.strip()

        if result.startswith("```"):
            result = result.replace("```json", "")
            result = result.replace("```", "")
            result = result.strip()

        parsed = json.loads(result)

        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

    except Exception as e:
        print(
            "⚠️ MEMORY EXTRACTION ERROR:",
            repr(e),
        )

    return []


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_memory_context(user_id):
    important = load_important_memory(user_id)
    history = load_memory(user_id)

    context = ""

    if important:
        context += "\n⭐ МААНИЛҮҮ MEMORY:\n"

        for item in important[-100:]:
            context += f"- {item}\n"

    if history:
        context += "\n🧠 АКЫРКЫ ДИАЛОГДОР:\n"

        for item in history[-10:]:
            context += (
                f"Колдонуучу: {item.get('user', '')}\n"
                f"AI: {item.get('assistant', '')}\n"
            )

    return context


# ============================================================
# AI
# ============================================================

def ask_groq(prompt):
    if not groq_client:
        return None

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Сен Бекболоттун жеке AI жардамчысысың. "
                        "Негизги максат — анын кирешесин көбөйтүүгө "
                        "жана өнүгүүсүнө жардам берүү. "
                        "Memory маалыматтарын тиешелүү учурда колдон. "
                        "Так, практикалык жана түшүнүктүү жооп бер."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content

    except Exception as e:
        print("❌ GROQ ERROR:", repr(e))
        return None


def ask_gemini(prompt):
    if not gemini_client:
        return None

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        if response and response.text:
            return response.text

    except Exception as e:
        print("❌ GEMINI ERROR:", repr(e))

    return None


def ask_openai(prompt):
    if not openai_client:
        return None

    try:
        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=(
                "Сен Бекболоттун жеке AI жардамчысысың. "
                "Практикалык, так жана пайдалуу жооп бер.\n\n"
                + prompt
            ),
        )

        return response.output_text

    except Exception as e:
        print("❌ OPENAI ERROR:", repr(e))
        return None


# ============================================================
# AI ROUTER
# ============================================================

def ask_ai(prompt):
    answer = ask_groq(prompt)

    if answer:
        print("🔵 Groq → OK")
        return answer

    answer = ask_gemini(prompt)

    if answer:
        print("🟢 Gemini → OK")
        return answer

    answer = ask_openai(prompt)

    if answer:
        print("🟣 OpenAI → OK")
        return answer

    return "❌ Азыр AI кызматтарынын баары жооп берген жок."


# ============================================================
# SEND LONG MESSAGE
# ============================================================

async def send_long_message(update, text):
    if not text:
        text = "❌ AI бош жооп кайтарды."

    for i in range(0, len(text), MAX_MESSAGE_LENGTH):
        chunk = text[i:i + MAX_MESSAGE_LENGTH]

        await update.message.reply_text(chunk)


# ============================================================
# MESSAGE HANDLER
# ============================================================

async def handle_message(update, context):
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    user_text = update.message.text.strip()

    print(f"📩 USER {user_id}: {user_text}")

    memory_context = await asyncio.to_thread(
        build_memory_context,
        user_id,
    )

    prompt = f"""
Сен Бекболоттун жеке AI жардамчысысың.

{memory_context}

ЖАҢЫ БИЛДИРҮҮ:
{user_text}

Эрежелер:
1. Memory маалыматтарын туура колдон.
2. Билбеген нерсени ойлоп чыгарба.
3. Кыргыз тилинде жооп бер.
4. Практикалык жана пайдалуу жооп бер.
5. Колдонуучунун жеке максаттарын эске ал.
"""

    answer = await asyncio.to_thread(
        ask_ai,
        prompt,
    )

    await send_long_message(
        update,
        answer,
    )

    saved = await asyncio.to_thread(
        save_memory,
        user_id,
        user_text,
        answer,
    )

    if saved:
        print("🧠 Memory автоматтык сакталды")

    important = await asyncio.to_thread(
        extract_important_memory,
        user_text,
        answer,
    )

    if important:
        saved_important = await asyncio.to_thread(
            save_important_memory,
            user_id,
            important,
        )

        if saved_important:
            print("⭐ Important Memory автоматтык сакталды")


# ============================================================
# /START
# ============================================================

async def start(update, context):
    await update.message.reply_text(
        "🤖 Салам, Бекболот!\n\n"
        "Мен сенин жеке AI жардамчыңмын.\n\n"
        "🧠 Normal Memory — диалогдорду сактайт.\n"
        "⭐ Important Memory — маанилүү маалыматтарды "
        "автоматтык сактайт.\n"
        "💰 Максат — кирешеңди көбөйтүүгө жардам берүү.\n\n"
        "/memory — Memory көрүү\n"
        "/clearmemory — Memory тазалоо\n"
        "/testapi — API текшерүү"
    )


# ============================================================
# /MEMORY
# ============================================================

async def memory_command(update, context):
    user_id = update.effective_user.id

    history = await asyncio.to_thread(
        load_memory,
        user_id,
    )

    important = await asyncio.to_thread(
        load_important_memory,
        user_id,
    )

    if not history and not important:
        await update.message.reply_text(
            "🧠 Memory азырынча бош."
        )
        return

    result = (
        "🧠 MEMORY STATUS\n\n"
        f"💬 Диалогдор: {len(history)}\n"
        f"⭐ Маанилүү Memory: {len(important)}"
    )

    if important:
        result += "\n\n⭐ Сакталган маанилүү маалыматтар:\n"

        for item in important[-20:]:
            result += f"• {item}\n"

    await send_long_message(
        update,
        result,
    )


# ============================================================
# /CLEARMEMORY
# ============================================================

async def clear_memory_command(update, context):
    user_id = update.effective_user.id

    success = await asyncio.to_thread(
        clear_memory,
        user_id,
    )

    if success:
        await update.message.reply_text(
            "🧹 Memory толугу менен тазаланды."
        )
    else:
        await update.message.reply_text(
            "❌ Memory тазаланган жок."
        )


# ============================================================
# /TESTAPI
# ============================================================

async def test_api(update, context):
    msg = await update.message.reply_text(
        "🔍 API TEST башталды..."
    )

    groq_ok = False
    gemini_ok = False
    openai_ok = False
    firestore_ok = False

    try:
        result = await asyncio.to_thread(
            ask_groq,
            "Жөн гана OK деп жооп бер.",
        )
        groq_ok = bool(result)
    except Exception as e:
        print("❌ GROQ TEST:", repr(e))

    try:
        result = await asyncio.to_thread(
            ask_gemini,
            "Жөн гана OK деп жооп бер.",
        )
        gemini_ok = bool(result)
    except Exception as e:
        print("❌ GEMINI TEST:", repr(e))

    try:
        result = await asyncio.to_thread(
            ask_openai,
            "Жөн гана OK деп жооп бер.",
        )
        openai_ok = bool(result)
    except Exception as e:
        print("❌ OPENAI TEST:", repr(e))

    try:
        if db is not None:
            test_ref = (
                db.collection("_system_test")
                .document("test")
            )

            test_ref.set(
                {
                    "status": "OK",
                    "timestamp": firestore.SERVER_TIMESTAMP,
                }
            )

            firestore_ok = True

    except Exception as e:
        print("❌ FIRESTORE TEST:", repr(e))

    result = (
        "🔍 API TEST\n\n"
        f"🔵 Groq: "
        f"{'🟢 OK' if groq_ok else '🔴 ERROR'}\n"
        f"🟢 Gemini: "
        f"{'🟢 OK' if gemini_ok else '🔴 ERROR'}\n"
        f"🟣 OpenAI: "
        f"{'🟢 OK' if openai_ok else '🔴 ERROR'}\n"
        f"🟡 Firestore: "
        f"{'🟢 OK' if firestore_ok else '🔴 ERROR'}"
    )

    await msg.edit_text(result)


# ============================================================
# HEALTH SERVER
# ============================================================

class HealthHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/plain",
        )

        self.end_headers()

        self.wfile.write(
            b"AIAssist2026 is running!"
        )

    def log_message(self, format, *args):
        return


def run_health_server():
    port = int(
        os.getenv(
            "PORT",
            "10000",
        )
    )

    server = HTTPServer(
        ("0.0.0.0", port),
        HealthHandler,
    )

    print(
        f"🌐 Health server running on port {port}"
    )

    server.serve_forever()


# ============================================================
# MAIN
# ============================================================

def main():

    if not TELEGRAM_TOKEN:
        raise Exception(
            "TELEGRAM_TOKEN жок!"
        )

    Thread(
        target=run_health_server,
        daemon=True,
    ).start()

    app = (
        Application
        .builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    app.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    app.add_handler(
        CommandHandler(
            "memory",
            memory_command,
        )
    )

    app.add_handler(
        CommandHandler(
            "clearmemory",
            clear_memory_command,
        )
    )

    app.add_handler(
        CommandHandler(
            "testapi",
            test_api,
        )
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    print(
        "🤖 AIAssist2026 иштеп жатат!"
    )

    print(
        "🧠 Automatic Memory ENABLED"
    )

    print(
        "⭐ Important Memory ENABLED"
    )

    app.run_polling(
        drop_pending_updates=True
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

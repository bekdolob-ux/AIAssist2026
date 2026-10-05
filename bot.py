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
    filters
)

from groq import Groq
from openai import OpenAI
from google import genai
from google.cloud import firestore
from google.oauth2 import service_account


# =========================
# ENV
# =========================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")

FIRESTORE_DATABASE = "ai-studio-e7d6a6f3-8921-414c-887e-4b97383c7bed"

GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3.8-flash"
OPENAI_MODEL = "gpt-5-mini"


# =========================
# API CLIENTS
# =========================

groq_client = None
gemini_client = None
openai_client = None
db = None


if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# FIRESTORE
# =========================

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
        database=FIRESTORE_DATABASE
    )

    print("✅ FIRESTORE CLIENT READY")

except Exception as e:
    print("❌ FIRESTORE INIT ERROR:", repr(e))
    db = None


# =========================
# MEMORY FUNCTIONS
# =========================

def get_user_ref(user_id):

    if db is None:
        return None

    return db.collection("users").document(str(user_id))


def load_memory(user_id):

    try:

        ref = get_user_ref(user_id)

        if ref is None:
            return []

        doc = ref.get()

        if not doc.exists:
            return []

        data = doc.to_dict() or {}

        return data.get("history", [])

    except Exception as e:

        print("❌ MEMORY LOAD ERROR:", repr(e))

        return []


def save_memory(user_id, user_text, ai_text):

    try:

        ref = get_user_ref(user_id)

        if ref is None:

            print("❌ MEMORY SAVE: Firestore unavailable")

            return False

        doc = ref.get()

        if doc.exists:

            data = doc.to_dict() or {}

            history = data.get("history", [])

        else:

            history = []

        history.append({

            "user": str(user_text),

            "assistant": str(ai_text)

        })

        # Акыркы 50 диалог сакталат
        history = history[-50:]

        ref.set({

            "telegram_user_id": str(user_id),

            "history": history,

            "updated": firestore.SERVER_TIMESTAMP

        })

        print(
            f"🧠 MEMORY SAVED: user={user_id}, "
            f"dialogs={len(history)}"
        )

        return True

    except Exception as e:

        print("❌ MEMORY SAVE ERROR:", repr(e))

        return False


def clear_memory(user_id):

    try:

        ref = get_user_ref(user_id)

        if ref is None:
            return False

        ref.set({

            "telegram_user_id": str(user_id),

            "history": [],

            "updated": firestore.SERVER_TIMESTAMP

        })

        print(f"🧹 MEMORY CLEARED: {user_id}")

        return True

    except Exception as e:

        print("❌ MEMORY CLEAR ERROR:", repr(e))

        return False


# =========================
# AI FUNCTIONS
# =========================

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
                        "Негизги максат — анын жеке кирешесин көбөйтүүгө "
                        "жардам берүү. Практикалык жана түшүнүктүү жооп бер."
                    )

                },

                {

                    "role": "user",

                    "content": prompt

                }

            ],

            temperature=0.7

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

            contents=prompt

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
            )

        )

        return response.output_text

    except Exception as e:

        print("❌ OPENAI ERROR:", repr(e))

        return None


# =========================
# AI ROUTER
# =========================

def ask_ai(prompt):

    # 1. Groq

    answer = ask_groq(prompt)

    if answer:

        print("🔵 Groq → OK")

        return answer


    # 2. Gemini

    answer = ask_gemini(prompt)

    if answer:

        print("🟢 Gemini → OK")

        return answer


    # 3. OpenAI

    answer = ask_openai(prompt)

    if answer:

        print("🟣 OpenAI → OK")

        return answer


    return "❌ Азыр AI кызматтарынын баары жооп берген жок."


# =========================
# MESSAGE
# =========================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message or not update.message.text:

        return

    user_id = update.effective_user.id

    user_text = update.message.text.strip()

    print(
        f"📩 USER {user_id}: {user_text}"
    )


    # =========================
    # ЭСКИ MEMORY
    # =========================

    history = load_memory(user_id)

    memory_text = ""

    if history:

        recent = history[-10:]

        memory_text = (
            "\n\nМурунку сүйлөшүүдөн контекст:\n"
        )

        for item in recent:

            memory_text += (

                f"Колдонуучу: "
                f"{item.get('user', '')}\n"

                f"AI: "
                f"{item.get('assistant', '')}\n"

            )


    # =========================
    # PROMPT
    # =========================

    prompt = f"""

{memory_text}

Жаңы билдирүү:

{user_text}

Колдонуучуга түз жооп бер.

"""


    # =========================
    # AI
    # =========================

    answer = await asyncio.to_thread(

        ask_ai,

        prompt

    )


    # =========================
    # TELEGRAM RESPONSE
    # =========================

    if not answer:

        answer = (
            "❌ AI бош жооп кайтарды."
        )


    # Telegram бир билдирүүгө чектөө коёт.
    # Узун жооп автоматтык түрдө бөлүнөт.

    MAX_MESSAGE_LENGTH = 4000

    for i in range(
        0,
        len(answer),
        MAX_MESSAGE_LENGTH
    ):

        chunk = answer[
            i:i + MAX_MESSAGE_LENGTH
        ]

        await update.message.reply_text(
            chunk
        )


    # =========================
    # AUTOMATIC MEMORY
    # =========================

    saved = await asyncio.to_thread(

        save_memory,

        user_id,

        user_text,

        answer

    )


    if saved:

        print(
            "🧠 Memory автоматтык сакталды"
        )

    else:

        print(
            "⚠️ Memory сакталган жок"
        )


# =========================
# /START
# =========================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(

        "🤖 Салам, Бекболот!\n\n"

        "Мен сенин жеке AI жардамчыңмын.\n"

        "🧠 Сүйлөшүүлөрдү эске сактайм.\n"

        "💰 Негизги максат — жеке кирешеңди көбөйтүү.\n\n"

        "/memory — эсте сакталган маалымат\n"

        "/clearmemory — Memory тазалоо\n"

        "/testapi — системаларды текшерүү"

    )


# =========================
# /MEMORY
# =========================

async def memory_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = update.effective_user.id

    history = await asyncio.to_thread(

        load_memory,

        user_id

    )


    if not history:

        await update.message.reply_text(

            "🧠 Memory азырынча бош."

        )

        return


    await update.message.reply_text(

        f"🧠 Memory иштеп жатат!\n\n"

        f"💾 Сакталган диалогдор: "
        f"{len(history)}\n"

        f"👤 User ID: "
        f"{user_id}"

    )


# =========================
# /CLEARMEMORY
# =========================

async def clear_memory_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = update.effective_user.id

    success = await asyncio.to_thread(

        clear_memory,

        user_id

    )


    if success:

        await update.message.reply_text(

            "🧹 Memory тазаланды."

        )

    else:

        await update.message.reply_text(

            "❌ Memory тазаланган жок."

        )


# =========================
# /TESTAPI
# =========================

async def test_api(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    msg = await update.message.reply_text(

        "🔍 API TEST башталды..."

    )


    # =========================
    # GROQ
    # =========================

    groq_ok = False

    try:

        result = await asyncio.to_thread(

            ask_groq,

            "Жөн гана OK деп жооп бер."

        )

        groq_ok = bool(result)

    except Exception:

        pass


    # =========================
    # GEMINI
    # =========================

    gemini_ok = False

    try:

        result = await asyncio.to_thread(

            ask_gemini,

            "Жөн гана OK деп жооп бер."

        )

        gemini_ok = bool(result)

    except Exception:

        pass


    # =========================
    # OPENAI
    # =========================

    openai_ok = False

    try:

        result = await asyncio.to_thread(

            ask_openai,

            "Жөн гана OK деп жооп бер."

        )

        openai_ok = bool(result)

    except Exception:

        pass


    # =========================
    # FIRESTORE
    # =========================

    firestore_ok = False

    try:

        if db is not None:

            test_ref = db.collection(

                "_system_test"

            ).document("test")


            test_ref.set({

                "status": "OK",

                "timestamp":
                    firestore.SERVER_TIMESTAMP

            })


            firestore_ok = True

    except Exception as e:

        print(
            "❌ FIRESTORE TEST ERROR:",
            repr(e)
        )


    # =========================
    # RESULT
    # =========================

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


# =========================
# HEALTH SERVER
# =========================

class HealthHandler(
    BaseHTTPRequestHandler
):

    def do_GET(self):

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/plain"
        )

        self.end_headers()

        self.wfile.write(
            b"AIAssist2026 is running!"
        )


    def log_message(
        self,
        format,
        *args
    ):

        return


def run_health_server():

    port = int(
        os.getenv(
            "PORT",
            "10000"
        )
    )


    server = HTTPServer(

        ("0.0.0.0", port),

        HealthHandler

    )


    print(
        f"🌐 Health server running "
        f"on port {port}"
    )


    server.serve_forever()


# =========================
# MAIN
# =========================

def main():

    if not TELEGRAM_TOKEN:

        raise Exception(
            "TELEGRAM_TOKEN жок!"
        )


    # Render health server

    Thread(

        target=run_health_server,

        daemon=True

    ).start()


    # Telegram application

    app = Application.builder().token(

        TELEGRAM_TOKEN

    ).build()


    # Commands

    app.add_handler(

        CommandHandler(
            "start",
            start
        )

    )


    app.add_handler(

        CommandHandler(
            "memory",
            memory_command
        )

    )


    app.add_handler(

        CommandHandler(
            "clearmemory",
            clear_memory_command
        )

    )


    app.add_handler(

        CommandHandler(
            "testapi",
            test_api
        )

    )


    # Normal messages

    app.add_handler(

        MessageHandler(

            filters.TEXT
            & ~filters.COMMAND,

            handle_message

        )

    )


    print(
        "🤖 AIAssist2026 иштеп жатат!"
    )

    print(
        "🧠 Automatic Memory ENABLED"
    )


    app.run_polling(

        drop_pending_updates=True

    )


# =========================
# RUN
# =========================

if __name__ == "__main__":

    main()

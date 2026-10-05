import os
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from groq import Groq
from google import genai
from openai import OpenAI

from google.cloud import firestore
from google.oauth2 import service_account


# =========================================================
# CONFIG
# =========================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")

GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3.8-flash"
OPENAI_MODEL = "gpt-5-mini"

FIRESTORE_DATABASE = "ai-studio-e7d6a6f3-8921-414c-887e-4b97383c7bed"


# =========================================================
# AI CLIENTS
# =========================================================

groq_client = None
gemini_client = None
openai_client = None

if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print("Groq init error:", e)

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print("Gemini init error:", e)

if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("OpenAI init error:", e)


# =========================================================
# FIRESTORE
# =========================================================

db = None

try:
    if GOOGLE_CREDENTIALS_JSON:
        credentials_info = json.loads(GOOGLE_CREDENTIALS_JSON)

        credentials = service_account.Credentials.from_service_account_info(
            credentials_info
        )

        db = firestore.Client(
            project=credentials_info.get("project_id"),
            credentials=credentials,
            database=FIRESTORE_DATABASE,
        )

        print("✅ Firestore connected")

    else:
        print("❌ GOOGLE_CREDENTIALS_JSON is missing")

except Exception as e:
    print("❌ Firestore connection error:", e)


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
Сен Бекболоттун жеке AI жардамчысы — AIAssist2026.

Негизги максат:
Бекболоттун жеке кирешесин көбөйтүүгө, кесиптик өсүшүнө,
бизнес идеяларына, программалоосуна жана жашоосун системалаштырууга жардам берүү.

Сен:
- кыргызча биринчи жооп бересиң;
- керек болсо орусча же англисче жооп бере аласың;
- мурунку маанилүү маалыматтарды эске аласың;
- жаңы пайдалуу маалыматты Memory'ге сактайсың;
- факт менен божомолду айырмалайсың;
- реалдуу жана практикалык кеңеш бересиң;
- керексиз узун жооп бербейсиң;
- кадам-кадам көрсөтмө берсең, жөнөкөй түшүндүрөсүң.

Маанилүү:
Колдонуучунун сырсөздөрүн, API key'лерин жана башка жашыруун
credential маалыматтарын Memory'ге сактаба.
"""


# =========================================================
# FIRESTORE MEMORY
# =========================================================

def get_memory(user_id):
    if db is None:
        return {}

    try:
        doc_ref = db.collection("users").document(str(user_id))
        doc = doc_ref.get()

        if doc.exists:
            return doc.to_dict()

        return {}

    except Exception as e:
        print("Memory read error:", e)
        return {}


def save_memory(user_id, memory):
    if db is None:
        return False

    try:
        doc_ref = db.collection("users").document(str(user_id))

        doc_ref.set(
            memory,
            merge=True
        )

        return True

    except Exception as e:
        print("Memory save error:", e)
        return False


def add_memory(user_id, user_text, assistant_text):
    memory = get_memory(user_id)

    if "history" not in memory:
        memory["history"] = []

    history = memory["history"]

    history.append({
        "user": user_text,
        "assistant": assistant_text
    })

    # Акыркы 30 диалогду гана сактайбыз
    memory["history"] = history[-30:]

    save_memory(user_id, memory)


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(user_id):

    memory = get_memory(user_id)

    if not memory:
        return ""

    history = memory.get("history", [])

    if not history:
        return ""

    context = "\n\nМурунку сүйлөшүүлөрдөн маанилүү маалымат:\n"

    for item in history[-15:]:
        context += f"""
Колдонуучу: {item.get("user", "")}
AI: {item.get("assistant", "")}
"""

    return context


# =========================================================
# GROQ
# =========================================================

def ask_groq(prompt):

    if groq_client is None:
        return None

    try:

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1500
        )

        if response and response.choices:

            text = response.choices[0].message.content

            if text and text.strip():
                return text.strip()

    except Exception as e:
        print("Groq error:", e)

    return None


# =========================================================
# GEMINI
# =========================================================

def ask_gemini(prompt):

    if gemini_client is None:
        return None

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=SYSTEM_PROMPT + "\n\n" + prompt
        )

        if response:

            text = getattr(response, "text", None)

            if text and text.strip():
                return text.strip()

    except Exception as e:
        print("Gemini error:", e)

    return None


# =========================================================
# OPENAI
# =========================================================

def ask_openai(prompt):

    if openai_client is None:
        return None

    try:

        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            instructions=SYSTEM_PROMPT,
            input=prompt,
            max_output_tokens=1500
        )

        text = getattr(response, "output_text", None)

        if text and text.strip():
            return text.strip()

    except Exception as e:
        print("OpenAI error:", e)

    return None


# =========================================================
# SMART AI ROUTER
# =========================================================

def smart_ai_router(prompt):

    # 1. GROQ
    try:
        answer = ask_groq(prompt)

        if answer:
            print("✅ AI: Groq")
            return answer, "Groq"

    except Exception as e:
        print("Groq router error:", e)


    # 2. GEMINI
    try:
        answer = ask_gemini(prompt)

        if answer:
            print("✅ AI: Gemini")
            return answer, "Gemini"

    except Exception as e:
        print("Gemini router error:", e)


    # 3. OPENAI
    try:
        answer = ask_openai(prompt)

        if answer:
            print("✅ AI: OpenAI")
            return answer, "OpenAI"

    except Exception as e:
        print("OpenAI router error:", e)


    return (
        "❌ Азыр үч AI кызматында тең жооп алуу мүмкүн болгон жок. "
        "Бир аздан кийин кайра аракет кыл.",
        None
    )


# =========================================================
# START
# =========================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user = update.effective_user

    await update.message.reply_text(
        f"""
🤖 AIAssist2026

Салам, {user.first_name or "Бекболот"}!

Мен сенин жеке AI жардамчыңмын.

🎯 Негизги максат:
Жеке кирешеңди көбөйтүү.

Мен сага жардам бере алам:

💰 Киреше
💼 Жумуш
🚀 Бизнес
📱 Онлайн киреше
💻 Программалоо
📊 Финансы
🧠 Өнүгүү
🗂 Memory

Жөн гана сурооңду жаз.
"""
    )


# =========================================================
# MEMORY COMMAND
# =========================================================

async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.effective_user.id

    memory = get_memory(user_id)

    history = memory.get("history", [])

    if not history:

        await update.message.reply_text(
            "🧠 Memory азырынча бош."
        )

        return

    await update.message.reply_text(
        f"""
🧠 MEMORY

Сакталган диалогдор:
{len(history)}

Memory Firestore'до сакталат.
Render кайра иштесе да маалымат өчпөйт.
"""
    )


# =========================================================
# CLEAR MEMORY
# =========================================================

async def clear_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.effective_user.id

    if db is None:

        await update.message.reply_text(
            "❌ Firestore туташкан эмес."
        )

        return

    try:

        db.collection("users").document(
            str(user_id)
        ).set(
            {
                "history": []
            },
            merge=True
        )

        await update.message.reply_text(
            "🧹 Memory тазаланды."
        )

    except Exception as e:

        print("Clear memory error:", e)

        await update.message.reply_text(
            "❌ Memory тазалоодо ката кетти."
        )


# =========================================================
# TEST API
# =========================================================

async def test_api(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        "🔍 API TEST башталды..."
    )

    results = []

    # GROQ
    groq_result = ask_groq("Жооп бер: OK")

    if groq_result:
        results.append("🔵 Groq: 🟢 OK")
    else:
        results.append("🔵 Groq: 🔴 ERROR")


    # GEMINI
    gemini_result = ask_gemini("Жооп бер: OK")

    if gemini_result:
        results.append("🟢 Gemini: 🟢 OK")
    else:
        results.append("🟢 Gemini: 🔴 ERROR")


    # OPENAI
    openai_result = ask_openai("Жооп бер: OK")

    if openai_result:
        results.append("🟣 OpenAI: 🟢 OK")
    else:
        results.append("🟣 OpenAI: 🔴 ERROR")


    firestore_status = (
        "🟡 Firestore: 🟢 CONNECTED"
        if db is not None
        else
        "🟡 Firestore: 🔴 ERROR"
    )

    results.append(firestore_status)

    await update.message.reply_text(
        "🔍 API TEST\n\n" +
        "\n".join(results)
    )


# =========================================================
# MESSAGE HANDLER
# =========================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not update.message:
        return

    user_id = update.effective_user.id

    user_text = update.message.text.strip()

    if not user_text:
        return

    # Memory context
    memory_context = build_context(user_id)

    prompt = f"""
{memory_context}

Жаңы билдирүү:

{user_text}

Колдонуучуга түз жана пайдалуу жооп бер.
"""


    await update.message.chat.send_action(
        action="typing"
    )

    answer, provider = smart_ai_router(prompt)

    if not answer:
        answer = "❌ AI жооп бере алган жок."


    # Memory save
    if provider:

        try:
            add_memory(
                user_id,
                user_text,
                answer
            )

        except Exception as e:

            print("Memory save failed:", e)


    # AI provider көрүнүп турсун
    if provider:
        answer = f"{answer}\n\n— {provider}"

    await update.message.reply_text(
        answer
    )


# =========================================================
# HEALTH SERVER FOR RENDER
# =========================================================

class HealthHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        self.send_response(200)

        self.send_header(
            "Content-type",
            "text/plain"
        )

        self.end_headers()

        self.wfile.write(
            b"AIAssist2026 is running!"
        )

    def log_message(self, format, *args):
        return


def run_health_server():

    port = int(
        os.environ.get(
            "PORT",
            "10000"
        )
    )

    server = HTTPServer(
        ("0.0.0.0", port),
        HealthHandler
    )

    print(
        f"🌐 Health server running on port {port}"
    )

    server.serve_forever()


# =========================================================
# MAIN
# =========================================================

def main():

    if not TELEGRAM_TOKEN:

        print(
            "❌ TELEGRAM_TOKEN жок!"
        )

        return


    threading.Thread(
        target=run_health_server,
        daemon=True
    ).start()


    print(
        "🤖 Telegram AI бот иштеп жатат..."
    )


    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )


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
            clear_memory
        )
    )

    app.add_handler(
        CommandHandler(
            "testapi",
            test_api
        )
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )


    app.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()

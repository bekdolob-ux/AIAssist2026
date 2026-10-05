import asyncio
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from google import genai
from groq import Groq
from openai import OpenAI

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)


# ==================================================
# 🔐 ENVIRONMENT VARIABLES
# ==================================================

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


# ==================================================
# 🤖 MODELS
# ==================================================

GEMINI_MODEL = "gemini-3.8-flash"

GROQ_MODEL = "openai/gpt-oss-120b"

OPENAI_MODEL = "gpt-5-mini"


# ==================================================
# 🤖 CLIENTS
# ==================================================

gemini = genai.Client(
    api_key=GEMINI_API_KEY
)

groq = Groq(
    api_key=GROQ_API_KEY
)

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ==================================================
# 🧠 SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = """
Сен күчтүү жеке AI жардамчысың.

Колдонуучу кайсы тилде жазса,
ошол тилде жооп бер.

Кыргызча суроого кыргызча жооп бер.
Орусча суроого орусча жооп бер.
Англисче суроого англисче жооп бер.

Татаал суроолордо:
- конкреттүү кадамдарды бер
- сандарды колдон
- мисал келтир
- артыкчылык тартибин көрсөт
- тобокелдиктерди айт

Жөн гана жалпы кеңеш менен чектелбе.
Пайдалуу жана практикалык жооп бер.

Негизги максат:
колдонуучуга кирешесин көбөйтүүгө,
кесиптик өсүүгө жана практикалык иштерди
аткарууга жардам берүү.
"""


# ==================================================
# 🌐 RENDER WEB SERVER
# ==================================================

class HealthHandler(BaseHTTPRequestHandler):

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

    def log_message(self, format, *args):
        return


def start_web_server():

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    server = HTTPServer(
        ("0.0.0.0", port),
        HealthHandler
    )

    print(
        f"🌐 Web server PORT: {port}"
    )

    server.serve_forever()


# ==================================================
# 📩 TELEGRAM LONG MESSAGE
# ==================================================

async def send_long_message(
    update: Update,
    text: str
):

    limit = 3900

    if len(text) <= limit:

        await update.message.reply_text(
            text
        )

        return

    while len(text) > limit:

        cut = text.rfind(
            "\n",
            0,
            limit
        )

        if cut < 1000:
            cut = limit

        part = text[:cut]

        text = text[cut:].lstrip()

        await update.message.reply_text(
            part
        )

        await asyncio.sleep(0.3)

    if text:

        await update.message.reply_text(
            text
        )


# ==================================================
# 🔵 GROQ
# ==================================================

def ask_groq(text):

    response = groq.chat.completions.create(

        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": text
            }
        ],

        temperature=0.7,

        max_tokens=8000
    )

    if not response.choices:

        raise Exception(
            "Groq choices бош"
        )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    if not answer:

        raise Exception(
            "Groq бош жооп берди"
        )

    return answer


# ==================================================
# 🟢 GEMINI
# ==================================================

def ask_gemini(text):

    response = gemini.models.generate_content(

        model=GEMINI_MODEL,

        contents=(
            SYSTEM_PROMPT
            + "\n\n"
            + text
        )
    )

    if not response.text:

        raise Exception(
            "Gemini бош жооп берди"
        )

    return response.text


# ==================================================
# 🟣 OPENAI
# ==================================================

def ask_openai(text):

    response = (
        openai_client
        .chat
        .completions
        .create(

            model=OPENAI_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": text
                }
            ],

            max_completion_tokens=8000
        )
    )

    if not response.choices:

        raise Exception(
            "OpenAI choices бош"
        )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    if not answer:

        raise Exception(
            "OpenAI бош жооп берди"
        )

    return answer


# ==================================================
# 🚀 /START
# ==================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(

        "🤖 AIAssist2026\n\n"

        "Салам! Мен сенин жеке AI "
        "жардамчыңмын.\n\n"

        "🔵 Groq\n"
        "🟢 Gemini\n"
        "🟣 OpenAI\n\n"

        "Үч AI системасы туташкан.\n\n"

        "Сурооңду жаза бер.\n\n"

        "/testapi — API'лерди текшерүү"
    )


# ==================================================
# 💬 CHAT
# ==================================================

async def chat(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_text = update.message.text

    waiting = await update.message.reply_text(
        "🧠 AI ойлонуп жатат..."
    )


    # ==================================================
    # 1️⃣ GROQ
    # ==================================================

    try:

        answer = await asyncio.to_thread(
            ask_groq,
            user_text
        )

        await waiting.delete()

        await send_long_message(
            update,
            "🔵 Groq\n\n" + answer
        )

        return

    except Exception as e:

        print("\n❌ GROQ ERROR:")
        print(repr(e))

        await waiting.edit_text(
            "⚠️ Groq жооп бере алган жок.\n"
            "🔄 Gemini текшерилип жатат..."
        )


    # ==================================================
    # 2️⃣ GEMINI
    # ==================================================

    try:

        answer = await asyncio.to_thread(
            ask_gemini,
            user_text
        )

        await waiting.delete()

        await send_long_message(
            update,
            "🟢 Gemini\n\n" + answer
        )

        return

    except Exception as e:

        print("\n❌ GEMINI ERROR:")
        print(repr(e))

        await waiting.edit_text(
            "⚠️ Gemini жооп бере алган жок.\n"
            "🔄 OpenAI текшерилип жатат..."
        )


    # ==================================================
    # 3️⃣ OPENAI
    # ==================================================

    try:

        answer = await asyncio.to_thread(
            ask_openai,
            user_text
        )

        await waiting.delete()

        await send_long_message(
            update,
            "🟣 OpenAI\n\n" + answer
        )

        return

    except Exception as e:

        print("\n❌ OPENAI ERROR:")
        print(repr(e))

        await waiting.edit_text(
            "❌ Үч AI тең жооп бере алган жок.\n\n"
            "API'лерди /testapi менен текшер."
        )


# ==================================================
# 🔍 /TESTAPI
# ==================================================

async def test_api(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    waiting = await update.message.reply_text(
        "🔍 Үч API текшерилип жатат..."
    )

    groq_ok = False
    gemini_ok = False
    openai_ok = False


    # ==================================================
    # GROQ TEST
    # ==================================================

    try:

        await asyncio.to_thread(
            ask_groq,
            "Бир сөз менен жооп бер: ИШТЕДИ"
        )

        groq_ok = True

        print(
            "\n🔵 GROQ TEST: OK"
        )

    except Exception as e:

        print(
            "\n❌ GROQ TEST ERROR:"
        )

        print(
            repr(e)
        )


    # ==================================================
    # GEMINI TEST
    # ==================================================

    try:

        await asyncio.to_thread(
            ask_gemini,
            "Бир сөз менен жооп бер: ИШТЕДИ"
        )

        gemini_ok = True

        print(
            "\n🟢 GEMINI TEST: OK"
        )

    except Exception as e:

        print(
            "\n❌ GEMINI TEST ERROR:"
        )

        print(
            repr(e)
        )


    # ==================================================
    # OPENAI TEST
    # ==================================================

    try:

        await asyncio.to_thread(
            ask_openai,
            "Бир сөз менен жооп бер: ИШТЕДИ"
        )

        openai_ok = True

        print(
            "\n🟣 OPENAI TEST: OK"
        )

    except Exception as e:

        print(
            "\n❌ OPENAI TEST ERROR:"
        )

        print(
            repr(e)
        )


    # ==================================================
    # RESULT
    # ==================================================

    result = (

        "🔍 API TEST ЖЫЙЫНТЫГЫ\n\n"

        f"🔵 Groq: "
        f"{'🟢 ИШТЕДИ' if groq_ok else '🔴 ERROR'}\n\n"

        f"🟢 Gemini: "
        f"{'🟢 ИШТЕДИ' if gemini_ok else '🔴 ERROR'}\n\n"

        f"🟣 OpenAI: "
        f"{'🟢 ИШТЕДИ' if openai_ok else '🔴 ERROR'}"
    )

    await waiting.edit_text(
        result
    )


# ==================================================
# 🚀 MAIN
# ==================================================

def main():

    # Render health server
    web_thread = threading.Thread(
        target=start_web_server,
        daemon=True
    )

    web_thread.start()


    # Telegram bot
    app = (
        Application
        .builder()
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
            "testapi",
            test_api
        )
    )


    app.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            chat
        )
    )


    print("")
    print("======================================")
    print("🤖 AIAssist2026 БОТ ИШТЕП ЖАТАТ")
    print("======================================")
    print("🔵 Groq:", GROQ_MODEL)
    print("🟢 Gemini:", GEMINI_MODEL)
    print("🟣 OpenAI:", OPENAI_MODEL)
    print("======================================")


    app.run_polling()


# ==================================================
# ▶️ START
# ==================================================

if __name__ == "__main__":

    main()

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


# =========================================================
# CONFIG
# =========================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3.8-flash"
OPENAI_MODEL = "gpt-5-mini"

MEMORY_FILE = "memory.json"


# =========================================================
# CLIENTS
# =========================================================

groq_client = Groq(api_key=GROQ_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# =========================================================
# MEMORY
# =========================================================

def load_memory():
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print("Memory load error:", e)

    return {}


def save_memory(memory):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(
                memory,
                f,
                ensure_ascii=False,
                indent=2
            )
    except Exception as e:
        print("Memory save error:", e)


memory = load_memory()


def get_user_memory(user_id):
    user_id = str(user_id)

    if user_id not in memory:
        memory[user_id] = {
            "facts": [],
            "goals": [],
            "knowledge": [],
            "experience": [],
            "history": []
        }

    user = memory[user_id]

    user.setdefault("facts", [])
    user.setdefault("goals", [])
    user.setdefault("knowledge", [])
    user.setdefault("experience", [])
    user.setdefault("history", [])

    return user


def memory_text(user_id):

    user = get_user_memory(user_id)

    result = ""

    if user["facts"]:
        result += "\n👤 МААНИЛҮҮ МААЛЫМАТ:\n"

        for item in user["facts"][-30:]:
            result += "- " + item + "\n"

    if user["goals"]:
        result += "\n🎯 МАКСАТТАР:\n"

        for item in user["goals"][-20:]:
            result += "- " + item + "\n"

    if user["knowledge"]:
        result += "\n📚 БИЛИМ:\n"

        for item in user["knowledge"][-20:]:
            result += "- " + item + "\n"

    if user["experience"]:
        result += "\n📈 ТАЖРЫЙБА:\n"

        for item in user["experience"][-20:]:
            result += "- " + item + "\n"

    if user["history"]:
        result += "\n💬 АКЫРКЫ СҮЙЛӨШҮҮЛӨР:\n"

        for item in user["history"][-5:]:

            result += (
                "Колдонуучу: "
                + item["user"]
                + "\n"
            )

            result += (
                "AI: "
                + item["assistant"][:1000]
                + "\n\n"
            )

    return result


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
Сен AIAssist2026 — колдонуучунун жеке AI жардамчысысың.

Максат:
Колдонуучуга мүмкүн болушунча пайдалуу, реалдуу жана практикалык жардам берүү.

Memory маалыматтарын контекст катары колдон.

Бирок:
- Memory'де жок нерсени ойлоп таппа.
- Белгисиз маалыматты факт катары айтпа.
- Сандык көрсөткүчтөрдү негизсиз ойлоп чыгарба.
- Кирешеге кепилдик бербе.
- Кыргызстандагы реалдуу шарттарды эске ал.
- Колдонуучунун мурдагы максаттарын жана тажрыйбасын эске ал.
- Эгер маалымат эски болушу мүмкүн болсо, аны этият колдон.

Бизнес жана киреше боюнча:
- биринчи кардарды табууга көңүл бур;
- чоң чыгымдан мурда тест кыл;
- чыгым менен таза кирешени айырмала;
- реалдуу кадамдарды сунушта.

Жооп:
- түшүнүктүү;
- практикалык;
- керексиз узун эмес;
- керек болсо кадам-кадам.
"""


# =========================================================
# ERROR DETECTION
# =========================================================

def is_retryable_error(error):

    error_text = str(error).lower()

    retry_words = [
        "429",
        "rate limit",
        "ratelimit",
        "too many requests",
        "quota",
        "503",
        "unavailable",
        "service unavailable",
        "timeout",
        "timed out",
        "temporarily",
        "overloaded",
        "server error"
    ]

    return any(
        word in error_text
        for word in retry_words
    )


# =========================================================
# GROQ
# =========================================================

def ask_groq(user_id, user_text):

    prompt = (
        SYSTEM_PROMPT
        + "\n"
        + memory_text(user_id)
    )

    try:

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ],
            max_tokens=5000
        )

        answer = response.choices[0].message.content

        if not answer:
            raise Exception("Groq бош жооп кайтарды")

        return answer

    except Exception as e:

        print("❌ GROQ ERROR:", e)

        raise


# =========================================================
# GEMINI
# =========================================================

def ask_gemini(user_id, user_text):

    prompt = (
        SYSTEM_PROMPT
        + "\n"
        + memory_text(user_id)
        + "\n\nКолдонуучу:\n"
        + user_text
    )

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        if not response.text:
            raise Exception(
                "Gemini бош жооп кайтарды"
            )

        return response.text

    except Exception as e:

        print("❌ GEMINI ERROR:", e)

        raise


# =========================================================
# OPENAI
# =========================================================

def ask_openai(user_id, user_text):

    prompt = (
        SYSTEM_PROMPT
        + "\n"
        + memory_text(user_id)
    )

    try:

        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ],
            max_completion_tokens=5000
        )

        answer = response.choices[0].message.content

        if not answer:
            raise Exception(
                "OpenAI бош жооп кайтарды"
            )

        return answer

    except Exception as e:

        print("❌ OPENAI ERROR:", e)

        raise


# =========================================================
# SMART ROUTER
# =========================================================

def smart_ai_router(user_id, user_text):

    # -----------------------------------------------------
    # 1. GROQ
    # -----------------------------------------------------

    try:

        answer = ask_groq(
            user_id,
            user_text
        )

        return "🔵 Groq", answer

    except Exception as e:

        print(
            "➡️ Groq иштеген жок. Geminiге өтөбүз."
        )


    # -----------------------------------------------------
    # 2. GEMINI
    # -----------------------------------------------------

    try:

        answer = ask_gemini(
            user_id,
            user_text
        )

        return "🟢 Gemini", answer

    except Exception as e:

        print(
            "➡️ Gemini иштеген жок. OpenAIге өтөбүз."
        )


    # -----------------------------------------------------
    # 3. OPENAI
    # -----------------------------------------------------

    try:

        answer = ask_openai(
            user_id,
            user_text
        )

        return "🟣 OpenAI", answer

    except Exception as e:

        print(
            "❌ ҮЧ AI ТЕҢ ИШТЕГЕН ЖОК."
        )


    return (
        "❌ AI кызматтарынын үчөө тең "
        "учурда жеткиликсиз.\n\n"
        "Бир аздан кийин кайра аракет кыл."
    ), None


# =========================================================
# MEMORY HISTORY
# =========================================================

def add_history(
    user_id,
    user_text,
    answer,
    ai_name
):

    user = get_user_memory(user_id)

    user["history"].append({
        "user": user_text,
        "assistant": answer,
        "ai": ai_name
    })

    user["history"] = user["history"][-30:]

    save_memory(memory)


# =========================================================
# SIMPLE AUTO MEMORY
# =========================================================

def simple_auto_memory(
    user_id,
    user_text
):

    """
    Кээ бир ачык айтылган маанилүү маалыматтарды
    автоматтык түрдө Memory'ге сактайт.

    Бул жерде кошумча API чакырбайбыз.
    Ошондуктан бекер лимиттерди жебейт.
    """

    user = get_user_memory(user_id)

    text = user_text.lower().strip()

    # Аты
    if (
        "менин атым" in text
        or "аты-жөнүм" in text
        or "мен бекболот" in text
    ):
        fact = user_text.strip()

        if fact not in user["facts"]:
            user["facts"].append(fact)

    # Максат
    goal_words = [
        "максатым",
        "максат —",
        "максат -",
        "каалайм",
        "максатым:"
    ]

    if any(
        word in text
        for word in goal_words
    ):
        if user_text not in user["goals"]:
            user["goals"].append(
                user_text.strip()
            )

    # Жөндөм / кесип
    skill_words = [
        "билем",
        "иштейм",
        "кесибим",
        "тажрыйбам",
        "тажрыйбам бар",
        "үйрөнгөм"
    ]

    if any(
        word in text
        for word in skill_words
    ):
        if user_text not in user["facts"]:
            user["facts"].append(
                user_text.strip()
            )

    user["facts"] = user["facts"][-50:]
    user["goals"] = user["goals"][-30:]

    save_memory(memory)


# =========================================================
# START
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🧠 AIAssist2026\n\n"
        "Мен сенин жеке AI жардамчыңмын.\n\n"
        "🤖 Smart AI Router:\n"
        "1️⃣ 🔵 Groq\n"
        "2️⃣ 🟢 Gemini\n"
        "3️⃣ 🟣 OpenAI\n\n"
        "Кайсы AI жеткиликтүү болсо, "
        "ошол автоматтык колдонулат.\n\n"
        "🧠 Memory иштейт."
    )


# =========================================================
# MEMORY COMMAND
# =========================================================

async def memory_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = update.effective_user.id

    user = get_user_memory(user_id)

    text = (
        "🧠 MEMORY\n\n"
        f"👤 Facts: {len(user['facts'])}\n"
        f"🎯 Goals: {len(user['goals'])}\n"
        f"📚 Knowledge: {len(user['knowledge'])}\n"
        f"📈 Experience: {len(user['experience'])}\n"
        f"💬 History: {len(user['history'])}\n"
    )

    if user["facts"]:

        text += "\n👤 Маалымат:\n"

        for item in user["facts"][-10:]:
            text += "• " + item + "\n"

    if user["goals"]:

        text += "\n🎯 Максаттар:\n"

        for item in user["goals"][-10:]:
            text += "• " + item + "\n"

    await update.message.reply_text(
        text[:4000]
    )


# =========================================================
# CLEAR MEMORY
# =========================================================

async def clear_memory(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = str(
        update.effective_user.id
    )

    memory[user_id] = {
        "facts": [],
        "goals": [],
        "knowledge": [],
        "experience": [],
        "history": []
    }

    save_memory(memory)

    await update.message.reply_text(
        "🧹 Memory толугу менен тазаланды."
    )


# =========================================================
# TEST API
# =========================================================

async def test_api(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    result = "🔍 API TEST\n\n"


    # GROQ
    try:

        groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Reply only: OK"
                }
            ],
            max_tokens=20
        )

        result += (
            "🔵 Groq: 🟢 ИШТЕДИ\n"
        )

    except Exception as e:

        result += (
            "🔵 Groq: 🔴 ERROR\n"
            + str(e)[:500]
            + "\n\n"
        )


    # GEMINI
    try:

        gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Reply only: OK"
        )

        result += (
            "🟢 Gemini: 🟢 ИШТЕДИ\n"
        )

    except Exception as e:

        result += (
            "🟢 Gemini: 🔴 ERROR\n"
            + str(e)[:500]
            + "\n\n"
        )


    # OPENAI
    try:

        openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Reply only: OK"
                }
            ],
            max_completion_tokens=20
        )

        result += (
            "🟣 OpenAI: 🟢 ИШТЕДИ\n"
        )

    except Exception as e:

        result += (
            "🟣 OpenAI: 🔴 ERROR\n"
            + str(e)[:500]
            + "\n"
        )


    await update.message.reply_text(
        result[:4000]
    )


# =========================================================
# MAIN MESSAGE
# =========================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = update.effective_user.id

    user_text = update.message.text


    # -----------------------------------------------------
    # MEMORY
    # -----------------------------------------------------

    simple_auto_memory(
        user_id,
        user_text
    )


    # -----------------------------------------------------
    # ROUTER
    # -----------------------------------------------------

    await update.message.reply_text(
        "🧠 AI текшерилип жатат...\n\n"
        "🔵 Groq → 🟢 Gemini → 🟣 OpenAI"
    )


    ai_name, answer = smart_ai_router(
        user_id,
        user_text
    )


    # Үчөө тең иштебесе
    if answer is None:

        await update.message.reply_text(
            ai_name
        )

        return


    # -----------------------------------------------------
    # HISTORY
    # -----------------------------------------------------

    add_history(
        user_id,
        user_text,
        answer,
        ai_name
    )


    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    final_text = (
        f"{ai_name}\n\n"
        + answer
    )

    await update.message.reply_text(
        final_text[:4000]
    )


# =========================================================
# RENDER WEB SERVER
# =========================================================

class HealthHandler(
    BaseHTTPRequestHandler
):

    def do_GET(self):

        self.send_response(200)

        self.send_header(
            "Content-type",
            "text/plain; charset=utf-8"
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


def run_server():

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
        f"🌐 Web server started on port {port}"
    )

    server.serve_forever()


# =========================================================
# MAIN
# =========================================================

def main():

    if not TELEGRAM_TOKEN:
        raise Exception(
            "TELEGRAM_TOKEN табылган жок!"
        )

    threading.Thread(
        target=run_server,
        daemon=True
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


    print(
        "🤖 AIAssist2026 иштеп жатат..."
    )


    app.run_polling()


if __name__ == "__main__":
    main()

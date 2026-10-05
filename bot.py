import asyncio
import os
import threading
import json
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

# =========================
# API KEYS
# =========================

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# =========================
# MODELS
# =========================

GEMINI_MODEL = "gemini-3.8-flash"
GROQ_MODEL = "openai/gpt-oss-120b"
OPENAI_MODEL = "gpt-5-mini"

gemini = genai.Client(api_key=GEMINI_API_KEY)
groq = Groq(api_key=GROQ_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# MEMORY
# =========================

MEMORY_FILE = "memory.json"


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
            json.dump(memory, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Memory save error:", e)


memory = load_memory()


def get_user_memory(user_id):
    user_id = str(user_id)

    if user_id not in memory:
        memory[user_id] = {
            "facts": [],
            "history": []
        }

    return memory[user_id]


def add_memory(user_id, user_text, answer):
    user = get_user_memory(user_id)

    # Акыркы сүйлөшүүнү сактайбыз
    user["history"].append({
        "user": user_text,
        "assistant": answer
    })

    # Өтө чоң болуп кетпеши үчүн акыркы 20 сүйлөшүү
    user["history"] = user["history"][-20:]

    save_memory(memory)


def memory_text(user_id):
    user = get_user_memory(user_id)

    facts = user.get("facts", [])
    history = user.get("history", [])

    result = ""

    if facts:
        result += "\nМААНИЛҮҮ ЭС ТУТУМ:\n"
        for fact in facts[-30:]:
            result += "- " + fact + "\n"

    if history:
        result += "\nАКЫРКЫ СҮЙЛӨШҮҮЛӨР:\n"

        for item in history[-5:]:
            result += (
                "Колдонуучу: "
                + item["user"]
                + "\n"
                + "AI: "
                + item["assistant"][:1000]
                + "\n\n"
            )

    return result


# =========================
# SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = """
Сен күчтүү жеке AI жардамчысың.

Колдонуучу кайсы тилде жазса,
ошол тилде жооп бер.

Кыргызча суроого кыргызча жооп бер.
Орусча суроого орусча жооп бер.
Англисче суроого англисче жооп бер.

Негизги максат:
колдонуучуга кирешесин көбөйтүүгө,
кесиптик өсүүгө, бизнеске,
программалоого жана практикалык иштерге
жардам берүү.

Маалыматты ойлоп чыгарба.

Эгер так маалымат жок болсо:
"Бул болжол" деп ачык айт.

Киреше боюнча:
- жүгүртүү менен таза пайданы айырмала
- кардар санын далилсиз жогору койбо
- чоң кирешени кепилдик катары көрсөтпө
- биринчи кардарды табууга басым жаса
- тобокелдиктерди көрсөт

Колдонуучунун Memory маалыматтарын
контекст катары колдон.

Бирок Memory'деги маалымат туура эмес
болушу мүмкүн болсо, аны факт катары кабыл алба.

Татаал суроолордо:
1. Анализ
2. Так кадамдар
3. Чыгым
4. Потенциалдуу пайда
5. Тобокелдик
6. Кийинки кадам

түрүндө жооп бер.
"""


# =========================
# WEB SERVER FOR RENDER
# =========================

class HealthHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(
            b"AIAssist2026 is running!"
        )

    def log_message(self, format, *args):
        return


def start_web_server():
    port = int(os.environ.get("PORT", 10000))

    server = HTTPServer(
        ("0.0.0.0", port),
        HealthHandler
    )

    print(f"🌐 Web server PORT: {port}")

    server.serve_forever()


# =========================
# GROQ
# =========================

def ask_groq(text, user_id):

    prompt = (
        SYSTEM_PROMPT
        + "\n"
        + memory_text(user_id)
        + "\n\nКОЛДОНУУЧУНУН ЖАҢЫ СУРООСУ:\n"
        + text
    )

    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": prompt
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
        raise Exception("Groq choices бош")

    answer = response.choices[0].message.content

    if not answer:
        raise Exception("Groq бош жооп берди")

    return answer


# =========================
# GEMINI
# =========================

def ask_gemini(text, user_id):

    prompt = (
        SYSTEM_PROMPT
        + "\n"
        + memory_text(user_id)
        + "\n\nКОЛДОНУУЧУНУН ЖАҢЫ СУРООСУ:\n"
        + text
    )

    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:
        raise Exception("Gemini бош жооп берди")

    return response.text


# =========================
# OPENAI
# =========================

def ask_openai(text, user_id):

    prompt = (
        SYSTEM_PROMPT
        + "\n"
        + memory_text(user_id)
        + "\n\nКОЛДОНУУЧУНУН ЖАҢЫ СУРООСУ:\n"
        + text
    )

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": text
            }
        ],
        max_completion_tokens=8000
    )

    if not response.choices:
        raise Exception("OpenAI choices бош")

    answer = response.choices[0].message.content

    if not answer:
        raise Exception("OpenAI бош жооп берди")

    return answer


# =========================
# FINAL AI
# =========================

def make_final_answer(user_text, groq_answer,
                      gemini_answer, openai_answer):

    prompt = f"""
Сен үч AI жоопторун анализдеп,
колдонуучуга БИР гана эң сапаттуу жооп
берген башкы AIсың.

Колдонуучунун суроосу:

{user_text}

GROQ ЖООБУ:
{groq_answer}

GEMINI ЖООБУ:
{gemini_answer}

OPENAI ЖООБУ:
{openai_answer}

Милдетиң:

1. Үч жоопту салыштыр.
2. Каталарды тап.
3. Реалдуу эмес киреше сандарын алып сал.
4. Ойлоп чыгарылган фактыларды колдонбо.
5. Эң пайдалуу маалыматтарды бириктир.
6. Колдонуучуга түшүнүктүү кыргызча жооп бер.
7. Керек болсо кадам-кадам көрсөт.
8. Үч AI жөнүндө узун түшүндүрмө бербе.
9. Акырында конкреттүү кийинки кадамды айт.

Жоопту түздөн-түз колдонуучуга бер.
"""

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
        max_completion_tokens=8000
    )

    if not response.choices:
        raise Exception("Final AI choices бош")

    answer = response.choices[0].message.content

    if not answer:
        raise Exception("Final AI бош жооп берди")

    return answer


# =========================
# LONG TELEGRAM MESSAGE
# =========================

async def send_long_message(update, text):

    limit = 3900

    if len(text) <= limit:
        await update.message.reply_text(text)
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

        await update.message.reply_text(part)

        await asyncio.sleep(0.3)

    if text:
        await update.message.reply_text(text)


# =========================
# START
# =========================

async def start(update, context):

    await update.message.reply_text(
        "🤖 AIAssist2026\n\n"
        "Салам! Мен сенин жеке AI жардамчыңмын.\n\n"

        "🧠 Memory — Эс тутум\n"
        "🔵 Groq\n"
        "🟢 Gemini\n"
        "🟣 OpenAI\n\n"

        "Үч AI жооп берип,\n"
        "андан кийин бириктирилип\n"
        "эң жакшы жооп түзүлөт.\n\n"

        "/memory — эс тутумду көрүү\n"
        "/clearmemory — эс тутумду тазалоо\n"
        "/testapi — API текшерүү\n\n"

        "Сурооңду жаза бер."
    )


# =========================
# MEMORY COMMAND
# =========================

async def show_memory(update, context):

    user_id = update.effective_user.id

    user = get_user_memory(user_id)

    facts = user.get("facts", [])
    history = user.get("history", [])

    await update.message.reply_text(
        "🧠 СЕНИН MEMORY\n\n"
        f"Маанилүү маалымат: {len(facts)}\n"
        f"Сүйлөшүү: {len(history)}\n\n"
        "Memory иштеп жатат."
    )


# =========================
# CLEAR MEMORY
# =========================

async def clear_memory(update, context):

    user_id = str(update.effective_user.id)

    memory[user_id] = {
        "facts": [],
        "history": []
    }

    save_memory(memory)

    await update.message.reply_text(
        "🗑 Memory тазаланды."
    )


# =========================
# CHAT
# =========================

async def chat(update, context):

    user_text = update.message.text
    user_id = update.effective_user.id

    waiting = await update.message.reply_text(
        "🧠 Үч AI иштеп жатат...\n\n"
        "🔵 Groq\n"
        "🟢 Gemini\n"
        "🟣 OpenAI"
    )

    groq_answer = ""
    gemini_answer = ""
    openai_answer = ""

    # =====================
    # GROQ
    # =====================

    try:

        groq_answer = await asyncio.to_thread(
            ask_groq,
            user_text,
            user_id
        )

        print("🔵 GROQ: OK")

    except Exception as e:

        print("❌ GROQ ERROR:", repr(e))


    # =====================
    # GEMINI
    # =====================

    try:

        gemini_answer = await asyncio.to_thread(
            ask_gemini,
            user_text,
            user_id
        )

        print("🟢 GEMINI: OK")

    except Exception as e:

        print("❌ GEMINI ERROR:", repr(e))


    # =====================
    # OPENAI
    # =====================

    try:

        openai_answer = await asyncio.to_thread(
            ask_openai,
            user_text,
            user_id
        )

        print("🟣 OPENAI: OK")

    except Exception as e:

        print("❌ OPENAI ERROR:", repr(e))


    # =====================
    # CHECK
    # =====================

    answers = [
        groq_answer,
        gemini_answer,
        openai_answer
    ]

    working_answers = [
        x for x in answers if x
    ]

    if not working_answers:

        await waiting.edit_text(
            "❌ Үч AI тең жооп бере алган жок.\n\n"
            "/testapi менен текшер."
        )

        return


    # =====================
    # FINAL ANSWER
    # =====================

    try:

        final_answer = await asyncio.to_thread(
            make_final_answer,
            user_text,
            groq_answer or "Жооп жок",
            gemini_answer or "Жооп жок",
            openai_answer or "Жооп жок"
        )

    except Exception as e:

        print("❌ FINAL AI ERROR:", repr(e))

        # OpenAI final жооп түзө албаса,
        # биринчи иштеген AI жооп берет.

        final_answer = working_answers[0]


    await waiting.delete()

    await send_long_message(
        update,
        "🧠 AIAssist2026\n\n"
        + final_answer
    )


    # =====================
    # SAVE MEMORY
    # =====================

    add_memory(
        user_id,
        user_text,
        final_answer
    )


# =========================
# TEST API
# =========================

async def test_api(update, context):

    waiting = await update.message.reply_text(
        "🔍 Үч API текшерилип жатат..."
    )

    groq_ok = False
    gemini_ok = False
    openai_ok = False

    try:

        await asyncio.to_thread(
            ask_groq,
            "Бир сөз менен жооп бер: ИШТЕДИ",
            update.effective_user.id
        )

        groq_ok = True

    except Exception as e:

        print("GROQ TEST ERROR:", repr(e))


    try:

        await asyncio.to_thread(
            ask_gemini,
            "Бир сөз менен жооп бер: ИШТЕДИ",
            update.effective_user.id
        )

        gemini_ok = True

    except Exception as e:

        print("GEMINI TEST ERROR:", repr(e))


    try:

        await asyncio.to_thread(
            ask_openai,
            "Бир сөз менен жооп бер: ИШТЕДИ",
            update.effective_user.id
        )

        openai_ok = True

    except Exception as e:

        print("OPENAI TEST ERROR:", repr(e))


    result = (
        "🔍 API TEST ЖЫЙЫНТЫГЫ\n\n"

        f"🔵 Groq: "
        f"{'🟢 ИШТЕДИ' if groq_ok else '🔴 ERROR'}\n\n"

        f"🟢 Gemini: "
        f"{'🟢 ИШТЕДИ' if gemini_ok else '🔴 ERROR'}\n\n"

        f"🟣 OpenAI: "
        f"{'🟢 ИШТЕДИ' if openai_ok else '🔴 ERROR'}"
    )

    await waiting.edit_text(result)


# =========================
# MAIN
# =========================

def main():

    web_thread = threading.Thread(
        target=start_web_server,
        daemon=True
    )

    web_thread.start()


    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )


    app.add_handler(
        CommandHandler("start", start)
    )

    app.add_handler(
        CommandHandler("memory", show_memory)
    )

    app.add_handler(
        CommandHandler("clearmemory", clear_memory)
    )

    app.add_handler(
        CommandHandler("testapi", test_api)
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            chat
        )
    )


    print("")
    print("======================================")
    print("🤖 AIAssist2026")
    print("🧠 MEMORY: ON")
    print("🔵 GROQ: ON")
    print("🟢 GEMINI: ON")
    print("🟣 OPENAI: ON")
    print("======================================")


    app.run_polling()


if __name__ == "__main__":
    main()

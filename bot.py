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
# MEMORY SYSTEM
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

    # Эски форматтагы memory болсо автоматтык толуктайбыз
    user.setdefault("facts", [])
    user.setdefault("goals", [])
    user.setdefault("knowledge", [])
    user.setdefault("experience", [])
    user.setdefault("history", [])

    return user


# =========================================================
# MEMORY TEXT
# =========================================================

def memory_text(user_id):
    user = get_user_memory(user_id)

    result = ""

    if user["facts"]:
        result += "\n👤 ТУРУКТУУ МААНИЛҮҮ МААЛЫМАТ:\n"
        for item in user["facts"][-30:]:
            result += "- " + item + "\n"

    if user["goals"]:
        result += "\n🎯 МАКСАТТАР:\n"
        for item in user["goals"][-20:]:
            result += "- " + item + "\n"

    if user["knowledge"]:
        result += "\n📚 ҮЙРӨНҮЛГӨН МААЛЫМАТ:\n"
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
                + item["assistant"][:1200]
                + "\n\n"
            )

    return result


# =========================================================
# AUTOMATIC LEARNING
# =========================================================

def learn_from_conversation(user_id, user_text, answer):
    """
    Ар бир сүйлөшүүдөн маанилүү маалыматтарды чыгарып,
    Memory'ге автоматтык сактайт.
    """

    user = get_user_memory(user_id)

    prompt = f"""
Сен жеке AI жардамчынын Memory менеджерисиң.

Колдонуучунун жаңы билдирүүсүн жана AI жоопту анализде.

Жаңы маалыматтардан КАЙСЫЛАРЫН келечекте колдонуу үчүн
эс тутумга сактоо керек экенин аныкта.

Маанилүү маалыматтар:
- аты
- жашаган жери
- кесиби
- жөндөмдөрү
- тажрыйбасы
- максаттары
- долбоорлору
- узак мөөнөттүү пландары
- бюджеттери
- артыкчылыктары
- эмне иштеген/иштебеген
- маанилүү чечимдер
- үйрөнүлгөн пайдалуу нерселер

Кыска жана так жаз.

МААНИЛҮҮ:
- Убактылуу сүйлөшүүнү сактаба.
- Жөн гана жалпы кеңештерди сактаба.
- Ойлоп таппа.
- Колдонуучу айтпаган маалыматты кошпо.
- Эгер жаңы маанилүү маалымат жок болсо, бош массив бер.

JSON гана кайтар:

{{
  "facts": [],
  "goals": [],
  "knowledge": [],
  "experience": []
}}

КОЛДОНУУЧУ:
{user_text}

AI ЖООБУ:
{answer}
"""

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Сен так Memory анализаторсуң. JSON гана кайтар."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=2000
        )

        raw = response.choices[0].message.content.strip()

        # Markdown ```json ... ``` болсо тазалайбыз
        if raw.startswith("```"):
            raw = raw.replace("```json", "")
            raw = raw.replace("```", "")
            raw = raw.strip()

        data = json.loads(raw)

        for category in [
            "facts",
            "goals",
            "knowledge",
            "experience"
        ]:
            new_items = data.get(category, [])

            if not isinstance(new_items, list):
                continue

            for item in new_items:
                if not isinstance(item, str):
                    continue

                item = item.strip()

                if not item:
                    continue

                # Дубликат болбосун
                if item not in user[category]:
                    user[category].append(item)

        # Өтө чоң болуп кетпесин
        user["facts"] = user["facts"][-50:]
        user["goals"] = user["goals"][-30:]
        user["knowledge"] = user["knowledge"][-50:]
        user["experience"] = user["experience"][-50:]

        save_memory(memory)

        print("🧠 Memory жаңырды:", user_id)

    except Exception as e:
        print("Learning error:", e)


# =========================================================
# HISTORY
# =========================================================

def add_history(user_id, user_text, answer):

    user = get_user_memory(user_id)

    user["history"].append({
        "user": user_text,
        "assistant": answer
    })

    user["history"] = user["history"][-20:]

    save_memory(memory)


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
Сен AIAssist2026 — колдонуучунун жеке AI жардамчысысың.

Негизги принцип:
Колдонуучуга реалдуу, практикалык жана пайдалуу жооп бер.

Колдонуучунун Memory маалыматтарын контекст катары колдон.

Бирок:
- Memory'де жок маалыматты ойлоп таппа.
- Белгисиз нерсени факт катары айтпа.
- Баалар так болбосо, болжол деп белгилеп кой.
- Киреше боюнча кепилдик бербе.
- Реалдуу шарттарды эске ал.
- Кыргызстандагы шартка мүмкүн болушунча ылайыкташтыр.
- Колдонуучунун мурдагы максаттарын жана тажрыйбасын эске ал.
- Эгер мурунку маалымат эскирген болушу мүмкүн болсо, аны такталган маалымат катары кабыл алба.

Эгер колдонуучу бизнес же киреше жөнүндө сураса:
1. Реалдуу мүмкүнчүлүктөрдү кара.
2. Чыгымдарды эсепке ал.
3. Кардарды кайдан табууну айт.
4. Биринчи кадамды көрсөт.
5. Чоң акча салуудан мурда тест кылууну сунушта.
6. Болжол менен кепилдикти айырмала.

Жооптор:
- түшүнүктүү
- кыска
- практикалык
- керексиз теориясыз
- мүмкүн болсо кадам-кадам болсун.

Колдонуучунун негизги узак мөөнөттүү максаты — кирешесин көбөйтүү,
бирок бул максат өзгөрүшү мүмкүн. Ошондуктан жаңы маалыматты эске ал.
"""


# =========================================================
# GROQ
# =========================================================

def ask_groq(user_id, user_text):

    context = memory_text(user_id)

    prompt = SYSTEM_PROMPT + "\n" + context

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

        return response.choices[0].message.content

    except Exception as e:
        return "Groq error: " + str(e)


# =========================================================
# GEMINI
# =========================================================

def ask_gemini(user_id, user_text):

    context = memory_text(user_id)

    prompt = SYSTEM_PROMPT + "\n" + context

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt + "\n\nКолдонуучу:\n" + user_text
        )

        if response.text:
            return response.text

        return "Gemini жооп берген жок."

    except Exception as e:
        return "Gemini error: " + str(e)


# =========================================================
# OPENAI
# =========================================================

def ask_openai(user_id, user_text):

    context = memory_text(user_id)

    prompt = SYSTEM_PROMPT + "\n" + context

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

        return response.choices[0].message.content

    except Exception as e:
        return "OpenAI error: " + str(e)


# =========================================================
# FINAL AI
# =========================================================

def make_final_answer(
    user_id,
    user_text,
    groq_answer,
    gemini_answer,
    openai_answer
):

    context = memory_text(user_id)

    prompt = f"""
Сен AIAssist2026нин башкы AIсың.

Үч AI бир суроого жооп берди.

Сен алардын жоопторун:
- салыштыр
- туура эмес жерлерин четке как
- пайдалуу маалыматтарды бириктир
- реалдуулугун текшер
- колдонуучунун Memory маалыматтарын эске ал
- бир гана мыкты жооп түз.

Ката маалыматты кайталаба.

Эгер үч AI бири-бирине каршы келсе,
эң негиздүү жана коопсуз вариантты танда.

Сандар так далилденбесе,
аларды кепилдик катары көрсөтпө.

Колдонуучуга үч AIнын жоопторун өзүнчө көрсөтпө.
Бирдиктүү жооп бер.

{context}

========================

🔵 GROQ:
{groq_answer}

========================

🟢 GEMINI:
{gemini_answer}

========================

🟣 OPENAI:
{openai_answer}

========================

ЭМИ БИР ГАНА МЫКТЫ ЖООП БЕР.
"""

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
            max_completion_tokens=8000
        )

        return response.choices[0].message.content

    except Exception as e:

        # Final AI иштебесе OpenAIнин кадимки жообун кайтарабыз
        return openai_answer


# =========================================================
# TELEGRAM START
# =========================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        "🧠 AIAssist2026\n\n"
        "Мен сенин жеке AI жардамчыңмын.\n\n"
        "🔵 Groq\n"
        "🟢 Gemini\n"
        "🟣 OpenAI\n\n"
        "Үч AI жоопторун анализдеп, "
        "бир мыкты жооп берем.\n\n"
        "🧠 Memory системасы да иштейт."
    )


# =========================================================
# MEMORY COMMAND
# =========================================================

async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

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
        text += "\n👤 Маанилүү маалымат:\n"

        for item in user["facts"][-10:]:
            text += "• " + item + "\n"

    if user["goals"]:
        text += "\n🎯 Максаттар:\n"

        for item in user["goals"][-10:]:
            text += "• " + item + "\n"

    await update.message.reply_text(text)


# =========================================================
# CLEAR MEMORY
# =========================================================

async def clear_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = str(update.effective_user.id)

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

async def test_api(update: Update, context: ContextTypes.DEFAULT_TYPE):

    result = "🔍 API TEST\n\n"

    # GROQ
    try:

        r = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Reply only: OK"
                }
            ],
            max_tokens=20
        )

        result += "🔵 Groq: 🟢 ИШТЕДИ\n"

    except Exception as e:

        result += "🔵 Groq: 🔴 ERROR\n"
        result += str(e)[:300] + "\n"

    # GEMINI
    try:

        r = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Reply only: OK"
        )

        result += "🟢 Gemini: 🟢 ИШТЕДИ\n"

    except Exception as e:

        result += "🟢 Gemini: 🔴 ERROR\n"
        result += str(e)[:300] + "\n"

    # OPENAI
    try:

        r = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Reply only: OK"
                }
            ],
            max_completion_tokens=20
        )

        result += "🟣 OpenAI: 🟢 ИШТЕДИ\n"

    except Exception as e:

        result += "🟣 OpenAI: 🔴 ERROR\n"
        result += str(e)[:300] + "\n"

    await update.message.reply_text(result)


# =========================================================
# MAIN MESSAGE
# =========================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = update.effective_user.id
    user_text = update.message.text

    await update.message.reply_text(
        "🧠 Үч AI иштеп жатат...\n\n"
        "🔵 Groq\n"
        "🟢 Gemini\n"
        "🟣 OpenAI"
    )

    # -----------------------------------------------------
    # 1. ҮЧ AI
    # -----------------------------------------------------

    groq_answer = ask_groq(
        user_id,
        user_text
    )

    gemini_answer = ask_gemini(
        user_id,
        user_text
    )

    openai_answer = ask_openai(
        user_id,
        user_text
    )

    # -----------------------------------------------------
    # 2. БИР ГАНА МЫКТЫ ЖООП
    # -----------------------------------------------------

    final_answer = make_final_answer(
        user_id,
        user_text,
        groq_answer,
        gemini_answer,
        openai_answer
    )

    # -----------------------------------------------------
    # 3. HISTORY САКТОО
    # -----------------------------------------------------

    add_history(
        user_id,
        user_text,
        final_answer
    )

    # -----------------------------------------------------
    # 4. AUTO LEARNING
    # -----------------------------------------------------

    learn_from_conversation(
        user_id,
        user_text,
        final_answer
    )

    # -----------------------------------------------------
    # 5. USERге ЖООП
    # -----------------------------------------------------

    await update.message.reply_text(
        final_answer
    )


# =========================================================
# RENDER WEB SERVER
# =========================================================

class HealthHandler(BaseHTTPRequestHandler):

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

    def log_message(self, format, *args):
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
# BOT START
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
            clear_memory
        )
    )

    app.add_handler(
        CommandHandler(
            "testapi",
            test_api
        )
    )

    # Messages
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    print(
        "🤖 AIAssist2026 Telegram AI бот иштеп жатат..."
    )

    app.run_polling()


if __name__ == "__main__":
    main()

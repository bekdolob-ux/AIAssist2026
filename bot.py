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

groq_client = Groq(
    api_key=GROQ_API_KEY
)

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)


# =========================================================
# MEMORY
# =========================================================

def load_memory():

    try:

        if os.path.exists(MEMORY_FILE):

            with open(
                MEMORY_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                return json.load(f)

    except Exception as e:

        print(
            "❌ Memory load error:",
            e
        )

    return {}


def save_memory(data):

    try:

        with open(
            MEMORY_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

    except Exception as e:

        print(
            "❌ Memory save error:",
            e
        )


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

    user.setdefault(
        "facts",
        []
    )

    user.setdefault(
        "goals",
        []
    )

    user.setdefault(
        "knowledge",
        []
    )

    user.setdefault(
        "experience",
        []
    )

    user.setdefault(
        "history",
        []
    )

    return user


def memory_text(user_id):

    user = get_user_memory(
        user_id
    )

    result = ""

    if user["facts"]:

        result += (
            "\n👤 МААНИЛҮҮ МААЛЫМАТ:\n"
        )

        for item in user["facts"][-30:]:

            result += (
                "- "
                + item
                + "\n"
            )

    if user["goals"]:

        result += (
            "\n🎯 МАКСАТТАР:\n"
        )

        for item in user["goals"][-20:]:

            result += (
                "- "
                + item
                + "\n"
            )

    if user["knowledge"]:

        result += (
            "\n📚 БИЛИМ:\n"
        )

        for item in user["knowledge"][-20:]:

            result += (
                "- "
                + item
                + "\n"
            )

    if user["experience"]:

        result += (
            "\n📈 ТАЖРЫЙБА:\n"
        )

        for item in user["experience"][-20:]:

            result += (
                "- "
                + item
                + "\n"
            )

    if user["history"]:

        result += (
            "\n💬 АКЫРКЫ СҮЙЛӨШҮҮЛӨР:\n"
        )

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
Колдонуучуга мүмкүн болушунча пайдалуу,
реалдуу жана практикалык жардам берүү.

Memory маалыматтарын контекст катары колдон.

Маанилүү эрежелер:

- Memory'де жок нерсени ойлоп таппа.
- Белгисиз маалыматты факт катары айтпа.
- Сандык көрсөткүчтөрдү негизсиз ойлоп чыгарба.
- Колдонуучунун кесибин же тажрыйбасын
  далилсиз кеңейтпе.
- Кирешеге кепилдик бербе.
- Кыргызстандагы реалдуу шарттарды эске ал.
- Колдонуучунун мурдагы максаттарын жана
  тажрыйбасын эске ал.
- Эгер маалымат эски болушу мүмкүн болсо,
  этият колдон.

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
# RESPONSE TEXT HELPERS
# =========================================================

def clean_answer(answer):

    if answer is None:
        return ""

    if not isinstance(answer, str):
        answer = str(answer)

    answer = answer.strip()

    return answer


# =========================================================
# GROQ
# =========================================================

def ask_groq(
    user_id,
    user_text
):

    prompt = (
        SYSTEM_PROMPT
        + "\n"
        + memory_text(user_id)
    )

    try:

        response = (
            groq_client
            .chat
            .completions
            .create(
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
        )

        print(
            "🔵 GROQ RAW:",
            str(response)[:1000]
        )

        answer = ""

        if (
            response
            and getattr(
                response,
                "choices",
                None
            )
        ):

            choice = response.choices[0]

            message = getattr(
                choice,
                "message",
                None
            )

            if message:

                answer = getattr(
                    message,
                    "content",
                    ""
                )

        answer = clean_answer(
            answer
        )

        if not answer:

            raise Exception(
                "Groq бош жооп кайтарды"
            )

        return answer

    except Exception as e:

        print(
            "❌ GROQ ERROR:",
            e
        )

        raise


# =========================================================
# GEMINI
# =========================================================

def ask_gemini(
    user_id,
    user_text
):

    prompt = (
        SYSTEM_PROMPT
        + "\n"
        + memory_text(user_id)
        + "\n\nКолдонуучу:\n"
        + user_text
    )

    try:

        response = (
            gemini_client
            .models
            .generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
        )

        answer = clean_answer(
            getattr(
                response,
                "text",
                ""
            )
        )

        if not answer:

            raise Exception(
                "Gemini бош жооп кайтарды"
            )

        return answer

    except Exception as e:

        print(
            "❌ GEMINI ERROR:",
            e
        )

        raise


# =========================================================
# OPENAI
# =========================================================

def ask_openai(
    user_id,
    user_text
):

    full_prompt = (
        SYSTEM_PROMPT
        + "\n"
        + memory_text(user_id)
        + "\n\nКолдонуучу:\n"
        + user_text
    )

    try:

        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=full_prompt,
            max_output_tokens=5000
        )

        print(
            "🟣 OPENAI RESPONSE:",
            str(response)[:1000]
        )

        answer = clean_answer(
            getattr(
                response,
                "output_text",
                ""
            )
        )

        if not answer:

            raise Exception(
                "OpenAI бош жооп кайтарды"
            )

        return answer

    except Exception as e:

        print(
            "❌ OPENAI ERROR:",
            e
        )

        raise


# =========================================================
# SMART ROUTER
# =========================================================

def smart_ai_router(
    user_id,
    user_text
):

    # =====================================================
    # GROQ
    # =====================================================

    try:

        answer = ask_groq(
            user_id,
            user_text
        )

        return (
            "🔵 Groq",
            answer
        )

    except Exception:

        print(
            "➡️ Groq иштеген жок."
            " Geminiге өтөбүз."
        )


    # =====================================================
    # GEMINI
    # =====================================================

    try:

        answer = ask_gemini(
            user_id,
            user_text
        )

        return (
            "🟢 Gemini",
            answer
        )

    except Exception:

        print(
            "➡️ Gemini иштеген жок."
            " OpenAIге өтөбүз."
        )


    # =====================================================
    # OPENAI
    # =====================================================

    try:

        answer = ask_openai(
            user_id,
            user_text
        )

        return (
            "🟣 OpenAI",
            answer
        )

    except Exception:

        print(
            "❌ ҮЧ AI ТЕҢ ИШТЕГЕН ЖОК."
        )


    return (
        "❌ AI кызматтарынын үчөө тең "
        "учурда жеткиликсиз.",
        None
    )


# =========================================================
# MEMORY HISTORY
# =========================================================

def add_history(
    user_id,
    user_text,
    answer,
    ai_name
):

    user = get_user_memory(
        user_id
    )

    user["history"].append({

        "user": user_text,

        "assistant": answer,

        "ai": ai_name
    })

    user["history"] = (
        user["history"][-30:]
    )

    save_memory(
        memory
    )


# =========================================================
# SIMPLE AUTO MEMORY
# =========================================================

def simple_auto_memory(
    user_id,
    user_text
):

    user = get_user_memory(
        user_id
    )

    text = (
        user_text
        .lower()
        .strip()
    )

    # Суроолорду Memory'ге сактаба
    if "?" in user_text:

        return

    # -----------------------------------------------------
    # NAME
    # -----------------------------------------------------

    if (
        "менин атым" in text
        or "аты-жөнүм" in text
    ):

        fact = user_text.strip()

        if fact not in user["facts"]:

            user["facts"].append(
                fact
            )


    # -----------------------------------------------------
    # GOAL
    # -----------------------------------------------------

    goal_words = [

        "максатым",

        "негизги максат",

        "максат —",

        "максат -",

        "каалайм"
    ]

    if any(
        word in text
        for word in goal_words
    ):

        if (
            user_text
            not in user["goals"]
        ):

            user["goals"].append(
                user_text.strip()
            )


    # -----------------------------------------------------
    # SKILLS
    # -----------------------------------------------------

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

        if (
            user_text
            not in user["facts"]
        ):

            user["facts"].append(
                user_text.strip()
            )


    user["facts"] = (
        user["facts"][-50:]
    )

    user["goals"] = (
        user["goals"][-30:]
    )

    save_memory(
        memory
    )


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

        "Кайсы AI жеткиликтүү болсо,"
        " ошол автоматтык колдонулат.\n\n"

        "🧠 Memory иштейт."
    )


# =========================================================
# MEMORY COMMAND
# =========================================================

async def memory_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = (
        update.effective_user.id
    )

    user = get_user_memory(
        user_id
    )

    text = (

        "🧠 MEMORY\n\n"

        f"👤 Facts: "
        f"{len(user['facts'])}\n"

        f"🎯 Goals: "
        f"{len(user['goals'])}\n"

        f"📚 Knowledge: "
        f"{len(user['knowledge'])}\n"

        f"📈 Experience: "
        f"{len(user['experience'])}\n"

        f"💬 History: "
        f"{len(user['history'])}\n"
    )

    if user["facts"]:

        text += (
            "\n👤 Маалымат:\n"
        )

        for item in user["facts"][-10:]:

            text += (
                "• "
                + item
                + "\n"
            )

    if user["goals"]:

        text += (
            "\n🎯 Максаттар:\n"
        )

        for item in user["goals"][-10:]:

            text += (
                "• "
                + item
                + "\n"
            )

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

    save_memory(
        memory
    )

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

    result = (
        "🔍 API TEST\n\n"
    )


    # =====================================================
    # GROQ
    # =====================================================

    try:

        response = (
            groq_client
            .chat
            .completions
            .create(

                model=GROQ_MODEL,

                messages=[
                    {
                        "role": "user",
                        "content": "Reply only: OK"
                    }
                ],

                max_tokens=20
            )
        )

        answer = ""

        if (
            response
            and getattr(
                response,
                "choices",
                None
            )
        ):

            message = getattr(
                response.choices[0],
                "message",
                None
            )

            if message:

                answer = clean_answer(
                    getattr(
                        message,
                        "content",
                        ""
                    )
                )

        if answer:

            result += (
                "🔵 Groq: 🟢 ИШТЕДИ\n"
            )

        else:

            result += (
                "🔵 Groq: 🔴 БОШ ЖООП\n"
            )

    except Exception as e:

        result += (
            "🔵 Groq: 🔴 ERROR\n"
            + str(e)[:500]
            + "\n\n"
        )


    # =====================================================
    # GEMINI
    # =====================================================

    try:

        response = (
            gemini_client
            .models
            .generate_content(

                model=GEMINI_MODEL,

                contents="Reply only: OK"
            )
        )

        answer = clean_answer(
            getattr(
                response,
                "text",
                ""
            )
        )

        if answer:

            result += (
                "🟢 Gemini: 🟢 ИШТЕДИ\n"
            )

        else:

            result += (
                "🟢 Gemini: 🔴 БОШ ЖООП\n"
            )

    except Exception as e:

        result += (
            "🟢 Gemini: 🔴 ERROR\n"
            + str(e)[:500]
            + "\n\n"
        )


    # =====================================================
    # OPENAI
    # =====================================================

    try:

        response = (
            openai_client
            .responses
            .create(

                model=OPENAI_MODEL,

                input="Reply only: OK",

                max_output_tokens=20
            )
        )

        answer = clean_answer(
            getattr(
                response,
                "output_text",
                ""
            )
        )

        if answer:

            result += (
                "🟣 OpenAI: 🟢 ИШТЕДИ\n"
            )

        else:

            result += (
                "🟣 OpenAI: 🔴 БОШ ЖООП\n"
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

    user_id = (
        update.effective_user.id
    )

    user_text = (
        update.message.text
    )


    # =====================================================
    # MEMORY
    # =====================================================

    simple_auto_memory(
        user_id,
        user_text
    )


    # =====================================================
    # STATUS
    # =====================================================

    await update.message.reply_text(

        "🧠 AI текшерилип жатат...\n\n"

        "🔵 Groq → "
        "🟢 Gemini → "
        "🟣 OpenAI"
    )


    # =====================================================
    # ROUTER
    # =====================================================

    ai_name, answer = (
        smart_ai_router(
            user_id,
            user_text
        )
    )


    # =====================================================
    # ALL FAILED
    # =====================================================

    if answer is None:

        await update.message.reply_text(
            ai_name
        )

        return


    # =====================================================
    # HISTORY
    # =======

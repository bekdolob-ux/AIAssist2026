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


# ============================================================
# ENV
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GOOGLE_CREDENTIALS_JSON = os.getenv(
    "GOOGLE_CREDENTIALS_JSON"
)

FIRESTORE_DATABASE = (
    "ai-studio-e7d6a6f3-8921-414c-887e-4b97383c7bed"
)

GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3.8-flash"
OPENAI_MODEL = "gpt-5-mini"


# ============================================================
# API CLIENTS
# ============================================================

groq_client = None
gemini_client = None
openai_client = None
db = None


if GROQ_API_KEY:
    try:
        groq_client = Groq(
            api_key=GROQ_API_KEY
        )
        print("✅ GROQ CLIENT READY")
    except Exception as e:
        print(
            "❌ GROQ INIT ERROR:",
            repr(e)
        )


if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )
        print("✅ GEMINI CLIENT READY")
    except Exception as e:
        print(
            "❌ GEMINI INIT ERROR:",
            repr(e)
        )


if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        print("✅ OPENAI CLIENT READY")
    except Exception as e:
        print(
            "❌ OPENAI INIT ERROR:",
            repr(e)
        )


# ============================================================
# FIRESTORE
# ============================================================

try:

    if not GOOGLE_CREDENTIALS_JSON:
        raise Exception(
            "GOOGLE_CREDENTIALS_JSON жок"
        )

    credentials_info = json.loads(
        GOOGLE_CREDENTIALS_JSON
    )

    credentials = (
        service_account
        .Credentials
        .from_service_account_info(
            credentials_info
        )
    )

    db = firestore.Client(
        project=credentials_info["project_id"],
        credentials=credentials,
        database=FIRESTORE_DATABASE
    )

    print("✅ FIRESTORE CLIENT READY")

except Exception as e:

    print(
        "❌ FIRESTORE INIT ERROR:",
        repr(e)
    )

    db = None


# ============================================================
# FIRESTORE USER REFERENCE
# ============================================================

def get_user_ref(user_id):

    if db is None:
        return None

    return (
        db.collection("users")
        .document(str(user_id))
    )


# ============================================================
# LOAD NORMAL MEMORY
# ============================================================

def load_memory(user_id):

    try:

        ref = get_user_ref(user_id)

        if ref is None:
            return []

        doc = ref.get()

        if not doc.exists:
            return []

        data = doc.to_dict() or {}

        return data.get(
            "history",
            []
        )

    except Exception as e:

        print(
            "❌ MEMORY LOAD ERROR:",
            repr(e)
        )

        return []


# ============================================================
# LOAD IMPORTANT MEMORY
# ============================================================

def load_important_memory(user_id):

    try:

        ref = get_user_ref(user_id)

        if ref is None:
            return []

        doc = ref.get()

        if not doc.exists:
            return []

        data = doc.to_dict() or {}

        memory = data.get(
            "important_memory",
            []
        )

        if not isinstance(memory, list):
            return []

        return memory

    except Exception as e:

        print(
            "❌ IMPORTANT MEMORY LOAD ERROR:",
            repr(e)
        )

        return []


# ============================================================
# SAVE NORMAL MEMORY
# ============================================================

def save_memory(
    user_id,
    user_text,
    ai_text
):

    try:

        ref = get_user_ref(user_id)

        if ref is None:

            print(
                "❌ MEMORY SAVE: Firestore unavailable"
            )

            return False

        doc = ref.get()

        if doc.exists:

            data = doc.to_dict() or {}

            history = data.get(
                "history",
                []
            )

        else:

            history = []

        history.append({

            "user": str(user_text),

            "assistant": str(ai_text)

        })

        # Акыркы 50 диалог
        history = history[-50:]

        ref.set({

            "telegram_user_id":
                str(user_id),

            "history":
                history,

            "updated":
                firestore.SERVER_TIMESTAMP

        }, merge=True)

        print(
            f"🧠 MEMORY SAVED: "
            f"user={user_id}, "
            f"dialogs={len(history)}"
        )

        return True

    except Exception as e:

        print(
            "❌ MEMORY SAVE ERROR:",
            repr(e)
        )

        return False


# ============================================================
# SAVE IMPORTANT MEMORY
# ============================================================

def save_important_memory(
    user_id,
    new_memories
):

    try:

        ref = get_user_ref(user_id)

        if ref is None:
            return False

        if not new_memories:
            return False

        if not isinstance(
            new_memories,
            list
        ):
            return False

        doc = ref.get()

        if doc.exists:

            data = doc.to_dict() or {}

            old_memory = data.get(
                "important_memory",
                []
            )

        else:

            old_memory = []

        if not isinstance(
            old_memory,
            list
        ):
            old_memory = []


        # Жаңы маанилүү маалыматтарды кошуу
        for item in new_memories:

            if not item:
                continue

            item = str(item).strip()

            if not item:
                continue

            # Дубликатты кошпойбуз
            if item not in old_memory:

                old_memory.append(
                    item
                )


        # Өтө чоң болуп кетпесин
        old_memory = old_memory[-100:]


        ref.set({

            "telegram_user_id":
                str(user_id),

            "important_memory":
                old_memory,

            "updated":
                firestore.SERVER_TIMESTAMP

        }, merge=True)


        print(
            f"⭐ IMPORTANT MEMORY SAVED: "
            f"user={user_id}, "
            f"items={len(old_memory)}"
        )

        return True

    except Exception as e:

        print(
            "❌ IMPORTANT MEMORY SAVE ERROR:",
            repr(e)
        )

        return False


# ============================================================
# CLEAR MEMORY
# ============================================================

def clear_memory(user_id):

    try:

        ref = get_user_ref(user_id)

        if ref is None:
            return False

        ref.set({

            "telegram_user_id":
                str(user_id),

            "history":
                [],

            "important_memory":
                [],

            "updated":
                firestore.SERVER_TIMESTAMP

        }, merge=True)

        print(
            f"🧹 MEMORY CLEARED: {user_id}"
        )

        return True

    except Exception as e:

        print(
            "❌ MEMORY CLEAR ERROR:",
            repr(e)
        )

        return False


# ============================================================
# IMPORTANT MEMORY EXTRACTION
# ============================================================

def extract_important_memory(
    user_text,
    ai_text
):

    prompt = f"""
Сен жеке AI жардамчынын Memory системасысың.

Төмөнкү сүйлөшүүдөн КЕЛЕЧЕКТЕ дагы пайдалуу боло турган
туруктуу маалыматтарды гана аныкта.

Маанилүү маалыматтар:
- узак мөөнөттүү максаттар
- киреше/карьера максаттары
- долбоорлор
- үйрөнүп жаткан нерселер
- туруктуу кызыгуулар
- жеке артыкчылыктар
- иштөө стили
- AI жардамчыга байланыштуу туруктуу талаптар

Маанилүү ЭМЕС:
- жөнөкөй суроолор
- убактылуу маанай
- бир жолку тапшырма
- жалпы билим
- AI берген сунуштар
- сырсөздөр
- API key
- токендер
- жеке кооптуу маалыматтар

ТЕК ГАНА JSON массив кайтар.

Мисалы:
["Максат: айына 100 000 сом киреше табуу",
 "Долбоор: AIAssist2026"]

Эгер маанилүү маалымат жок болсо:
[]

Колдонуучу:
{user_text}

AI:
{ai_text}
"""

    # Extraction үчүн Groq колдонобуз
    try:

        if groq_client:

            response = groq_client.chat.completions.create(

                model=GROQ_MODEL,

                messages=[

                    {
                        "role": "system",
                        "content": (
                            "Сен Memory extractorсиң. "
                            "Ар дайым valid JSON array гана кайтар."
                        )
                    },

                    {
                        "role": "user",
                        "content": prompt
                    }

                ],

                temperature=0

            )

            result = (
                response
                .choices[0]
                .message
                .content
            )

            if not result:
                return []

            result = result.strip()

            # Markdown ```json ... ``` болсо тазалайбыз
            if result.startswith(
                "```"
            ):

                result = (
                    result
                    .replace(
                        "```json",
                        ""
                    )
                    .replace(
                        "```",
                        ""
                    )
                    .strip()
                )

            parsed = json.loads(
                result
            )

            if isinstance(
                parsed,
                list
            ):

                return [
                    str(x).strip()
                    for x in parsed
                    if str(x).strip()
                ]

    except Exception as e:

        print(
            "⚠️ MEMORY EXTRACTION ERROR:",
            repr(e)
        )

    return []


# ============================================================
# BUILD MEMORY CONTEXT
# ============================================================

def build_memory_context(
    user_id
):

    important = load_important_memory(
        user_id
    )

    history = load_memory(
        user_id
    )

    text = ""


    # Маанилүү Memory
    if important:

        text += (
            "\n\n⭐ УЗАК МӨӨНӨТТҮҮ "
            "МААНИЛҮҮ MEMORY:\n"
        )

        for item in important[-100:]:

            text += (
                f"- {item}\n"
            )


    # Акыркы диалогдор
    if history:

        recent = history[-10:]

        text += (
            "\n\n🧠 АКЫРКЫ "
            "СҮЙЛӨШҮҮЛӨРДӨН КОНТЕКСТ:\n"
        )

        for item in recent:

            text += (

                f"Колдонуучу: "
                f"{item.get('user', '')}\n"

                f"AI: "
                f"{item.get('assistant', '')}\n"

            )

    return text


# ============================================================
# GROQ
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
                        "Сен Бекболоттун жеке AI "
                        "жардамчысысың. "
                        "Сенин негизги максатың — "
                        "анын кирешесин көбөйтүүгө "
                        "жана жеке өнүгүүсүнө жардам берүү. "
                        "Мурунку Memory маалыматтарын "
                        "эске ал. "
                        "Практикалык, так жана түшүнүктүү "
                        "жооп бер. "
                        "Эгер контекстте маанилүү "
                        "маалымат болсо, аны "
                        "тиешелүү учурда колдон."
                    )

                },

                {

                    "role": "user",

                    "content": prompt

                }

            ],

            temperature=0.7

        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        print(
            "❌ GROQ ERROR:",
            repr(e)
        )

        return None


# ============================================================
# GEMINI
# ============================================================

def ask_gemini(prompt):

    if not gemini_client:
        return None

    try:

        response = (
            gemini_client
            .models
            .generate_content(

                model=GEMINI_MODEL,

                contents=prompt

            )
        )

        if response and response.text:

            return response.text

    except Exception as e:

        print(
            "❌ GEMINI ERROR:",
            repr(e)
        )

    return None


# ============================================================
# OPENAI
# ============================================================

def ask_openai(prompt):

    if not openai_client:
        return None

    try:

        response = (
            openai_client
            .responses
            .create(

                model=OPENAI_MODEL,

                input=(

                    "Сен Бекболоттун жеке AI "
                    "жардамчысысың. "
                    "Практикалык, так жана "
                    "пайдалуу жооп бер.\n\n"

                    + prompt

                )

            )
        )

        return response.output_text

    except Exception as e:

        print(
            "❌ OPENAI ERROR:",
            repr(e)
        )

        return None


# ============================================================
# AI ROUTER
# ============================================================

def ask_ai(prompt):

    # 1. Groq

    answer = ask_groq(
        prompt
    )

    if answer:

        print(
            "🔵 Groq → OK"
        )

        return answer


    # 2. Gemini

    answer = ask_gemini(
        prompt
    )

    if answer:

        print(
            "🟢 Gemini → OK"
        )

        return answer


    # 3. OpenAI

    answer = ask_openai(
        prompt
    )

    if answer:

        print(
            "🟣 OpenAI → OK"
        )

        return answer


    return (
        "❌ Азыр AI кызматтарынын "
        "баары жооп берген жок."
    )


# ============================================================
# TELEGRAM LONG MESSAGE
# ============================================================

async def send_long_message(
    update,
    text
):

    if not text:

        text = (
            "❌ AI бош жооп кайтарды."
        )

    MAX_MESSAGE_LENGTH = 4000

    for i in range(
        0,
        len(text),
        MAX_MESSAGE_LENGTH
    ):

        chunk = text[
            i:i + MAX_MESSAGE_LENGTH
        ]

        await update.message.reply_text(
            chunk
        )


# ============================================================
# MESSAGE HANDLER
# ============================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if (
        not update.message
        or not update.message.text
    ):

        return


    user_id = (
        update
        .effective_user
        .id
    )

    user_text = (
        update
        .message
        .text
        .strip()
    )


    print(
        f"📩 USER {user_id}: "
        f"{user_text}"
    )


    # ========================================================
    # MEMORY CONTEXT
    # ========================================================

    memory_context = (
        await asyncio.to_thread(
            build_memory_context,
            user_id
        )
    )


    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""

Сенин колдонуучуң Бекболот.

Төмөндө анын Memory маалыматтары берилген.

{memory_context}

ЖАҢЫ БИЛДИРҮҮ:

{user_text}

Эрежелер:
1. Memory'деги маалыматты туура колдон.
2. Билбеген нерсени ойлоп чыгарба.
3. Колдонуучуга түз жана пайдалуу жооп бер.
4. Кыргыз тилинде жооп бер, эгер башка тил суралбаса.
5. Ашыкча узартпа, бирок керектүү деталдарды бер.

"""


    # ========================================================
    # AI
    # ========================================================

    answer = await asyncio.to_thread(
        ask_ai,
        prompt
    )


    # ========================================================
    # TELEGRAM RESPONSE
    # ========================================================

    await send_long_message(
        update,
        answer
    )


    # ========================================================
    # NORMAL MEMORY SAVE
    # ========================================================

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


    # ========================================================
    # IMPORTANT MEMORY EXTRACTION
    # ========================================================

    important = await asyncio.to_thread(

        extract_important_memory,

        user_text,

        answer

    )


    if important:

        saved_important = (
            await asyncio.to_thread(
                save_important_memory,
                user_id,
                important
            )
        )

        if saved_important:

            print(
                "⭐ Маанилүү Memory "
                "автоматтык сакталды:"
            )

            for item in important:

                print(
                    f"   • {item}"
                )


# ============================================================
# /START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(

        "🤖 Салам, Бекболот!\n\n"

        "Мен сенин жеке AI жардамчыңмын.\n\n"

        "🧠 Normal Memory — "
        "сүйлөшүүлөрдү сактайт.\n"

        "⭐ Important Memory — "
        "маанилүү маалыматтарды "
        "автоматтык түрдө бөлүп сактайт.\n\n"

        "💰 Негизги максат — "
        "кирешеңди көбөйтүүгө жардам берүү.\n\n"

        "/memory — Memory абалы\n"

        "/clearmemory — Memory тазалоо\n"

        "/testapi — системаларды текшерүү"

    )


# ============================================================
# /MEMORY
# ============================================================

async def memory_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = (
        update
        .eff

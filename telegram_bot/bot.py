import os
import requests
import asyncio
import uuid
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    ContextTypes,
    filters
)

TOKEN = os.getenv("TELEGRAM_TOKEN", "8728740717:AAH36D0H3NA54GHCVHHeQh840Wd-oXSQNtM")
SERVER_URL = os.getenv(
    "SERVER_URL",
    "http://127.0.0.1:8000/analyze"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(TEMP_DIR, exist_ok=True)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):

    try:
        await update.message.reply_text("Фото получены, анализируем...")

        photos = update.message.photo
        media_group_id = update.message.media_group_id

        files_to_send = []

        # если это альбом
        if media_group_id:

            # ждём, пока все фото альбома соберутся
            if "media_groups" not in context.bot_data:
                context.bot_data["media_groups"] = {}

            if media_group_id not in context.bot_data["media_groups"]:
                context.bot_data["media_groups"][media_group_id] = []

            context.bot_data["media_groups"][media_group_id].append(update.message)

            await asyncio.sleep(1)

            messages = context.bot_data["media_groups"].pop(media_group_id, [])

        else:
            messages = [update.message]

        for msg in messages:
            photo = msg.photo[-1]
            file = await photo.get_file()

            input_path = os.path.join(
                TEMP_DIR,
                f"{uuid.uuid4()}.jpg"
            )

            await file.download_to_drive(input_path)

            files_to_send.append(
                ("images", open(input_path, "rb"))
            )

        response = requests.post(
            SERVER_URL,
            files=files_to_send,
            timeout=120
        )

        if response.status_code != 200:
            await update.message.reply_text("Ошибка сервера")
            return

        data = response.json()

        for result in data.get("results", []):

            metrics = result.get("metrics", {})
            metrics_text = f"""Результаты анализа:

        Корень:
        Длина: {metrics.get('root_length_cm', 0)} см
        Площадь: {metrics.get('root_area_cm2', 0)} см²

        Стебель:
        Длина: {metrics.get('stem_length_cm', 0)} см
        Площадь: {metrics.get('stem_area_cm2', 0)} см²

        Листья:
        Площадь: {metrics.get('leaf_area_cm2', 0)} см²
        """

            img_response = requests.get(result["image_url"], timeout=60)

            if img_response.status_code == 200:

                result_path = os.path.join(
                    TEMP_DIR,
                    f"{uuid.uuid4()}.jpg"
                )

                with open(result_path, "wb") as f:
                    f.write(img_response.content)

                with open(result_path, "rb") as img:
                    await update.message.reply_photo(
                    photo=img,
                    caption=metrics_text
                    )

        await update.message.reply_text("Анализ завершен")

    except Exception as e:
        print("BOT ERROR:", e)
        await update.message.reply_text("Ошибка анализа изображения")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        "Пожалуйста, отправьте фотографию для анализа."
    )


def main():
    print("Bot starting...")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(
        MessageHandler(filters.PHOTO, handle_photo)
    )
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )
    print("Bot started")
    
    # Увеличиваем timeout для long polling до 60 секунд
    app.run_polling(poll_interval=1, timeout=60)

if __name__ == "__main__":
    main()
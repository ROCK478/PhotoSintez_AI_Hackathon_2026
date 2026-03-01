import requests
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# Вставьте сюда токен вашего бота (латиница, без пробелов!)
TOKEN = "8728740717:AAH36D0H3NA54GHCVHHeQh840Wd-oXSQNtM"

SERVER_URL = "http://localhost:5000/analyze"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):

    try:
        photo = update.message.photo[-1]
        file = await photo.get_file()
        input_path = "temp_input.jpg"
        await file.download_to_drive(input_path)
        await update.message.reply_text("Фото получено, анализируем...")

        # отправляем в Flask
        with open(input_path, "rb") as f:

            response = requests.post(
                "http://127.0.0.1:5000/analyze",
                files={"image": f}
            )

        if response.status_code != 200:

            await update.message.reply_text("Ошибка сервера Flask")
            return

        data = response.json()

        image_url = data["image_url"]
        filename = image_url.split("/")[-1]
        result_path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(result_path):
            await update.message.reply_text("Файл результата не найден")
            return

        # отправляем фото
        with open(result_path, "rb") as img:

            await update.message.reply_photo(photo=img)

        await update.message.reply_text("Анализ завершен")

    except Exception as e:

        print("BOT ERROR:", e)

        await update.message.reply_text("Ошибка анализа изображения")


app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
print("Bot started...")
app.run_polling()
# bot.py
import logging
import requests
from io import BytesIO
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# Вставьте сюда токен вашего бота (латиница, без пробелов!)
TOKEN = "8728740717:AAH36D0H3NA54GHCVHHeQh840Wd-oXSQNtM"

logging.basicConfig(level=logging.INFO)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # получаем фото от пользователя (последнее в массиве)
        photo = update.message.photo[-1]
        file = await photo.get_file()
        await file.download_to_drive("input.jpg")  # сохраняем локально

        await update.message.reply_text("Фото получено, анализируем...")

        # отправка на Flask сервер
        with open("input.jpg", "rb") as f:
            response = requests.post("http://127.0.0.1:5000/analyze", files={"image": f})
        data = response.json()
        metrics = data["metrics"]

        # скачиваем обработанное изображение с сервера
        url = "http://127.0.0.1:5000/results/input.jpg"
        res = requests.get(url)
        res.raise_for_status()  # если что-то не так, будет исключение

        # отправляем фото пользователю
        await update.message.reply_photo(photo=BytesIO(res.content))

        # отправляем метрики
        await update.message.reply_text(
            f"Корень: {metrics['root_length']} см\n"
            f"Стебель: {metrics['stem_length']} см\n"
            f"Листья: {metrics['leaf_area']} см²"
        )

    except Exception as e:
        logging.error(f"Ошибка при обработке фото: {e}")
        await update.message.reply_text("Произошла ошибка при обработке изображения.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Отправьте изображение растения.")

# создаем приложение бота
app = ApplicationBuilder().token(TOKEN).build()

# добавляем обработчики
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
app.add_handler(MessageHandler(filters.TEXT, handle_text))

print("Bot started...")
app.run_polling()
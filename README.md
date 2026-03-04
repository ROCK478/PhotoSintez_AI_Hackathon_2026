# ИИ-Хакатон
**Цель: обучить ML-модель на сегментацию компонентов растений с вычислением их длин и площади. Разработать веб-сервис и телеграм бота, позволяющих
взаимодействовать с моделью**

**Технологии: YOLO, RoboFlow, Python, HTML+CSS, JS**

**Дата: Март 2026**

**Результат: ???**

## Инструкция пользователя: 1
Для простого тестирования напишите мне в тг: @medvedvityan

Таким образом сайт будет работать по ссылке: https://rock478.github.io/PhotoSintez_AI_Hackathon_2026/

Телеграм бот: @PhotoSintezAI_bot

## Инструкция пользователя: 2
1. Скачиваем репозиторий
2. Устанавливаем зависимости
```bash
pip install -r requirements.txt
```
3. Запускаем сервер:
```bash
python app.py
```
4. Запускаем бота во втором терминале:
```bash
python telegram_bot/bot.py
```
5. Открываем index.html (либо двойным нажатием, либо через http)

Пример открытия через http:
```bash
python -m http.server 8080                    
```

Далее в браузере вставляем ссылку: http://{ip}:8080/index.html, где ip - ip своего компьютера. 
Бот @PhotoSintezAI_bot также станет доступен к использованию.

## Возможность автономной разметки:
Обучение модели с теми же параметрами, которые использовались, а также получение результата через консольную команду можно следующим образом:
```bash
python yolo_console.py train --dataset path/to/dataset.yaml
```

```bash
python yolo_console.py inference --model model/best.pt --image path/to/image.jpg --output results/
```

## Особенности:
- Клиент-серверная архитектура с единым ML-ядром
- 4 модуля (FlaskAPI, ML-model, Frontend, Telegram-bot)
- Индивидуальная разработка
- Проект RoboFlow доступен по ссылке: https://app.roboflow.com/felikss-workspace/my-first-project-pczmy/
- Ресурсы от Google Colab позволили использовать модель с более затратными ресурсами (например: imgsize=980)

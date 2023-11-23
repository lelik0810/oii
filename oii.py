import random
import re
import nltk

import json
with open ("my/bot.json","r") as config_file:
    data = json.load(config_file)
INTENTS = data["intents"]
#print(INTENTS)
#print("фраз ", len(INTENTS))


def filter_text(text):
    text.strip()# удаление лишних пробелов в начале и в конце знаки препинания, поэтому подключаем пакет re
    expression = r'[^\w\s]'   # регулярное выражение = “все что не слово и не пробел»
# '^'  - это отрицание    \w – это обозначение слов      \s – пробел
    text = re.sub(expression, "", text)   # sub – заменить все “все что не слово и не пробел» на «пустоту» в  text
    return text


def text_match(user_text, example):
    user_text = user_text.lower()  # приводим текст к нижнему регистру .  Для решении проблемы 1
    example = example.lower()
    # Дописать функцию так, что бы все примеры ниже работали

    user_text = filter_text(user_text)# фильтруем пользовательский ввод

    if user_text.find(example) != -1:
        return True
        # внутри одной строчки есть другая
        # фраза найдена в user_text

    text_len = len(user_text) #длина текста
    difference = nltk.edit_distance(user_text, example) / text_len
    # отношение кол-ва ошибок к длине слова, 1 - слово целиком другое, 0 - слово полностью совподает
    return difference < 0.4

'''INTENTS = {
    "hello": {
    "examples": ["Привет", "Hello", "Здравствуйте"],
    "responses": ["Здрасте", "Hi!!!", "Приветики"],
    },
    "how_do_oyu_do":{
        "examples": ["Как дела?","Вопрос", "Какие планы?"],
        "responses": ["Функционирую в пределах заданных параметров", "OK!!!Как дела?","Болтать"],
    },
    "your_name": {
        "examples": ["Как твое имя?", "Как тебя зовут?"],
        "responses": ["Чат-бот ПГУ 1.0", "Автоответчик"],
    },
}'''
# функция, которая находит намерение пользователя по его тексту  с помощью text_match
def get_intent(user_text):
    for intent in INTENTS:
        examples = INTENTS[intent]["examples"]# список фраз
        for example in examples:
            if len(filter_text(example))<3:
                continue
            if text_match(user_text, example):
                return intent # найденное намерение подходит к польз. тексту
    return None # ничего не найдено

# функция возвращает случайную фразу по данному контент
def get_random_response(intent):
    return random.choice(INTENTS[intent]["responses"])

'''user_text = ""
while user_text != "Пока":
    print("[USER]: ", end = '')
    user_text = input()
    intent = get_intent(user_text)
    response = get_random_response(intent)
    print(f'[BOT]:{response}')'''

#https://drive.google.com/file/d/1_L5CYGsO58zkB3LMBG73ezIEwYFD07Ed/view
# классификация текста - к какому классу (интент) относится текст
# модель должна будет это делать сама.
# создадим обучающую выборку (фраза + интент)
# обучающая выботка состоит из входных и выходных данных
# фраза (х) - на вход
# Интерт (у) - на выход
X = []
y = []
for intent in INTENTS:
    examples = INTENTS[intent]["examples"]
    for example in examples:
        example = filter_text(example)
        if len(example)<3:
            continue # пропускаем текст, если он слишком короткий
        X.append(example) # добавляем вразу в список х
        y.append(intent)
#print(len(X), len (y))
# векторизация текстов
# подобрать, настроить, обучить модель. Интегрировать модель в общую логику бота

# векторизация текстов: из набора текста делаем вектор [1,2,3,4]
# любая модель машинного обучения - математическая, т.е. найти закономерность между входными и выходными данными
'''
1. Набор Текстов = {
     "мама мыла раму",
     ...
2. векторайзер. Нужно обучить векторайзер на наборе данных
мама = 1
мыла = 2
раму = 3
3. Векторизация
[1 2 3]

}
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
https://colab.research.google.com/drive/1fIvs3k7PsgsYFZ-x95bF6SkN9CAWt3FU?usp=sharing#scrollTo=ufxqHdxn7d1B
Например: CountVectorizer, TfIDFVectorizer
'''
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer() # Настройки в скобочках
# ДЗ: попробовать ngram_range, analyzer
vectorizer.fit(X) # Обучаем векторайзер
vecX = vectorizer.transform(X) # Все тексты преобразуем в вектора
#print(X[1200], y[1200])
#print(vecX[1200])
# Пример
# print(vectorizer.transform(["смотришь ты телевизор смотришь телевизор"]))

# Обучаем модель классификации
# Выбираем алгоритм/модель (экспериментировать, иметь опыт)
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
# ДЗ: попробовать MLPRegressor
model = RandomForestClassifier() # ДЗ, попробовать Настройки, n_estimators = ?, max_depth?
model.fit(vecX, y) # Обучение модели
model.predict(vectorizer.transform(["Насколько ты разумна?"]))

# Метрика = мат. инструмент для оценки качества модели
from sklearn.metrics import accuracy_score, f1_score
# https://scikit-learn.org/stable/modules/classes.html#classification-metrics
# Интенты предсказанные моделью
y_pred = model.predict(vecX)

#print("accuracy_score", accuracy_score(y, y_pred)) # Сравниваем y и y_pred
#print("f1_score", f1_score(y, y_pred, average="macro")) # Сравниваем y и y_pred

def get_intent_ml(user_text):
    user_text = filter_text(user_text)
    vec_text = vectorizer.transform([user_text])
    intent = model.predict(vec_text)[0]
    # model.predict_proba()
    return intent

def bot(user_text):
    intent = get_intent(user_text)
    if intent:
        return get_random_response(intent)
    intent = get_intent_ml(user_text)
    return get_random_response(intent)

#print(bot("чувак! я мери"))

import pandas as pd
#proba = model.predict_proba(vectorizer.transform(["Привет, как дела"]))

# Вывести на экран список вероятных интентов
#print(pd.DataFrame(columns=model.classes_, data=[proba[0]]).T.sort_values(by=0, ascending=False))
import nest_asyncio
nest_asyncio.apply() # Решение ошибки "This event loop is already running"
TOKEN="6111624795:AAGTtcAJaYnLiXNCwYLkmokmVU1rZT6aag0"
from telegram import Update  # Обновление пришедшее к нам с серверов ТГ
from telegram.ext import ApplicationBuilder, MessageHandler, filters

# Создаем и настраиваем бот-приложение
app = ApplicationBuilder().token(TOKEN).build()


async def telegram_reply(upd: Update, ctx):
    name = upd.message.from_user.full_name
    user_text = upd.message.text
    print(f"{name}: {user_text}")
    reply = bot(user_text)
    print(f"BOT: {reply}")
    await upd.message.reply_text(reply)


handler = MessageHandler(filters.TEXT, telegram_reply)  # Создаем обработчик текстовых сообщений
app.add_handler(handler)  # Добавляем обработчик в приложение

# app.run_polling # Наш код регулярно опрашивает сервер на предмет новых Апдейтов
# app.run_webhook # Запускает веб-сервер, к которому будет подключаться сам ТГ и присылать туда апдейты
app.run_polling()
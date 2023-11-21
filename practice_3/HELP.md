# Практическая работа №3

**Выполнил** - студент группы РИМ 130963, **Бакланов Иван Леонидович**

**Для копирования проекта и перехода в рабочюю директорию необходимов исполнить команды:**

    git clone git@github.com:kwazart/demo-ml.git
    cd demo-ml
    git checkout practice-3-baklanov
    cd practice_3

**Установка окружения и зависимостей:**

    pip install virtualenv
    virtualenv venv
    source venv/bin/activate
    python -m pip install -r requirements.txt

**Без requirements.txt зависимости следующие:**

    python -m pip install fastapi uvicorn httpx Pillow transformers torch torchvision torchaudio

**Запуск веб-сервера для разработки:**

    uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000

Далее можно производить HTTP запросы:

    Endpoint: http://127.0.0.1:8000/predict/
    Method: POST
    Content-Type: application/json
    Body: {"url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"}

Подставить любое изображение из сети в формате jpg

Пример curl запроса:

    curl -X 'POST' \
        'http://127.0.0.1:8000/predict/' \
        -H 'Content-Type: application/json' \
        -d '{
        "url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    }'

В ответе будет текстовое описание для изображения.

# Практическое задание №4
__Выполнил:__ *Домченко Максим*

__Студент группы:__ *РИМ-130962*

__Задача:__ *Поиск объектов в изображении*

__Описание:__

* Выбрана обученная модель `google/owlvit-base-patch32`
с сайта [huggingface.co](https://huggingface.co/google/owlvit-base-patch32)
* Модель принимает на вход изображение и название объекта(объектов) которое необходимо найти
* На выходе модель отдаёт массив найденных объектов, каждый объект содержит:
  * `score` - вероятность определения объекта
  * `label` - название объекта
  * `box` - координаты квадрата в котором найден объект на изображении

__Варианты использования:__
* Определение количества машин на парковке
* Определение животных в кадре
* Поиск предметов на конвейере
* Поиск запрещенных объектов

__Реализация:__
* Реализовано API с помощью библиотеки FastAPI
* Для первоначального определения объектов на изображении интегрирована обученная модель `EfficientNetB0`
* Для определения объектов по названию в изображении, интегрирована обученная модель `google/owlvit-base-patch32`
* API размещено в Яндекс.Облако

__Локальный запуск:__
* Стянут проект `git clone git@github.com:kwazart/demo-ml.git`
* Перейти в проект `cd demo-ml`
* Переключить ветку на практическую #3 `git checkout practice_4-domchenko`
* Перейти в папку с практической `cd ./practice_4-domchenko/`
* Подготовить окружение (_виртуальная среда, установка пакетов_) командой `make deps`
* Запустить командой `make run`

__Запуск в облаке:__
* Подключиться к ВМ в облаке по ssh
* Стянут проект по https `git clone https://github.com/kwazart/demo-ml.git`
* Перейти в проект `cd demo-ml`
* Переключить ветку на практическую #4 `git checkout practice_4-domchenko`
* Перейти в папку с практической `cd ./practice_4-domchenko/`
* Установить утилиту `make` командой `sudo apt install make`
* Установить `venv` для `python` командой `sudo apt install python3.10-venv`
* Подготовить окружение (_виртуальная среда, установка пакетов_) командой `make deps`
* Запустить командой `make run`

Дополнительные команды описаны в `Makefile`

Для систем в которых отсутствует утилита `make`, запуск можно выполнить путём выполнения содержимого соответствующих команд описанных в `Makefile`

__Результат:__
* Разработано API с интегрированными обученными моделями ML:
* Локальный:
  * `http://127.0.0.1:8000/` - расположено приветствие, можно проверить, что сервер запущен
  * `http://127.0.0.1:8000/docs` - расположена документация
  * `http://127.0.0.1:8000/predict/` - метод с моделями, примеры запросов ниже
* В облаке:
  * `http://158.160.132.70:8000/` - расположено приветствие, можно проверить, что сервер запущен
  * `http://158.160.132.70:8000/docs` - расположена документация
  * `http://158.160.132.70:8000/predict/` - метод с моделями, примеры запросов ниже

__Примеры запросов для curl локально:__

`curl -X 'POST'
    'http://127.0.0.1:8000/predict/'
    -H 'Content-Type: application/json'
    -d '{
    "url": "https://parkingcars.ru/wp-content/uploads/2021/02/stoyanka-1024x683.jpg",
    "targets": "car"
}'`

`curl -X 'POST'
    'http://127.0.0.1:8000/predict/'
    -H 'Content-Type: application/json'
    -d '{
    "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ2_q3Ph31cc_MgsovOHJOKqIyTxaWnWmckLw&usqp=CAU",
    "targets": "hog"
}'`

`curl -X 'POST'
    'http://127.0.0.1:8000/predict/'
    -H 'Content-Type: application/json'
    -d '{
    "url": "https://storage.yandexcloud.net/mfi/1242/products/main/3474.jpg",
    "targets": "pineapple"
}'`

__Примеры запросов для curl в облаке:__

`curl -X 'POST'
    'http://158.160.132.70:8000/predict/'
    -H 'Content-Type: application/json'
    -d '{
    "url": "https://parkingcars.ru/wp-content/uploads/2021/02/stoyanka-1024x683.jpg",
    "targets": "car"
}'`

`curl -X 'POST'
    'http://158.160.132.70:8000/predict/'
    -H 'Content-Type: application/json'
    -d '{
    "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ2_q3Ph31cc_MgsovOHJOKqIyTxaWnWmckLw&usqp=CAU",
    "targets": "hog"
}'`

`curl -X 'POST'
    'http://158.160.132.70:8000/predict/'
    -H 'Content-Type: application/json'
    -d '{
    "url": "https://storage.yandexcloud.net/mfi/1242/products/main/3474.jpg",
    "targets": "pineapple"
}'`

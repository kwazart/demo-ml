# Для использования необходимо находится в practice_3-domchenko/

# Подготовка виртуального окружения
.PHONY: venv
venv:
	python3 -m venv ./venv

# Установка указанных пакетов (когда отсутствует файл `requirements.txt`)
.PHONY: direct_deps
direct_deps: venv
	(. ./venv/bin/activate; pip3 install \
		requests transformers Pillow torch transformers numpy tensorflow pydantic fastapi uvicorn)

# Установка пакетов указанных в `requirements.txt`
.PHONY: deps
deps: venv
	(. ./venv/bin/activate; pip install -r requirements.txt)

# Удаление виртуального окружения
.PHONY: del_deps
del_deps:
	rm -R ./venv

# Сохранение зависимостей в файл `requirements.txt`
.PHONY: freeze
freeze:
	(. ./venv/bin/activate; pip freeze > requirements.txt)

# Запуск
.PHONY: run
run:
	(. ./venv/bin/activate; uvicorn main:app --workers 4 --host 127.0.0.1 --port 8000)

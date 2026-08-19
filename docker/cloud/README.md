# Образ для облачного обучения B2 — сборка и запуск

Эта директория собирает автономный Docker-образ для запуска обучения с
подкреплением (Isaac Lab) на четвероногом роботе (Unitree B2) на чистой
облачной GPU-машине.

## Что лежит в этой директории

- `Dockerfile` — собирает образ.
- `b2_overlay/` — небольшой набор конфигурационных файлов задачи,
  копируемых в образ при сборке (описание робота/задачи, гиперпараметры
  обучения).
- `sync_results.sh` — вспомогательный скрипт для скачивания готовых
  чекпоинтов туда, откуда его запускают, через `rsync`/`ssh`.

Больше ничего из этой директории для сборки или запуска образа не нужно.

## Требования к целевой машине

- Ubuntu 22.04, x86_64.
- NVIDIA GPU (ориентир: H100/H200, RTX 5090, RTX 6000) со свежим драйвером
  (>= 550) и установленным `nvidia-container-toolkit`, чтобы работал
  `docker run --gpus all`.
- Docker Engine (20.10+) с настроенным NVIDIA runtime.
- ~40 ГБ свободного диска под образ и логи обучения.
- Доступ в интернет во время сборки (Dockerfile скачивает базовый образ и
  клонирует два публичных репозитория GitHub).

Быстрая проверка, что GPU-passthrough работает, до начала сборки:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Сборка

Из этой директории (именно она должна быть контекстом сборки Docker — не
собирать из родительской директории):

```bash
cd docker/cloud
docker build -t b2-cloud-train .
```

В первый раз сборка займёт время (скачивание базового образа + полная
установка Isaac Lab).

## Запуск обучения

```bash
mkdir -p ./cloud_logs
docker run --rm -it --gpus all \
  -v "$(pwd)/cloud_logs:/workspace/robot_lab/logs" \
  b2-cloud-train \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task <TASK_NAME> --headless
```

Вместо `<TASK_NAME>` подставить выданный вам идентификатор Gym-задачи
(например, что-то из зарегистрированных под
`RobotLab-Isaac-Velocity-Rough-Unitree-B2-*`). Чтобы увидеть полный список
задач, зарегистрированных в образе:

```bash
docker run --rm b2-cloud-train \
  ./isaaclab.sh -p scripts/environments/list_envs.py
```

Полезные флаги `train.py`:
- `--headless` — без GUI, обязателен на сервере без дисплея.
- `--num_envs <N>` — переопределить число параллельных сред симуляции.
- `--max_iterations <N>` — переопределить длину обучения.
- `--resume --load_run <run_dir> --checkpoint <file>` — продолжить с
  чекпоинта, уже лежащего под `logs/` (например, скопированного из
  предыдущей сессии).

Обучение пишет чекпоинты (`model_*.pt`), снимок конфига и файлы событий
TensorBoard в `logs/rsl_rl/<task>/<run_timestamp>/` внутри контейнера, что
благодаря монтированию `-v` выше сразу появляется на хосте в
`./cloud_logs/rsl_rl/<task>/<run_timestamp>/`.

Смотреть прогресс удалённо:
```bash
tensorboard --logdir ./cloud_logs --bind_all
```

## Забрать результаты обратно

С машины, куда нужны результаты (необязательно той же, что собирала
образ), при наличии SSH-доступа к машине обучения:

```bash
./sync_results.sh user@training-host [remote_path_to_robot_lab]
```

Это делает `rsync`-выкачку `model_*.pt`, `*.onnx`, `*.yaml` и файлов
событий TensorBoard из `<remote_path_to_robot_lab>/logs/` в
`../../cloud_results/` (относительно этой директории).

Вручную то же самое:
```bash
rsync -avz user@training-host:~/robot_lab/logs/ ./cloud_results/
```

## Экспорт обученной политики в ONNX (опционально)

Isaac Lab / rsl_rl поставляют отдельный экспортёр чекпоинт→ONNX, которому
не нужен ни GPU, ни запущенная симуляция — удобно экспортировать уже на
более лёгкой машине после скачивания `.pt`-файла:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task <TASK_NAME> --checkpoint <path_to_model.pt> --headless
```

(смотрите `--help` этого скрипта на предмет флагов только-экспорта в вашей
версии образа, если не нужен проигрыш/визуализация эпизода).

## Диагностика

Если шаг сборки падает на разрешении версий — проверьте, что тег базового
образа в `Dockerfile` и закреплённые версии Python-пакетов всё ещё
доступны — реестр контейнеров NVIDIA и PyPI иногда снимают с публикации
старые теги/wheel-файлы.

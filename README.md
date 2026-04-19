# Детекция госномеров (YOLO + OCR)

Минимальный README только с основными командами.

## Студенты

| ФИО | Группа |
|-----|--------|
| Пивень Егор Дмитриевич | 972402 |
| Гончиков Александр Константинович | 972403 |

## 1. Обучение нейросети

После обучения лучшие веса сохраняются в models/best.pt.
Исходный Roboflow-экспорт содержит только train; train/valid/test генерируются автоматически в data/dataset.

```bash
python train.py --dataset-source "My First Project.v1i.yolov8" --dataset-out data/dataset --epochs 50 --imgsz 640 --batch 16
```

## 2. Обработка видео

Пример: обработать файл Video/video1.mp4 и сохранить результат в Video/video1_detected.mp4.

```bash
python app.py --mode video --weights models/best.pt --source Video/video1.mp4 --output Video/video1_detected.mp4
```

## 3. Запуск веб-камеры

```bash
python app.py --mode camera --weights models/best.pt --camera-index 0 --show
```

## 4. Как запустить оценку и увидеть precision

Запуск оценки на тестовом сплите (по умолчанию `test`):

```bash
python evaluate.py --weights models/best.pt --dataset-source "My First Project.v1i.yolov8" --dataset-out data/dataset --split test --output data/eval_metrics.json
```

После выполнения в консоли будет напечатан JSON с метриками, включая `precision`.

Также метрики сохраняются в файл `data/eval_metrics.json`.
Пример содержимого:

```json
{
	"mAP50": 0.87,
	"mAP50-95": 0.61,
	"precision": 0.83,
	"recall": 0.79,
	"latency_ms": 5.2,
	"split": "test"
}
```

Если нужен другой сплит, укажите `--split val` или `--split train`.

## 5. Если нужен Poetry (опционально)

Если хотите запускать через poetry run, установите Poetry:

```powershell
py -m pip install --user poetry
```

После этого можно использовать команды формата:

```bash
poetry run python app.py --mode video --weights models/best.pt --source Video/video1.mp4 --output Video/video1_detected.mp4
```

## 6. Папка с фото распознанных номеров

Папка WebCamPhotoResults содержит фотографии с веб-камеры, на которых система распознала госномера.

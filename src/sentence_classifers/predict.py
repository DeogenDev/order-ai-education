"""Предсказание на основе предобученной модели"""


from pathlib import Path
from typing import Optional

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base import SentenceClassifierlBase

class SentenceClassifier(SentenceClassifierlBase):

    OUTPUT_COLUMNS = ["id", "text", "label", "probability"]
    NDITS = 4

    def __init__(self, config):
        self.config = config
        self.label2id = config.LABEL_MAPPING
        self.final_model_dir = str(self.config.FINAL_MODEL_DIR)

        self.model = AutoModelForSequenceClassification.from_pretrained(self.final_model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.final_model_dir)

        # Автоматически выбираем GPU, если он доступен
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def run(
        self,
        text: Optional[str] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None
        ) -> None:
        self.model.eval()
        if text:
            label, prob = self._predict_text(text)
            print(f"label: {label}")
            print(f"probability: {prob:.4f}")
        elif input_path and output_path:
            self._process_csv(input_path, output_path)
        else:
            print("❌ Ошибка: Укажите либо --text, либо одновременно --input и --output.")

    def _predict_text(self, text: str):
        """Инференс для одной строки текста (возвращает label и prob)"""
        inputs = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.config.MAX_LENGTH, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        label_name = self.model.config.id2label.get(predicted_class_id, str(predicted_class_id))

        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)
        order_id = self.config.LABEL_MAPPING.get("order", 0)
        order_probability = probabilities[order_id].item()

        return label_name, order_probability

    def _process_csv(self, input_path: str, output_path: str):
        df = pd.read_csv(input_path)

        if "id" not in df.columns or "text" not in df.columns:
            print("❌ Ошибка: Входной CSV должен обязательно содержать колонки 'id' и 'text'!")
            return


        texts = df["text"].astype(str).tolist()
        ids = df["id"].tolist()

        output_records = []

        # Запускаем цикл по батчам
        for i in range(0, len(texts), self.config.BATCH_SIZE):
            batch_texts = texts[i : i + self.config.BATCH_SIZE]
            batch_ids = ids[i : i + self.config.BATCH_SIZE]

            # Токенизируем всю пачку сразу
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.MAX_LENGTH,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            # Находим предсказанные классы для всего батча
            predicted_class_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            # Считаем вероятности для всего батча
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            order_id = self.config.LABEL_MAPPING.get("order", 0)
            # Извлекаем колонку вероятностей только для целевого класса 'order'
            order_probabilities = probabilities[:, order_id].cpu().tolist()

            for row_id, text_content, class_id, order_prob in zip(batch_ids, batch_texts, predicted_class_ids, order_probabilities):
                pred_label = self.model.config.id2label.get(class_id, str(class_id))
                row_values = [row_id, text_content, pred_label, round(order_prob, self.NDITS)]

                output_records.append(dict(zip(self.OUTPUT_COLUMNS, row_values)))

        output_df = pd.DataFrame(output_records)
        output_df = output_df[self.OUTPUT_COLUMNS]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        print(f"✅ Результаты успешно сохранены в формат ТЗ по пути: {output_path}")

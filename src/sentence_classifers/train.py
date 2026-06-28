
import numpy as np
import datetime
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
    set_seed
)

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from .base import SentenceClassifierlBase


class SentenceClassifierTrainer(SentenceClassifierlBase):
    def __init__(self, config):
        self.config = config

        self.label2id = config.LABEL_MAPPING
        self.id2label = {v: k for k, v in self.label2id.items()} # {0: 'order', 1: 'not_order'}
        self.generated_baseline_metrics = None

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.MODEL_NAME, 
            num_labels=self.config.NUM_LABELS,
            id2label=self.id2label,
            label2id=self.label2id
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)

        self.training_args = TrainingArguments(
            output_dir=str(config.CHECKPOINTS_DIR),
            learning_rate=config.LEARNING_RATE,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            num_train_epochs=config.EPOCHS,
            logging_dir=str(config.LOGS_DIR),
            logging_steps=10,         
            eval_strategy="epoch",    
            save_strategy="epoch",
            seed=config.SEED,
            remove_unused_columns=False
        )

    def run(self, evaluate: bool = False):

        set_seed(self.config.SEED)

        tokenized_datasets = self._get_tokenized_datasets(evaluate=evaluate)

        train_data = tokenized_datasets["train"]
        val_data = tokenized_datasets["validation"]

        self.config.FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(self.config.FINAL_MODEL_DIR))

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics
        )

        trainer.train()
        if evaluate:
            eval_results = trainer.evaluate()
            self.config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            report_content = self._get_report_content(eval_results)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            base_report_path = Path(self.config.VALIDATION_REPORT_FILE)
            unique_report_name = f"{base_report_path.stem}_{timestamp}{base_report_path.suffix}"
            final_report_path = base_report_path.with_name(unique_report_name)
            
            with open(final_report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
                
            print(f"📊 Отчет валидации успешно сохранен: {final_report_path}")

        self.model.save_pretrained(str(self.config.FINAL_MODEL_DIR))

    def _generate_baseline_metrics(self, raw_datasets):
        """Автоматическое обучение простейшего бейслайна и расчет его метрик"""
        try:
            train_texts = raw_datasets["train"]["text"]
            train_labels = [int(self.label2id[lbl]) for lbl in raw_datasets["train"]["label"]]
            
            val_texts = raw_datasets["validation"]["text"]
            val_labels = [int(self.label2id[lbl]) for lbl in raw_datasets["validation"]["label"]]

            vectorizer = TfidfVectorizer(max_features=5000)
            X_train = vectorizer.fit_transform(train_texts)
            X_val = vectorizer.transform(val_texts)

            clf = LogisticRegression(max_iter=200, random_state=self.config.SEED)
            clf.fit(X_train, train_labels)
            
            baseline_preds = clf.predict(X_val)
            class_names = [str(k) for k in sorted(self.label2id.keys(), key=lambda x: self.label2id[x])]

            report_text = classification_report(
                val_labels, 
                baseline_preds, 
                target_names=class_names, 
                zero_division=0
            )

            cm = confusion_matrix(val_labels, baseline_preds)

            precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels,
                baseline_preds,
                average="binary",
                zero_division=0,
                pos_label=self.config.LABEL_MAPPING["order"]
            )
            acc = accuracy_score(val_labels, baseline_preds)

            self.generated_baseline_metrics = {
                "f1": f1,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "classification_report": report_text,
                "confusion_matrix": cm,
            }
        except Exception as e:
            print(f"⚠️ Не удалось сгенерировать автоматический бейслайн: {e}")
            self.generated_baseline_metrics = None

    def _get_report_content(self, eval_results):
        """Генерация Markdown отчета с идеальным форматированием строк"""
        
        def safe_fmt(val, fmt, default="N/A"):
            return f"{val:{fmt}}" if isinstance(val, (int, float)) else default

        loss_str = safe_fmt(eval_results.get("eval_loss"), ".4f")
        runtime_str = safe_fmt(eval_results.get("eval_runtime"), ".2f")
        s_per_sec_str = safe_fmt(eval_results.get("eval_samples_per_second"), ".2f")

        ml_acc = eval_results.get("eval_accuracy", 0.0)
        ml_f1 = eval_results.get("eval_f1", 0.0)
        ml_prec = eval_results.get("eval_precision", 0.0)
        ml_rec = eval_results.get("eval_recall", 0.0)

        baseline = self.generated_baseline_metrics or {}
        base_report = baseline.get("classification_report", "Нет данных")
        base_cm = baseline.get("confusion_matrix", "Нет данных")

        def format_confusion_matrix(cm):
            if cm is None or not hasattr(cm, "shape") or cm.shape != (2, 2):
                return "*Нет корректных данных матрицы ошибок*"

            tp = cm[0][0]
            fn = cm[0][1]
            fp = cm[1][0]
            tn = cm[1][1]

            table_str = (
                f"| Реальный \\ Предсказанный | **order (0)** | **not_order (1)** |\n"
                f"| :--- | :---: | :---: |\n"
                f"| **order (0)** | {tp} *(TP)* | {fn} *(FN)* |\n"
                f"| **not_order (1)** | {fp} *(FP)* | {tn} *(TN)* |"
            )

            return table_str


        cm_markdown_table = format_confusion_matrix(base_cm)

        if getattr(self, "generated_baseline_metrics", None):
            base_f1 = self.generated_baseline_metrics["f1"]
            base_acc = self.generated_baseline_metrics["accuracy"]
            base_prec = self.generated_baseline_metrics["precision"]
            base_rec = self.generated_baseline_metrics["recall"]

            def get_delta_str(ml_val, base_val):
                delta = ml_val - base_val
                return f"📈 (+{delta:.4f})" if delta > 0 else f"📉 ({delta:.4f})"

            metrics_block = (
                f"## 📊 Сравнение результатов: Авто-Baseline (TF-IDF+LR) vs ML (Transformer)\n\n"
                f"| Метрика | Baseline | Текущая ML Модель | Прогресс (Delta) |\n"
                f"| :--- | :---: | :---: | :---: |\n"
                f"| **F1-Score (Target Class)** | {base_f1:.4f} | {ml_f1:.4f} | {get_delta_str(ml_f1, base_f1)} |\n"
                f"| **Accuracy** | {base_acc:.4f} | {ml_acc:.4f} | {get_delta_str(ml_acc, base_acc)} |\n"
                f"| **Precision** | {base_prec:.4f} | {ml_prec:.4f} | {get_delta_str(ml_prec, base_prec)} |\n"
                f"| **Recall** | {base_rec:.4f} | {ml_rec:.4f} | {get_delta_str(ml_rec, base_rec)} |\n\n"
                f"### 🔍 Детальный анализ Baseline (по обоим классам)\n\n"
                f"#### Матрица ошибок (Confusion Matrix):\n"
                f"{cm_markdown_table}\n\n"
                f"#### Поклассовый отчет (Classification Report):\n"
                f"```text\n"
                f"{base_report}\n"
                f"```"
            )
        else:
            metrics_block = (
                f"## 📊 Результаты валидации модели (Validation)\n\n"
                f"| Метрика | Значение |\n"
                f"| :--- | :---: |\n"
                f"| **F1-Score (Target Class)** | {ml_f1:.4f} |\n"
                f"| **Accuracy** | {ml_acc:.4f} |\n"
                f"| **Precision** | {ml_prec:.4f} |\n"
                f"| **Recall** | {ml_rec:.4f} |"
            )

        raw_report = f"""# Отчет по валидации модели (Validation Report)

* **Дата фиксации:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
* **Архитектура ML-модели:** `{self.config.MODEL_NAME}`

{metrics_block}

## ⚡️ Метрики скорости инференса (Inference)
* **Общее время инференса валидации:** {runtime_str} сек.
* **Скорость обработки (пропускная способность):** {s_per_sec_str} строк/сек.
* **Итоговая ошибка (Eval Loss):** {loss_str}

---
*Конфигурация обучения: Эпох: {self.config.EPOCHS}, Батч: {self.config.BATCH_SIZE}, LR: {self.config.LEARNING_RATE}*
"""

        return raw_report

    def _compute_metrics(self, eval_pred):
        """Расчет бинарных метрик для текущей ML-модели"""
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="binary",
            zero_division=0,
            pos_label=self.config.LABEL_MAPPING["order"]
        )
        acc = accuracy_score(labels, preds)
        
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def _get_tokenized_datasets(self, evaluate: bool = False):
        """Загрузка сырых CSV данных, расчет бейслайна и токенизация"""
        raw_datasets = load_dataset(
            "csv", 
            data_files={
                "train": str(self.config.TRAIN_CSV),
                "validation": str(self.config.VAL_CSV)
            }
        )

        if evaluate:
            self._generate_baseline_metrics(raw_datasets)

        tokenized_datasets = raw_datasets.map(
            self._tokenize_function, 
            batched=True,
            remove_columns=raw_datasets["train"].column_names
        )
        
        tokenized_datasets = tokenized_datasets.with_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )

        return tokenized_datasets

    def _tokenize_function(self, examples):
        result = self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.config.MAX_LENGTH
        )

        result["labels"] = [int(self.label2id[lbl]) for lbl in examples["label"]]
        return result

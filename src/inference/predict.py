

import argparse
import config
from src.sentence_classifers import SentenceClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инференс модели по формату ТЗ.")
    parser.add_argument("--text", type=str, help="Текст для классификации.")
    parser.add_argument("--input", type=str, help="Путь к исходному CSV-файлу с колонками id и text.")
    parser.add_argument("--output", type=str, help="Путь для сохранения итогового CSV.")
    args = parser.parse_args()

    sentence_classifier = SentenceClassifier(config)

    sentence_classifier.run(args.text, args.input, args.output)

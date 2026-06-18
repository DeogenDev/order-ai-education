
import config
from src.sentence_classifers import SentenceClassifierTrainer

if __name__ == "__main__":
    trainer = SentenceClassifierTrainer(config)

    trainer.run(evaluate=config.EVALUATE)
    print("🎉 Обучение успешно завершено! Модель сохранена.")

from expert import main as expert
from data_spliter import main as data_spliter
from train_dataset_cleaner import main as train_dataset_cleaner
from huggingface_hub import login
from datasets import load_dataset, Image
import shutil


if __name__ == "__main__":
    shutil.rmtree("./expert_data")
    expert()
    data_spliter()
    train_dataset_cleaner()
    token = input("Введите ваш токен от hf, убедитесь что у токена есть права на write: ")
    repo = input("Ввыедите путь для датасета, убедитесь, что репозиторий уже создан: ")
    login(token)

    dataset = load_dataset(
        "json",
        data_files={
            "train": "expert_data/train.jsonl",
            "train_clean": "expert_data/train_clean.jsonl",
            "test": "expert_data/test.jsonl"
        }
    )
    dataset = dataset.cast_column("images", Image())
    dataset.push_to_hub(repo)
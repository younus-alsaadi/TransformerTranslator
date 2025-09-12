from src.pytorch.train import train_model as train_model_pytorch
from src.tf.train import train_model as train_model_tf
from src.config import get_config

def do_training(cfg: dict):
    lib = cfg.get("lib", "").lower()
    if lib == "pytorch" or lib == "torch":
        return train_model_pytorch(cfg)
    elif lib == "tf" or lib == "tensorflow":
        return train_model_tf(cfg)
    else:
        raise ValueError(f"Unsupported lib '{cfg.get('lib')}'. Use 'pytorch' or 'tf'.")


if __name__ == "__main__":
    cfg = get_config()
    do_training(cfg)
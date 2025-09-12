from pathlib import Path
import re

def get_config():
    return {
        "project":"Translation_EN-DE_Transformer",
        "lib": "torch", # pytorch or tf
        "batch_size": 8,
        "num_epochs": 20,
        'max_train_batches':5, #-1 use all of the batche size
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "de",
        "lang_tgt": "en",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "resume": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }


def get_weights_file_path(config, lib:str, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') /lib/ model_folder / model_filename)


# Find the latest weights file in the weights folder
import re

def latest_weights_file_path(config: dict, lib: str = "pytorch"):
    """
    Returns the latest checkpoint path for the given lib.
    - PyTorch: looks for *.pt in  <lib>/<datasource>_<model_folder>/
    - TF: uses tf.train.latest_checkpoint(<lib>/<datasource>_<model_folder>/)
    """
    lib_key = lib.strip().lower()
    base_dir = Path(lib_key) / f"{config['datasource']}_{config['model_folder']}"
    if not base_dir.exists():
        print(f"[latest] directory does not exist: {base_dir}")
        return None

    stem = config["model_basename"]  # e.g. "tmodel_"

    if lib_key in ("pytorch", "torch"):
        # e.g. pytorch/opus_books_weights/tmodel_XX.pt
        candidates = list((base_dir).glob(f"{stem}*.pt"))
        print(f"[latest] candidates: {[str(p) for p in candidates]}")
        if not candidates:
            return None

        # extract trailing integer from filename (tmodel_XX.pt -> XX)
        def epoch_num(p: Path) -> int:
            m = re.search(rf"{re.escape(stem)}(\d+)\.pt$", p.name)
            return int(m.group(1)) if m else -1

        candidates.sort(key=epoch_num)
        return str(candidates[-1])

    elif lib_key in ("tf", "tensorflow"):
        # e.g. tf/opus_books_weights/tmodel_XX (TF will add .index/.data)
        try:
            import tensorflow as tf
            ckpt = tf.train.latest_checkpoint(str(base_dir))
            print(f"[latest] tf.latest_checkpoint -> {ckpt}")
            return ckpt
        except Exception as e:
            print(f"[latest][tf] warning: {e}")
            return None

    else:
        raise ValueError(f"Unsupported lib '{lib}'. Use 'pytorch' or 'tf'.")

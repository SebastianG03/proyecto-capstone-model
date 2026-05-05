import logging
import json
import threading
import time
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from ultralytics.models import YOLO
import torch
from roboflow import Project, Roboflow
from watchfiles import watch

from app.core.config import (
    BALL_MODEL_NAME, PLAYER_MODEL_NAME,
    ROBOFLOW_API_KEY, ROBOFLOW_BALL_DATASET,
    ROBOFLOW_PLAYER_DATASET, ROBOFLOW_USER)
from app.utils import routes

@dataclass
class TrainData:
    label: Path
    image: Path

@dataclass
class RetrainingConfig:
    # Umbrales de activación
    min_hard_examples: int = 300        # mínimo para considerar reentrenamiento
    min_auto_labels: int = 300          # mínimo de pseudo-labels generados
    max_label_noise_ratio: float = 0.10 # máximo 10% de detecciones conflictivas
    # Métricas mínimas para aceptar el nuevo modelo
    min_map50: float = 0.72             # el nuevo modelo debe superar este mAP
    min_map50_95: float = 0.50
    max_regression_delta: float = 0.05  # no puede caer más de 5% vs baseline

    retraining_log_path: str = "retraining_log.json"


class RetrainingOrchestrator:
    """
    Orquesta el ciclo completo de reentrenamiento con:
    - Validación previa (¿vale la pena entrenar?)
    - Entrenamiento del 'challenger'
    - Evaluación comparativa vs 'champion' actual
    - Promoción o rechazo del challenger
    - Rollback automático si algo falla
    """

    def __init__(self):
        self.config = RetrainingConfig()
        self.logger = logging.getLogger("retraining")
        self._log: list[dict] = self._load_log()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def retrain(
        self
    ):
        """
        Decide si es momento de reentrenar y lo ejecuta si corresponde.
        Retorna True si se promovió un nuevo modelo.
        """
        for changes in watch(routes.DATASETS_DIR, recursive=True, stop_event=self._stop_event):
            print(changes)
            self.logger.info("=== Evaluando condiciones para reentrenamiento")
            player_retrain = self._should_retrain(routes.PLAYER_CUSTOM_DATASET)
            ball_retrain = self._should_retrain(routes.BALL_CUSTOM_DATASET)
            
            if not player_retrain and not ball_retrain:
                self.logger.info("No hay suficientes ejemplos para reentrenar. Abortando.")
                continue
            
            player_promoted, ball_promoted = self._run_retraining_cycle()
            
            if player_promoted:
                self.logger.info("Promoviendo nuevo modelo de jugador.")
            if ball_promoted:
                self.logger.info("Promoviendo nuevo modelo de bola.")

    
    def _should_retrain(self, dataset_path: Path) -> bool:
        files = list(dataset_path.glob("*.jpg"))
        labels = list(dataset_path.glob("*.txt"))
        min_files = len(files) > self.config.min_hard_examples
        anotation_files = len(labels) == len(files)
        
        return min_files and anotation_files and all((dataset_path / (f.stem + ".txt")).exists() for f in files)

    def group_train_data(self, dataset_images: List[Path], dataset_route: Path):
        train_data = []
        for image in dataset_images:
            label_name = image.stem + ".txt"
            label_path = dataset_route / label_name
            train_data.append(TrainData(label=label_path, image=image))
        return train_data
    
    def create_yaml(self, route: Path, label: str, project: str, version: int):
        with open(route / "data.yaml", "w") as f:
            f.writelines(f"train: {route / 'train'} \n")
            f.writelines("nc: 1 \n")
            f.writelines(f"names: ['{label}'] \n")
            f.writelines(f"roboflow: \n")
            f.writelines(f"  workspace: {ROBOFLOW_USER} \n")
            f.writelines(f"  project: {project} \n")
            f.writelines(f"  version: {version} \n")
            f.writelines("license: CC BY 4.0")
    
    def create_sub_folders(self, route: Path):
        (route / "train" / "labels").mkdir(parents=True, exist_ok=True) 
        (route / "train" / "images").mkdir(parents=True, exist_ok=True)
        (route / "data.yaml").touch(exist_ok=True)

    def prepare_custom_dataset(self):
        player_images = list(routes.PLAYER_CUSTOM_DATASET.glob("*.jpg"))
        ball_images = list(routes.BALL_CUSTOM_DATASET.glob("*.jpg"))
        self.create_sub_folders(routes.PLAYER_CUSTOM_DATASET)
        self.create_sub_folders(routes.BALL_CUSTOM_DATASET)

        player_data = self.group_train_data(player_images, routes.PLAYER_CUSTOM_DATASET)
        ball_data = self.group_train_data(ball_images, routes.BALL_CUSTOM_DATASET)
        self.create_yaml(routes.PLAYER_CUSTOM_DATASET, "player", ROBOFLOW_PLAYER_DATASET, 2)
        self.create_yaml(routes.BALL_CUSTOM_DATASET, "ball", ROBOFLOW_BALL_DATASET, 2)
        
        for player in player_data:
            image_name = Path(routes.PLAYER_CUSTOM_DATASET / "train" / "images" / player.image.name)
            label_name = Path(routes.PLAYER_CUSTOM_DATASET / "train" / "labels" / player.label.name)

            player.image.rename(image_name.as_posix())
            player.label.rename(label_name.as_posix())
            
            player.image = image_name
            player.label = label_name
        
        for ball in ball_data:
            image_name = Path(routes.BALL_CUSTOM_DATASET / "train" / "images" / ball.image.name)
            label_name = Path(routes.BALL_CUSTOM_DATASET / "train" / "labels" / ball.label.name)
            
            ball.image.rename(image_name.as_posix())
            ball.label.rename(label_name.as_posix())
            
            ball.image = image_name
            ball.label = label_name
        
        return player_data, ball_data
    
    
    def _run_retraining_cycle(
        self,
    ):
        start_time = time.time()
        self.logger.info("Iniciando ciclo de reentrenamiento...")

        try:
            # Paso 1: Preparar datos mezclados
            rf = Roboflow(api_key=ROBOFLOW_API_KEY)
            workspace = rf.workspace(ROBOFLOW_USER)
            ball_project = workspace.project(ROBOFLOW_BALL_DATASET)
            ball_dataset = (
                ball_project.version(2)
                .download("yolov12", location=routes.DATASETS_DIR.as_posix(), overwrite=True))
            player_project = workspace.project(ROBOFLOW_PLAYER_DATASET)
            player_dataset = (
                player_project.version(2)
                .download("yolov12", location=routes.DATASETS_DIR.as_posix(), overwrite=True))

            ball_yaml = Path(ball_dataset.location, "data.yaml")
            player_yaml = Path(player_dataset.location, "data.yaml")
            player_data, ball_data = self.prepare_custom_dataset()

            self._merge_datasets(
                custom_data=ball_data,
                original_dataset=Path(ball_dataset.location))
            self._merge_datasets(
                custom_data=player_data,
                original_dataset=Path(player_dataset.location))

            ball_model = self._train_model(
                dataset_yaml=ball_yaml.as_posix(),
                model_name="ball",
                model_path=(routes.MODELS_DIR / BALL_MODEL_NAME).as_posix())

            player_model = self._train_model(
                dataset_yaml=player_yaml.as_posix(),
                model_name="player",
                model_path=(routes.MODELS_DIR / PLAYER_MODEL_NAME).as_posix())

            if not ball_model or not player_model:
                return False, False

            player_promoted = self._evaluate_and_promote(
                new_model_path=player_model,
                original_model_path=(routes.MODELS_DIR / PLAYER_MODEL_NAME).as_posix(),
                model_yaml=player_yaml.as_posix(),
            )
            
            ball_promoted = self._evaluate_and_promote(
                new_model_path=ball_model,
                original_model_path=(routes.MODELS_DIR / BALL_MODEL_NAME).as_posix(),
                model_yaml=ball_yaml.as_posix(),
            )

            elapsed = time.time() - start_time

            self._log_event("player cycle_complete", {
                "promoted": player_promoted,
                "elapsed_seconds": elapsed,
                "tests_count": len(list(Path(player_dataset.location, "test").glob("*.jpg")))
            })
            self._log_event("ball cycle_complete", {
                "promoted": ball_promoted,
                "elapsed_seconds": elapsed,
                "tests_count": len(list(Path(ball_dataset.location, "test").glob("*.jpg")))
            })

            self._archive_training_data(images=player_data, project=player_project)
            self._archive_training_data(ball_data, ball_project)

            return player_promoted, ball_promoted
        except Exception as e:
            self.logger.exception(f"Error en ciclo de reentrenamiento: {e}")
            self._log_event("cycle_failed", {"error": str(e)})
            return False, False

    def _train_model(self, dataset_yaml: str, model_name: str, model_path: str) -> Optional[str]:
        """Entrena el modelo challenger con fine-tuning sobre el champion actual."""
        try:
            self.logger.info("Cargando champion para fine-tuning...")
            model = YOLO(model_path)
            training_name = f"run_{model_name}_{int(time.time())}"

            self.logger.info("Entrenando challenger...")
            model.train(
                data=dataset_yaml,
                epochs=30,
                imgsz=1024,
                batch=-1,
                freeze=30,
                device="cuda" if torch.cuda.is_available() else "cpu",
                exist_ok=True,
                project=routes.CUSTOM_MODELS,
                name=training_name,
                lr0=0.001,
                lrf=0.1,
                warmup_epochs=3,
                weight_decay=0.001,
                augment=True,
                mixup=0.1,
                copy_paste=0.1,
            )

            # El mejor checkpoint del run
            
            best_path = routes.CUSTOM_MODELS / training_name / "weights" / "best.pt"

            if not best_path.exists():
                self.logger.error("No se encontró best.pt tras entrenamiento.")
                return None

            return best_path.as_posix()
        except Exception as e:
            self.logger.exception(f"Error entrenando challenger: {e}")
            return None

    def _evaluate_and_promote(
        self,
        new_model_path: str,
        original_model_path: str,
        model_yaml: str,
    ) -> bool:
        """
        Compara challenger vs champion en el validation set.
        Solo promueve si el challenger es estrictamente mejor
        dentro del margen de regresión aceptado.
        """
        self.logger.info("Evaluando modelo original...")
        original_model_metrics = self._evaluate_model(
            original_model_path, model_yaml, label="original"
        )

        self.logger.info("Evaluando modelo nuevo...")
        new_model_metrics = self._evaluate_model(
            new_model_path, model_yaml, label="new"
        )

        if not original_model_metrics or not new_model_metrics:
            self.logger.error("No se pudieron obtener métricas. Abortando promoción.")
            return False

        original_map = original_model_metrics["map50"]
        new_map = new_model_metrics["map50"]
        delta = new_map - original_map

        self.logger.info(
            f"Original mAP50: {original_map:.4f} | "
            f"Nuevo mAP50: {new_map:.4f} | "
            f"Delta: {delta:+.4f}"
        )

        meets_absolute = new_model_metrics["map50"] >= self.config.min_map50
        no_regression = delta >= -self.config.max_regression_delta
        is_better = delta > 0

        if meets_absolute and no_regression and is_better:
            self.logger.info("Modelo PROMOVIDO.")
            self._promote_model(Path(original_model_path), Path(new_model_path))
            self._log_event("promoted", {
                "original_map50": original_map,
                "new_map50": new_map,
                "delta": delta,
            })
            return True
        else:
            reasons = []
            if not meets_absolute:
                reasons.append(f"mAP50 {new_map:.3f} < mínimo {self.config.min_map50}")
            if not no_regression:
                reasons.append(f"regresión {delta:.3f} supera umbral")
            if not is_better:
                reasons.append("no supera el modelo anterior")

            self.logger.warning(f"Modelo RECHAZADO: {'; '.join(reasons)}")
            self._log_event("rejected", {
                "reasons": reasons,
                "map50": new_map,
            })
            return False

    def _evaluate_model(
        self, model_path: str, data_yaml: str, label: str
    ) -> Optional[dict]:
        try:
            model = YOLO(model_path)
            results = model.val(data=data_yaml, imgsz=1024, verbose=False)
            metrics = {
                "map50": float(results.box.map50),
                "map50_95": float(results.box.map),
                "precision": float(results.box.p.mean()),
                "recall": float(results.box.r.mean()),
            }
            self.logger.info(f"[{label}] Métricas: {metrics}")
            return metrics
        except Exception as e:
            self.logger.exception(f"Error evaluando {label}: {e}")
            return None

    def _promote_model(self, original_model_path: Path, new_model_path: Path):
        backup_name = f"{original_model_path.stem}_backup_{int(time.time())}.pt"
        backup_path = routes.MODELS_BACKUP_DIR / backup_name
        shutil.copy2(original_model_path, backup_path)
        shutil.copy2(new_model_path, original_model_path)
        self.logger.info(f"Backup en {backup_path}, nuevo modelo en {original_model_path}")

    def _merge_datasets(self, custom_data: List[TrainData], original_dataset: Path):
        labels_dir = original_dataset / "train" / "labels"
        images_dir = original_dataset / "train" / "images"
        
        for data in custom_data:
            new_image = images_dir / data.image.name
            new_label = labels_dir / data.label.name
            shutil.move(data.image.as_posix(), (images_dir / data.image.name).as_posix())
            shutil.move(data.label.as_posix(), (labels_dir / data.label.name).as_posix())
            data.image = new_image
            data.label = new_label

    def _archive_training_data(self, images: List[TrainData], project: Project):
        """Elimina los datos en las carpetas de entrenamiento personalizado y los sube a roboflow."""
        batch_name = f"retraining_{int(time.time())}"
        for data in images:
            if not data.image.exists() or data.image.suffix != ".jpg":
                self.logger.error(f"Imagen no encontrada: {data.image}")
                continue

            try:
                project.upload(
                batch_name=batch_name,
                image_path=data.image.as_posix(),
                annotation_path=data.label.as_posix() if data.label.exists() else None,
                is_prediction=True,
                num_retry_uploads=3
                )
                data.image.unlink(missing_ok=True)
                data.label.unlink(missing_ok=True)
            except Exception as e:
                self.logger.error(f"Error subiendo {data.image}: {e}")

    def _load_log(self) -> list:
        p = Path(self.config.retraining_log_path)
        return json.loads(p.read_text()) if p.exists() else []

    def _log_event(self, event_type: str, data: dict):
        entry = {"timestamp": time.time(), "event": event_type, **data}
        self._log.append(entry)
        Path(self.config.retraining_log_path).write_text(json.dumps(self._log, indent=2))

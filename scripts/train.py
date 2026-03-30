'''

from pathlib import Path

import logging
import pytorch_lightning as pl
import hydra

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from cross_view_transformer.common import setup_config, setup_experiment, load_backbone
from cross_view_transformer.callbacks.gitdiff_callback import GitDiffCallback
from cross_view_transformer.callbacks.visualization_callback import VisualizationCallback


log = logging.getLogger(__name__)

CONFIG_PATH = '/content/cross_view_transformers/config'
CONFIG_NAME = 'config.yaml'


def maybe_resume_training(experiment):
    save_dir = Path(experiment.save_dir).resolve()
    checkpoints = list(save_dir.glob(f'**/{experiment.uuid}/checkpoints/*.ckpt'))

    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None

    log.info(f'Found {checkpoints[-1]}.')

    return checkpoints[-1]


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)

    pl.seed_everything(cfg.experiment.seed, workers=True)

    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    # Create and load model/data
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # Optionally load model
    ckpt_path = maybe_resume_training(cfg.experiment)

    if ckpt_path is not None:
        model_module.backbone = load_backbone(ckpt_path)

    # Loggers and callbacks
    logger = pl.loggers.WandbLogger(project=cfg.experiment.project,
                                    save_dir=cfg.experiment.save_dir,
                                    id=cfg.experiment.uuid)

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(filename='model',
                        every_n_train_steps=cfg.experiment.checkpoint_interval),

        VisualizationCallback(viz_fn, cfg.experiment.log_image_interval),
        GitDiffCallback(cfg)
    ]

    # Train
    trainer = pl.Trainer(logger=logger,
                         callbacks=callbacks,
                         strategy=DDPStrategy(find_unused_parameters=False),
                         **cfg.trainer)
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()

'''


from pathlib import Path
import logging
import pytorch_lightning as pl
import hydra

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from cross_view_transformer.common import setup_config, setup_experiment, load_backbone
from cross_view_transformer.callbacks.gitdiff_callback import GitDiffCallback
from cross_view_transformer.callbacks.visualization_callback import VisualizationCallback

log = logging.getLogger(__name__)

# 첫 번째 코드의 경로 설정 유지
CONFIG_PATH = '/content/cross_view_transformers/config'
CONFIG_NAME = 'config.yaml'

def maybe_resume_training(experiment):
    save_dir = Path(experiment.save_dir).resolve()
    
    # 두 번째 코드의 정교한 경로 계산 (프로젝트명 포함)
    project_name = getattr(experiment, 'project', '')
    checkpoint_dir = save_dir / project_name / experiment.uuid / 'checkpoints'
    
    # 폴백(Fallback): 프로젝트 폴더가 없는 구버전 경로 대응
    if not checkpoint_dir.exists():
        checkpoint_dir = save_dir / experiment.uuid / 'checkpoints'

    log.info(f'Searching in {checkpoint_dir}')

    if not checkpoint_dir.exists():
        log.info('체크포인트 폴더가 없습니다. 처음부터 학습을 시작합니다.')
        return None

    # 1순위: last.ckpt 확인
    last_ckpt = checkpoint_dir / 'last.ckpt'
    if last_ckpt.exists():
        log.info(f'Resuming from last checkpoint: {last_ckpt}')
        return last_ckpt

    # 2순위: 가장 최근에 수정된 .ckpt 파일 찾기
    checkpoints = list(checkpoint_dir.rglob('*.ckpt'))
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x.stat().st_mtime)
    latest_ckpt = checkpoints[-1]
    
    log.info(f'Found latest checkpoint: {latest_ckpt}')
    return latest_ckpt


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # --- 체크포인트 저장 경로 설정 ---
    save_dir = Path(cfg.experiment.save_dir).resolve()
    project_name = getattr(cfg.experiment, 'project', '')
    checkpoint_dir = str(save_dir / project_name / cfg.experiment.uuid / 'checkpoints')
    
    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)

    # 모델 및 데이터 로드
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # 체크포인트 경로 탐색
    ckpt_path = maybe_resume_training(cfg.experiment)

    # 첫 번째 코드의 로직 유지: 백본 가중치 명시적 로드
    if ckpt_path is not None:
        log.info(f'Loading backbone from {ckpt_path}')
        model_module.backbone = load_backbone(ckpt_path)

    # 로거 설정
    logger = pl.loggers.WandbLogger(
        project=cfg.experiment.project,
        save_dir=cfg.experiment.save_dir,
        id=cfg.experiment.uuid
    )

    # --- 콜백 설정 (요청 사항 반영) ---
    
    # 1. Top-3 저장 (Validation Loss 기준)
    checkpoint_best = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor='val/loss', 
        filename='best-{epoch:02d}-valLoss-{val/loss:.4f}',
        auto_insert_metric_name=False,
        save_top_k=3, 
        mode='min',
        save_last=False
    )

    # 2. last.ckpt 저장 (매 설정된 step 마다 갱신)
    checkpoint_last = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=0,
        save_last=True,
        every_n_train_steps=cfg.experiment.checkpoint_interval
    )
    
    # 3. 3 Epoch 단위 인터벌 저장
    checkpoint_interval = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='interval-{epoch:02d}',
        every_n_epochs=3,
        save_top_k=-1, # 모든 인터벌 파일 유지
        monitor=None
    )

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        checkpoint_best,
        checkpoint_last,
        checkpoint_interval,
        VisualizationCallback(viz_fn, cfg.experiment.log_image_interval),
        GitDiffCallback(cfg)
    ]

    # --- 트레이너 실행 (DDP 제거) ---
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        # strategy 라인을 삭제하여 기본(Single Device) 모드로 동작하게 함
        **cfg.trainer
    )
    
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()

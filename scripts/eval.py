from pathlib import Path
import logging
import pytorch_lightning as pl
import hydra
import torch

from cross_view_transformer.common import setup_config, setup_experiment
from cross_view_transformer.callbacks.visualization_callback import VisualizationCallback

log = logging.getLogger(__name__)

# 기존 학습 코드와 동일한 설정 경로
CONFIG_PATH = '/content/cross_view_transformers/config'
CONFIG_NAME = 'config.yaml'

def get_evaluation_checkpoint(experiment):
    """
    평가를 위한 체크포인트를 탐색합니다.
    사용자 요청에 따라 '가장 최근까지 학습된(last)' 모델을 1순위로 찾습니다.
    """
    save_dir = Path(experiment.save_dir).resolve()
    project_name = getattr(experiment, 'project', '')
    checkpoint_dir = save_dir / project_name / experiment.uuid / 'checkpoints'
    
    # 폴백(Fallback): 프로젝트 폴더가 없는 구버전 경로 대응
    if not checkpoint_dir.exists():
        checkpoint_dir = save_dir / experiment.uuid / 'checkpoints'

    log.info(f'Searching for evaluation checkpoint in {checkpoint_dir}')

    if not checkpoint_dir.exists():
        log.error('❌ 체크포인트 폴더가 존재하지 않습니다.')
        return None

    # 1순위: last.ckpt (가장 최근 학습 저장본 최우선 사용)
    last_ckpt = checkpoint_dir / 'last.ckpt'
    if last_ckpt.exists():
        log.info(f'✅ 1순위 - Last 체크포인트 발견: {last_ckpt}')
        return last_ckpt

    # 2순위: best-로 시작하는 체크포인트 (Last가 없다면 Validation Loss 최저 모델 사용)
    best_checkpoints = list(checkpoint_dir.glob('best-*.ckpt'))
    if best_checkpoints:
        # 여러 개일 경우 가장 최근에 수정된 파일 선택
        best_checkpoints.sort(key=lambda x: x.stat().st_mtime)
        best_ckpt = best_checkpoints[-1]
        log.info(f'✅ 2순위 - Best 체크포인트 발견: {best_ckpt}')
        return best_ckpt

    # 3순위: 그 외 가장 최근에 수정된 .ckpt 파일
    all_checkpoints = list(checkpoint_dir.rglob('*.ckpt'))
    if all_checkpoints:
        all_checkpoints.sort(key=lambda x: x.stat().st_mtime)
        latest_ckpt = all_checkpoints[-1]
        log.info(f'✅ 3순위 - 일반 최신 체크포인트 발견: {latest_ckpt}')
        return latest_ckpt

    log.error('❌ 사용 가능한 체크포인트(.ckpt) 파일이 없습니다.')
    return None


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)
    
    # 평가의 일관성과 재현성을 위해 시드 고정
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # 모델, 데이터셋, 시각화 함수 로드
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # 평가할 체크포인트 경로 가져오기
    ckpt_path = get_evaluation_checkpoint(cfg.experiment)
    
    if ckpt_path is None:
        log.error("평가를 진행할 수 없어 프로그램을 종료합니다.")
        return

    # Wandb 로거 설정 (평가용임을 명시하기 위해 name에 eval 접두사 추가 권장)
    logger = pl.loggers.WandbLogger(
        project=cfg.experiment.project,
        save_dir=cfg.experiment.save_dir,
        id=cfg.experiment.uuid,
        name=f"eval_{cfg.experiment.uuid}"
    )

    # 평가용 콜백 설정 (학습용 콜백은 모두 제거하고 시각화만 남김)
    callbacks = [
        VisualizationCallback(viz_fn, cfg.experiment.log_image_interval)
    ]

    # 트레이너 설정
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=cfg.trainer.get('precision', 32)
    )

    log.info(f"🚀 Evaluation(검증) 시작 (사용 체크포인트: {ckpt_path})")
    
    # 학습이 아니므로 fit 대신 validate를 사용하며, 체크포인트 가중치를 로드합니다.
    trainer.validate(model=model_module, datamodule=data_module, ckpt_path=str(ckpt_path))

    # 만약 test 데이터셋이 따로 구성되어 있다면 아래 코드를 사용할 수도 있습니다.
    # trainer.test(model=model_module, datamodule=data_module, ckpt_path=str(ckpt_path))


if __name__ == '__main__':
    main()

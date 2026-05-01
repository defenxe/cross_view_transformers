import torch
import pytorch_lightning as pl
import time  # 시간 측정을 위해 추가


class ModelModule(pl.LightningModule):
    def __init__(self, backbone, loss_func, metrics, optimizer_args, scheduler_args=None, cfg=None):
        super().__init__()

        self.save_hyperparameters(
            cfg,
            ignore=['backbone', 'loss_func', 'metrics', 'optimizer_args', 'scheduler_args'])

        self.backbone = backbone
        self.loss_func = loss_func
        self.metrics = metrics

        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args

    def forward(self, batch):
        return self.backbone(batch)

    '''
    def shared_step(self, batch, prefix='', on_step=False, return_output=True):
        pred = self(batch)
        loss, loss_details = self.loss_func(pred, batch)

        self.metrics.update(pred, batch)

        if self.trainer is not None:
            self.log(f'{prefix}/loss', loss.detach(), on_step=on_step, on_epoch=True)
            self.log_dict({f'{prefix}/loss/{k}': v.detach() for k, v in loss_details.items()},
                          on_step=on_step, on_epoch=True)

        # Used for visualizations
        if return_output:
            return {'loss': loss, 'batch': batch, 'pred': pred}

        return {'loss': loss}
    '''
    
    def shared_step(self, batch, prefix='', on_step=False, return_output=True, pred=None):
        #batch에 epoch 데이터 같이 보내기
        batch['current_epoch'] = self.current_epoch
        
        if pred is None:
            pred = self(batch)

        
        
        loss, loss_details = self.loss_func(pred, batch)

        self.metrics.update(pred, batch)

        if self.trainer is not None:
            self.log(f'{prefix}/loss', loss.detach(), on_step=on_step, on_epoch=True)
            self.log_dict({f'{prefix}/loss/{k}': v.detach() for k, v in loss_details.items()},
                          on_step=on_step, on_epoch=True)

        # Used for visualizations
        if return_output:
            return {'loss': loss, 'batch': batch, 'pred': pred}

        return {'loss': loss}


    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train', True,
                                batch_idx % self.hparams.experiment.log_image_interval == 0)

    '''
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val', False,
                                batch_idx % self.hparams.experiment.log_image_interval == 0)
    '''
    def validation_step(self, batch, batch_idx):
        # Warm-up: 첫 5개 배치는 측정에서 제외 (왜곡 방지)
        if batch_idx < 5:
            return self.shared_step(batch, prefix='val', on_step=False)
        
        # 1. GPU 동기화 및 시작 시간 기록
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # 2. 순수 추론(Inference) 진행
        # shared_step 대신 직접 self(batch)를 호출하여 손실 계산 시간 등을 제외합니다.
        batch['current_epoch'] = self.current_epoch
        pred = self(batch)

        # 3. GPU 동기화 및 종료 시간 기록
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        # 4. 소요 시간 저장 (초 단위 -> 밀리초 단위로 저장 추천)
        duration = (end_time - start_time) * 1000 
        self.inference_times.append(duration)

        # 나머지 검증 로직(Loss, Metrics)은 기존 shared_step 활용
        # 시간을 잰 뒤에 loss를 계산해야 추론 시간만 깔끔하게 측정됩니다.
        output = self.shared_step(batch, 'val', False, 
                                  batch_idx % self.hparams.experiment.log_image_interval == 0, pred=pred)
        return output

    def on_validation_start(self) -> None:
        self._log_epoch_metrics('train')
        self._enable_dataloader_shuffle(self.trainer.val_dataloaders)

    '''
    def validation_epoch_end(self, outputs):
        self._log_epoch_metrics('val')

    '''

    def validation_epoch_end(self, outputs):
        # 5. 평균 추론 시간 계산 및 로깅
        if self.inference_times:
            avg_inference_time = sum(self.inference_times) / len(self.inference_times)
            self.log('val/avg_inference_time_ms', avg_inference_time, prog_bar=True)
            print(f"\n[Epoch {self.current_epoch}] Average Inference Time: {avg_inference_time:.2f} ms/batch")
            
            # 다음 에포크를 위해 리스트 초기화
            self.inference_times.clear()

        self._log_epoch_metrics('val')

    def _log_epoch_metrics(self, prefix: str):
        """
        lightning is a little odd - it goes

        on_train_start
        ... does all the training steps ...
        on_validation_start
        ... does all the validation steps ...
        on_validation_epoch_end
        on_train_epoch_end
        """
        metrics = self.metrics.compute()

        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, val in value.items():
                    self.log(f'{prefix}/metrics/{key}{subkey}', val)
            else:
                self.log(f'{prefix}/metrics/{key}', value)

        self.metrics.reset()

    def _enable_dataloader_shuffle(self, dataloaders):
        """
        HACK for https://github.com/PyTorchLightning/pytorch-lightning/issues/11054
        """
        for v in dataloaders:
            v.sampler.shuffle = True
            v.sampler.set_epoch(self.current_epoch)

    def configure_optimizers(self, disable_scheduler=False):
        parameters = [x for x in self.backbone.parameters() if x.requires_grad]
        optimizer = torch.optim.AdamW(parameters, **self.optimizer_args)

        if disable_scheduler or self.scheduler_args is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda lr: 1)
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **self.scheduler_args)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

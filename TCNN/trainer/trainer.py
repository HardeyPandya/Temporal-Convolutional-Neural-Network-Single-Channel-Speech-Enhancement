import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.base_trainer import BaseTrainer
from util.metrics import compute_PESQ, compute_STOI
from util.utils import synthesis_noisy_y,sliceframe, OverlapAndAdd, compLossMask

plt.switch_backend("agg")


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            optimizer,
            loss_function,
            train_dl,
            validation_dl,
    ):
        super(Trainer, self).__init__(config, resume, model, optimizer, loss_function)
        self.train_dataloader = train_dl
        self.validation_dataloader = validation_dl

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for noisy_mag, clean_mag, n_frames_list in self.train_dataloader:
            self.optimizer.zero_grad()

            noisy_mag = noisy_mag.float().to(self.device)
            
            clean_mag = clean_mag.float().to(self.device)
            pred_clean = self.model(noisy_mag)
           
            loss_mask = compLossMask(clean_mag,n_frames_list)
            
            loss = self.loss_function(outputs=pred_clean, labels=clean_mag, loss_mask=loss_mask, nframes=n_frames_list)
          
            loss_total += loss
            print("Loss: ")
            #print(noisy_mag.shape)
            #print(pred_clean.shape)
            print(loss_total)
            loss.backward()
            self.optimizer.step()

        self.writer.add_scalar("Loss/Train", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
       
        stoi_c_n = []
        stoi_c_d = []
        pesq_c_n = []
        pesq_c_d = []
        print("hello")
        for i, (noisy_y, clean_y, name) in enumerate(self.validation_dataloader):
            assert len(name) == 1, "The batch size of validation dataloader must be 1."
            name = name[0]
            noisy_y_np = noisy_y
            noisy_y=sliceframe(noisy_y)
            
            noisy_y=torch.tensor(noisy_y)
            noisy_y=noisy_y.float().to(self.device)
            noisy_y = noisy_y[None,:,:]


            
            clean_y=clean_y.float()
            
            # noisy_y = noisy_y.reshape(-1)
            clean_y = clean_y.reshape(-1)
            
            print("pred_clean: ")
            print(noisy_y.shape)
            pred_clean = self.model(noisy_y)
           
            pred_clean_y = OverlapAndAdd(pred_clean.cpu().numpy(),160)
            noisy_y_np = noisy_y_np.numpy().reshape(-1)
            clean_y = clean_y.numpy().reshape(-1)
            pred_clean_y = pred_clean_y.reshape(-1)
            min_len = min(len(noisy_y_np), len(pred_clean_y), len(clean_y))
            noisy_y_np = noisy_y_np[:min_len]
            pred_clean_y = pred_clean_y[:min_len]
            clean_y = clean_y[:min_len]

           
            
            # Metrics
            stoi_c_n.append(compute_STOI(clean_y, noisy_y_np, sr=16000))
            stoi_c_d.append(compute_STOI(clean_y, pred_clean_y, sr=16000))
            
            pesq_c_n.append(compute_PESQ(clean_y, noisy_y_np, sr=16000))
            pesq_c_d.append(compute_PESQ(clean_y, pred_clean_y, sr=16000))

        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.writer.add_scalars(f"Metrics/STOI", {
            "clean and noisy": get_metrics_ave(stoi_c_n),
            "clean and denoisy": get_metrics_ave(stoi_c_d)
        }, epoch)
        self.writer.add_scalars(f"Metrics/PESQ", {
            "clean and noisy": get_metrics_ave(pesq_c_n),
            "clean and denoisy": get_metrics_ave(pesq_c_d)
        }, epoch)

        score = (get_metrics_ave(stoi_c_d) + self._transform_pesq_range(get_metrics_ave(pesq_c_d))) / 2
        return score

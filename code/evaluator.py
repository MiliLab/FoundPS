import numpy as np   
from scipy import ndimage
import cv2
import math
from scipy.signal import convolve2d
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity
from scipy import fft as spfft
import os
import json
import scipy
from skimage.transform.integral import integral_image as integral
from math import ceil, floor, log2
 
class Evaluator:
    def __init__(self, ratio=4, GNyq=0.15, N=41, kaiser_beta=0.5):
        self.filter = self._get_filter(ratio=ratio, GNyq=GNyq, N=N, beta=kaiser_beta)
        self._PHASECONG_CACHE = {}

    # ---------- MTF PAN ----------
    def _gaussian_fspecial(self, N: int, sigma: float) -> np.ndarray:
        if N % 2 == 0:
            raise ValueError("N must be odd.")
        ax = np.arange(-(N // 2), N // 2 + 1, dtype=np.float64)
        xx, yy = np.meshgrid(ax, ax, indexing="xy")
        H = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        H /= H.sum()
        return H

    def _fwind1_2d(self, Hd: np.ndarray, beta: float = 0.5) -> np.ndarray:
        N = Hd.shape[0]
        if Hd.shape[0] != Hd.shape[1] or N % 2 == 0:
            raise ValueError("Hd must be square and N must be odd.")

        h = np.fft.ifft2(np.fft.ifftshift(Hd))
        h = np.real(np.fft.fftshift(h))

        w = np.kaiser(N, beta)
        h *= np.outer(w, w)

        s = h.sum()
        if s != 0:
            h /= s
        return h

    def _get_filter(self, ratio=4, GNyq=0.15, N=41, beta=0.5):
        fcut = 1.0 / float(ratio)
        alpha = np.sqrt((N * (fcut / 2.0)) ** 2 / (-2.0 * np.log(GNyq)))
        H = self._gaussian_fspecial(N, alpha)
        Hd = H / np.max(H)
        return self._fwind1_2d(Hd, beta=beta)

    def MTF_PAN(self,image_pan):
        pan = np.pad(image_pan,((20,20),(20,20)),mode='edge')
        image_pan_filter = scipy.signal.correlate2d(pan,self.filter,mode='valid')
        pan_filter = image_pan_filter # np.round(image_pan_filter + 0.5)
        return pan_filter

    def local_cross_correlation(self, img_1, img_2, half_width): 

        w = int(half_width)
        ep = 1e-20

        if (len(img_1.shape)) != 3:
            img_1 = np.expand_dims(img_1, axis=-1)
        if (len(img_2.shape)) != 3:
            img_2 = np.expand_dims(img_2, axis=-1)

        img_1 = np.pad(img_1.astype(np.float64), ((w, w), (w, w), (0, 0)))
        img_2 = np.pad(img_2.astype(np.float64), ((w, w), (w, w), (0, 0)))

        img_1_cum = np.zeros(img_1.shape)
        img_2_cum = np.zeros(img_2.shape)
        for i in range(img_1.shape[-1]):
            img_1_cum[:, :, i] = integral(img_1[:, :, i]).astype(np.float64)
        for i in range(img_2.shape[-1]):
            img_2_cum[:, :, i] = integral(img_2[:, :, i]).astype(np.float64)

        img_1_mu = (img_1_cum[2 * w:, 2 * w:, :] - img_1_cum[:-2 * w, 2 * w:, :] - img_1_cum[2 * w:, :-2 * w, :]
                    + img_1_cum[:-2 * w, :-2 * w, :]) / (4 * w ** 2)
        img_2_mu = (img_2_cum[2 * w:, 2 * w:, :] - img_2_cum[:-2 * w, 2 * w:, :] - img_2_cum[2 * w:, :-2 * w, :]
                    + img_2_cum[:-2 * w, :-2 * w, :]) / (4 * w ** 2)

        img_1 = img_1[w:-w, w:-w, :] - img_1_mu
        img_2 = img_2[w:-w, w:-w, :] - img_2_mu

        img_1 = np.pad(img_1.astype(np.float64), ((w, w), (w, w), (0, 0)))
        img_2 = np.pad(img_2.astype(np.float64), ((w, w), (w, w), (0, 0)))

        i2 = img_1 ** 2
        j2 = img_2 ** 2
        ij = img_1 * img_2

        i2_cum = np.zeros(i2.shape)
        j2_cum = np.zeros(j2.shape)
        ij_cum = np.zeros(ij.shape)

        for i in range(i2_cum.shape[-1]):
            i2_cum[:, :, i] = integral(i2[:, :, i]).astype(np.float64)
        for i in range(j2_cum.shape[-1]):
            j2_cum[:, :, i] = integral(j2[:, :, i]).astype(np.float64)
        for i in range(ij_cum.shape[-1]):
            ij_cum[:, :, i] = integral(ij[:, :, i]).astype(np.float64)

        sig2_ij_tot = (ij_cum[2 * w:, 2 * w:, :] - ij_cum[:-2 * w, 2 * w:, :] - ij_cum[2 * w:, :-2 * w, :]
                    + ij_cum[:-2 * w, :-2 * w, :])
        sig2_ii_tot = (i2_cum[2 * w:, 2 * w:, :] - i2_cum[:-2 * w, 2 * w:, :] - i2_cum[2 * w:, :-2 * w, :]
                    + i2_cum[:-2 * w, :-2 * w, :])
        sig2_jj_tot = (j2_cum[2 * w:, 2 * w:, :] - j2_cum[:-2 * w, 2 * w:, :] - j2_cum[2 * w:, :-2 * w, :]
                    + j2_cum[:-2 * w, :-2 * w, :])

        sig2_ii_tot = np.clip(sig2_ii_tot, ep, sig2_ii_tot.max())
        sig2_jj_tot = np.clip(sig2_jj_tot, ep, sig2_jj_tot.max())

        xcorr = sig2_ij_tot / ((sig2_ii_tot * sig2_jj_tot) ** 0.5 + ep)

        return xcorr


    # ---------- block utils ----------
    @staticmethod
    def _block_view_2d(a: np.ndarray, S: int) -> np.ndarray: 
        H, W = a.shape
        H2 = (H // S) * S
        W2 = (W // S) * S
        a = a[:H2, :W2]
        nH, nW = H2 // S, W2 // S 
        return a.reshape(nH, S, nW, S).transpose(0, 2, 1, 3)

    @staticmethod
    def _uqi_blocks(xb: np.ndarray, yb: np.ndarray, eps: float = 1e-24) -> np.ndarray: 
        xb = xb.astype(np.float64, copy=False)
        yb = yb.astype(np.float64, copy=False)

        mu_x = xb.mean(axis=(-1, -2))
        mu_y = yb.mean(axis=(-1, -2))

        dx = xb - mu_x[..., None, None]
        dy = yb - mu_y[..., None, None]
 
        n = xb.shape[-1] * xb.shape[-2]
        denom = (n - 1) if n > 1 else 1.0

        C11 = (dx * dx).sum(axis=(-1, -2)) / denom
        C22 = (dy * dy).sum(axis=(-1, -2)) / denom
        C12 = (dx * dy).sum(axis=(-1, -2)) / denom

        Q = (4.0 * C12 * mu_x * mu_y) / (C11 + C22 + eps) / (mu_x * mu_x + mu_y * mu_y + eps)
        return Q

    # ---------- D_s and D_lambda ----------
    def D_s(self, fusion: np.ndarray, ms: np.ndarray, pan: np.ndarray, S: int, q: float) -> float:
        fusion = np.asarray(fusion, dtype=np.float64)
        ms = np.asarray(ms, dtype=np.float64)
        pan = np.asarray(pan, dtype=np.float64)

        H, W, C = fusion.shape
        pan_filt = self.MTF_PAN(pan)

        pan_blk = self._block_view_2d(pan, S)
        panf_blk = self._block_view_2d(pan_filt, S)

        acc = 0.0
        for i in range(C):
            f_blk = self._block_view_2d(fusion[:, :, i], S)
            m_blk = self._block_view_2d(ms[:, :, i], S)

            Q_high = self._uqi_blocks(f_blk, pan_blk).mean()
            Q_low  = self._uqi_blocks(m_blk, panf_blk).mean()

            acc += np.abs(Q_high - Q_low) ** q

        return (acc / C) ** (1.0 / q)

    def D_lambda(self, fusion: np.ndarray, ms: np.ndarray, S: int, p: float) -> float:
        fusion = np.asarray(fusion, dtype=np.float64)
        ms = np.asarray(ms, dtype=np.float64)
        H, W, C = fusion.shape 
        ms_blk = [self._block_view_2d(ms[:, :, i], S) for i in range(C)]
        fu_blk = [self._block_view_2d(fusion[:, :, i], S) for i in range(C)]

        acc = 0.0
        cnt = 0
        for i in range(C - 1):
            for j in range(i + 1, C):
                Q_exp = self._uqi_blocks(ms_blk[i], ms_blk[j]).mean()
                Q_fus = self._uqi_blocks(fu_blk[i], fu_blk[j]).mean()
                acc += np.abs(Q_fus - Q_exp) ** p
                cnt += 1

        return (acc / cnt) ** (1.0 / p)

    # ---------- QNR ----------
    def QNR(self, fusion, ms, pan, S=32, p=1, q=1, alpha=1, beta=1, data_range = 65535.0):
        fusion = np.asarray(fusion, dtype=np.float64) / data_range
        pan = np.asarray(pan, dtype=np.float64) / data_range
        ms = np.asarray(ms, dtype=np.float64) / data_range

        H, W, C = fusion.shape

        # 关键：先 float，再 bicubic，避免 uint 行为差异
        ms_up = cv2.resize(ms, (W, H), interpolation=cv2.INTER_CUBIC)

        Dl = self.D_lambda(fusion, ms_up, S, p)
        Ds = self.D_s(fusion, ms_up, pan, S, q)

        # clamp：防止极小数值误差导致 (1-D)<0
        one_minus_Dl = max(0.0, 1.0 - Dl)
        one_minus_Ds = max(0.0, 1.0 - Ds)
        qnr = (one_minus_Dl ** alpha) * (one_minus_Ds ** beta)

        return Dl, Ds, qnr

    # # ---------- QAVE ----------
    def QAVE(self,I_ms,I_f, data_range=65535.0):
        f, ms = I_f.astype(np.float64) / data_range, I_ms.astype(np.float64) / data_range
        h, w, c = f.shape
        ms_mean = np.mean(ms,axis=-1)
        f_mean = np.mean(f,axis=-1)
        Qx_sum,Qy_sum,Qxy_sum = 0,0,0
        for i in range(c):
            M = ms[:,:,i] - ms_mean
            Qx_sum = Qx_sum + np.power(M,2) 
            F = f[:, :, i] - f_mean
            Qy_sum = Qy_sum + np.power(F,2)
            Qxy_sum = Qxy_sum + M * F 
        Qx = (1/(c - 1)) * Qx_sum
        Qy = (1/(c - 1)) * Qy_sum
        Qxy = (1/(c - 1)) * Qxy_sum
        Q = (4 * Qxy * ms_mean * f_mean) / ( (Qx + Qy) * ( np.power(ms_mean,2) + np.power(f_mean,2) ) + 2.2204e-16)
        qave = np.sum( Q) / h / w
        return qave

    # # ---------- PSNR ----------
    def cal_psnr(self,img_ref, img_gen, data_range = 65535.0):
        img_gen = np.asarray(img_gen, dtype=np.float64) /data_range
        img_ref = np.asarray(img_ref, dtype=np.float64) /data_range       
        mse = np.mean((img_ref - img_gen) ** 2)
        if mse < 1.0e-10:
            return 100
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    # # ---------- SSIM ----------
    def cal_ssim(self,img_ref, img_gen,data_range):
        img_gen = np.asarray(img_gen, dtype=np.float64) 
        img_ref = np.asarray(img_ref, dtype=np.float64) 
        ssim_val = 0
        for i in range(img_ref.shape[-1]):
            ssim_val = ssim_val + structural_similarity(img_ref[:,:,i], img_gen[:,:,i], data_range=data_range)
        return ssim_val/img_ref.shape[-1]
    

    # # ---------- ERGAS ----------
    def ERGAS(self, I_ms, I_f, eps=1e-12): 
        f = I_f.astype(np.float64)
        ms = I_ms.astype(np.float64) 
        mean_f = f.mean(axis=(0, 1))  
        rmse = np.sqrt(((ms - f) ** 2).mean(axis=(0, 1))) 
        ergas = 25.0 * np.sqrt(np.mean((rmse / (mean_f + eps)) ** 2))
        return ergas

    # # ---------- RMSE ----------
    def RMSE(self,I_ms,I_f):
        f, ms = I_f.astype(np.float64), I_ms.astype(np.float64)
        if len(I_f.shape) == 2:
            h, w = f.shape
            c = 1
        else:
            h, w, c = f.shape
            pass
        D = np.power(ms - f,2)
        rmse = np.sqrt(np.sum(D)/h/w/c)
        return rmse

    # # ---------- RASE ----------
    def RASE(self,I_ms,I_f, eps=1e-12):
        ms = I_ms.astype(np.float64)
        f  = I_f.astype(np.float64)
        mse = ((ms - f) ** 2).mean(axis=(0, 1))  
        mse_mean = mse.mean() 
        mu_ms = ms.mean()
        rase = 100.0 * np.sqrt(mse_mean) / (mu_ms + eps)
        return rase

    # # ---------- SAM ----------
    def SAM(self, I1: np.ndarray, I2: np.ndarray, eps: float = 1e-12): 

        I1 = np.asarray(I1, dtype=np.float64)
        I2 = np.asarray(I2, dtype=np.float64)
        prod_scal = np.sum(I1 * I2, axis=2)   
        norm_orig = np.sum(I1 * I1, axis=2)
        norm_fusa = np.sum(I2 * I2, axis=2)

        prod_norm = np.sqrt(norm_orig * norm_fusa)   
 
        prod_map = np.maximum(prod_norm, eps)
        cos_map = prod_scal / prod_map
        cos_map = np.clip(cos_map, -1.0, 1.0)      
 
        prod_scal_flat = prod_scal.reshape(-1)
        prod_norm_flat = prod_norm.reshape(-1)

        valid = prod_norm_flat > 0    
        angles = np.arccos(
            np.clip(
                prod_scal_flat[valid] / prod_norm_flat[valid],
                -1.0, 1.0
            )
        )

        SAM_index = np.mean(angles) * 180.0 / np.pi

        return SAM_index
    
    def DRho(self,fusion, pan, sigma=4, data_range = 65535.0):
        fusion = np.asarray(fusion, dtype=np.float64) / data_range
        pan = np.asarray(pan, dtype=np.float64) / data_range
        half_width = ceil(sigma / 2)
        rho = np.clip(self.local_cross_correlation(fusion, pan, half_width), a_min=-1.0, a_max=1.0)
        d_rho = 1.0 - rho
        return np.mean(d_rho).item()

    # # ---------- METRICS ----------
    def _get_metrics(self,I_f,I_gt,I_full,I_ms,I_pan,data_range): 
        Dl, Ds, qnr = self.QNR(I_full,I_ms,I_pan,data_range = data_range)
        metric_dict = {
            'psnr' : self.cal_psnr(I_gt,I_f,data_range = data_range),
            'ssim' : self.cal_ssim(I_gt,I_f,data_range = data_range),
            'ergas' : self.ERGAS(I_gt,I_f),
            'qave' : self.QAVE(I_gt,I_f),
            'rmse' : self.RMSE(I_gt,I_f),
            'sam' : self.SAM(I_gt,I_f),
            'd_lambda' : Dl,
            'd_s' : Ds,
            'qnr' : qnr,
            'd_rho' : self.DRho(I_full,I_pan,data_range = data_range)
        }

        return metric_dict

        
if __name__ == '__main__':
    print('hello world') 
    
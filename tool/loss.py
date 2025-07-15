import torch
import torch.nn as nn
import numpy as np

def compute_weights(y_true):
    y_np = y_true.cpu().detach().numpy().squeeze()
    labels = np.empty_like(y_np, dtype=int)
    thr1 = np.log1p(5)
    thr2 = np.log1p(25)
    thr3 = np.log1p(50)
    labels[y_np < thr1] = 1
    labels[(y_np >= thr1) & (y_np < thr2)] = 2
    labels[(y_np >= thr2) & (y_np < thr3)] = 3
    labels[y_np >= thr3] = 4
    # Extreme: 0.0040%, Heavy: 0.0173%, Moderate: 0.7135%, Weak: 99.2653%
    fixed_weights = {
        1: 0.2,  
        2: 30.0,
        3: 2500.0,
        4: 20000.0
    }
    sample_weights = np.empty_like(labels, dtype=float)
    for label, weight in fixed_weights.items():
        sample_weights[labels == label] = weight
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32, device=y_true.device)
    return sample_weights.unsqueeze(1)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class QuantizedRMSELoss(nn.Module):
    """
    RMSE ponderado por faixas (bins) de precipitação.
    Calcula o RMSE dentro de cada faixa e faz média ponderada.
    
    Args:
        bins: Lista de limites das faixas. Se None, usa faixas meteorológicas padrão.
        weighted: Se True, pondera pelo inverso da frequência de cada faixa.
        use_log_scale: Se True, assume que os dados estão em escala log1p.
        aggressive_weights: Se True, usa pesos extremamente agressivos para classes raras.
    """
    def __init__(self, bins=None, weighted=True, use_log_scale=True, aggressive_weights=False):
        super().__init__()
        
        if bins is None:
            if use_log_scale:
                # Bins adequados para dados em escala log1p
                # Convertendo limites meteorológicos: 0, 5, 25, 50, 100 mm
                self.bins = [
                    0,  # log1p(0) = 0
                    torch.log1p(torch.tensor(5.0)).item(),   # ~1.79
                    torch.log1p(torch.tensor(25.0)).item(),  # ~3.26
                    torch.log1p(torch.tensor(50.0)).item(),  # ~3.93
                    torch.log1p(torch.tensor(100.0)).item()  # ~4.61
                ]
            else:
                # Bins para dados em escala original (mm)
                self.bins = [0, 5, 25, 50, 100]
        else:
            self.bins = bins
            
        self.weighted = weighted
        self.use_log_scale = use_log_scale
        self.aggressive_weights = aggressive_weights
        
        # Pesos baseados na distribuição real dos dados
        # [0-5]: 151926794, [5-25]: 1213226, [25-50]: 15008, [50+]: 177
        if aggressive_weights:
            # Pesos extremamente agressivos para forçar aprendizado dos extremos
            self.extreme_weights = torch.tensor([
                1.0,      # Classe 0 (0-5mm): peso normal
                125.0,    # Classe 1 (5-25mm): 125x mais importante
                10000.0,  # Classe 2 (25-50mm): 10000x mais importante  
                850000.0  # Classe 3 (50+mm): 850000x mais importante
            ], dtype=torch.float32)
            print("🔥 USANDO PESOS EXTREMAMENTE AGRESSIVOS:")
            print(f"   Classe 0 (0-5mm):   {self.extreme_weights[0]:.0f}x")
            print(f"   Classe 1 (5-25mm):  {self.extreme_weights[1]:.0f}x") 
            print(f"   Classe 2 (25-50mm): {self.extreme_weights[2]:.0f}x")
            print(f"   Classe 3 (50+mm):   {self.extreme_weights[3]:.0f}x")

    def forward(self, y_pred, y_true):
        """
        Calcula RMSE ponderado por faixas de precipitação.
        
        Args:
            y_pred: Predições (batch, channels, T, H, W) - pode ter múltiplos canais
            y_true: Valores reais (batch, 1, T, H, W) - sempre 1 canal
        """
        # Se y_pred tem múltiplos canais, pegar apenas o último (ou o canal de precipitação)
        if y_pred.shape[1] != y_true.shape[1]:
            if y_pred.shape[1] > 1:
                # Assumir que o último canal é a precipitação (convenção comum)
                if len(y_pred.shape) == 5:  # (batch, channels, T, H, W)
                    y_pred = y_pred[:, -1:, :, :, :]  # Pegar último canal, manter dimensão
                elif len(y_pred.shape) == 4:  # (batch, channels, H, W)
                    y_pred = y_pred[:, -1:, :, :]  # Pegar último canal, manter dimensão
                else:
                    raise ValueError(f"Unsupported tensor shape: {y_pred.shape}")
                
                # Warning só na primeira vez para não poluir o log
                if not hasattr(self, '_warned_channels'):
                    print(f"Warning: Model output has {y_pred.shape[1]} channels, using last channel for precipitation.")
                    self._warned_channels = True
            else:
                raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}")
        
        # Verificar se as dimensões restantes são compatíveis
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch after channel selection: y_pred {y_pred.shape} vs y_true {y_true.shape}")
        
        # Flatten para processamento - usando reshape para garantir contiguidade
        y_pred_flat = y_pred.reshape(-1)
        y_true_flat = y_true.reshape(-1)
        
        mse_per_bin = []
        weights = []
        bin_counts = []  # Para debugging

        for i in range(len(self.bins)-1):
            low, high = self.bins[i], self.bins[i+1]
            mask = (y_true_flat >= low) & (y_true_flat < high)
            count = mask.sum().item()
            
            if count > 0:
                # Calcular MSE para esta faixa
                err = (y_pred_flat[mask] - y_true_flat[mask]) ** 2
                mse = torch.mean(err)
                mse_per_bin.append(mse)
                
                # Escolher tipo de peso
                if self.aggressive_weights:
                    # Usar pesos extremos pré-definidos
                    weight = self.extreme_weights[i].item()
                elif self.weighted:
                    # Peso baseado no inverso da frequência
                    weight = 1.0 / count
                else:
                    # Sem ponderação
                    weight = 1.0
                    
                weights.append(weight)
                bin_counts.append(count)

        # Se nenhuma faixa tem dados, retorna zero
        if len(mse_per_bin) == 0:
            return torch.tensor(0.0, device=y_true.device, requires_grad=True)

        # Combinar MSEs com pesos
        mse_per_bin = torch.stack(mse_per_bin)
        weights = torch.tensor(weights, device=y_true.device)
        
        if self.aggressive_weights:
            # Para pesos agressivos, NÃO normalizar (queremos que os extremos dominem)
            weighted_mse = (weights * mse_per_bin).sum() / len(mse_per_bin)
        else:
            # Normalizar pesos para somarem 1 (comportamento original)
            weights = weights / weights.sum()
            weighted_mse = (weights * mse_per_bin).sum()
        
        # RMSE ponderado final
        return torch.sqrt(weighted_mse + 1e-8)  # eps para estabilidade numérica
    
    def get_bin_info(self):
        """
        Retorna informações sobre as faixas configuradas.
        Útil para debugging e análise.
        """
        if self.use_log_scale:
            original_limits = [torch.expm1(torch.tensor(b)).item() if b > 0 else 0 for b in self.bins]
            print("Faixas em escala log1p:")
            for i in range(len(self.bins)-1):
                print(f"  Faixa {i}: {self.bins[i]:.2f} - {self.bins[i+1]:.2f} (log1p)")
            print("Equivalente em mm/h:")
            for i in range(len(original_limits)-1):
                print(f"  Faixa {i}: {original_limits[i]:.1f} - {original_limits[i+1]:.1f} mm/h")
        else:
            print("Faixas em escala original (mm/h):")
            for i in range(len(self.bins)-1):
                print(f"  Faixa {i}: {self.bins[i]:.1f} - {self.bins[i+1]:.1f} mm/h")
        
        print(f"Ponderação ativa: {self.weighted}")

class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        weights = compute_weights(y_true)
        # loss = torch.mean(torch.abs(y_true - y_pred))
        # print(f"y_pred.shape: {y_pred.shape}")
        # print(f"y_true.shape: {y_true.shape}")
        # print(f"weights.shape: {weights.shape}")
        # exit(0)
        # return loss
        return torch.sum(weights * torch.abs(y_true - y_pred)) / torch.sum(weights)
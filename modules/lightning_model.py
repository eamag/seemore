import lightning as L
from .vision_language_model import VisionLanguageModel
import torch


class LitVisionLanguageModel(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 128,
        image_embed_dim: int = 512,
        n_layer: int = 8,
        img_size: int = 96,
        patch_size: int = 16,
        n_head: int = 8,
        num_blks: int = 3,
        emb_dropout: float = 0.1,
        blk_dropout: float = 0.1,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.model = VisionLanguageModel(
            n_embd,
            image_embed_dim,
            vocab_size,
            n_layer,
            img_size,
            patch_size,
            n_head,
            num_blks,
            emb_dropout,
            blk_dropout,
        )
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        images, idx, targets = batch
        logits, loss = self.model(images, idx, targets)
        mlf_logger = self.logger
        mlf_logger.log_metrics({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        images, idx, targets = batch
        logits, loss = self.model(images, idx, targets)
        mlf_logger = self.logger
        mlf_logger.log_metrics({"val_loss": loss})
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

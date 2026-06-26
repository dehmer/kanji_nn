import torch
import torch.nn as nn

# ---------------------------------------------------------
# 1. HYBRIDES HOCHLEISTUNGS-MODELL (Conv1D + BiLSTM + Pooling)
# ---------------------------------------------------------
class KanjiVGModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=256, num_classes=12000):
        super(KanjiVGModel, self).__init__()

        # Lokale Feature-Extraktion (Erkennt Kanten, Winkel und präzise Kurvenverläufe)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout1d(0.1),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU()
        )

        # Bidirektionales LSTM verarbeitet die zeitliche Abfolge der Striche
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Klassifikations-Kopf mit kombiniertem Pooling
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2 * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Transpose für Conv1d: [Batch, Sequence, Features] -> [Batch, Features, Sequence]
        x = x.transpose(1, 2)
        features = self.conv_block(x)

        # Zurück für LSTM: [Batch, Sequence, Features]
        features = features.transpose(1, 2)
        lstm_out, _ = self.lstm(features)

        # Global Average und Global Max Pooling aggregieren die Features über die Zeitachse
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        return self.fc(pooled)

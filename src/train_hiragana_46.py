#!/usr/bin/env python3
import os
import time
import datetime
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from kanji_nn.KanjiVGDataset import *
from kanji_nn.KanjiVGModel import *
from kanji_nn.filters import *
from kanji_nn.plot import save


if __name__ == "__main__":

    ZIP_PATH = "/Users/dehmer/Public/Data/kanjivg-20250816-all.zip"
    MODEL_OUTPUT = "hiragana_46.pt"
    MULTIPLIER = 64
    MAX_POINTS = 32 # hiragana
    # MAX_POINTS = 20 # katakana
    # MAX_POINTS = 80 # kanji

    BATCH_SIZE = 1024
    LOADER_WORKERS = 0
    EPOCHS = 15

    # Synchronize and start the precise epoch timer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pin_memory = False if torch.backends.mps.is_available() else True
    if device.type == "mps":
        torch.mps.synchronize()

    start_time = time.perf_counter()

    print(f"Run: {datetime.datetime.now()}")
    print("Preparing dataset (HIRAGANA_46)...")
    print()

    filter = lambda literal, _: literal == 'ア'
    dataset = KanjiVGDataset(ZIP_PATH, hiragana_46, MAX_POINTS, MULTIPLIER)
    training_size = int(0.90 * len(dataset))
    training_set, validation_set = torch.utils.data.random_split(
        dataset,
        [training_size, len(dataset) - training_size]
    )

    training_start_time = time.perf_counter()

    training_loader = DataLoader(
        training_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=LOADER_WORKERS,
        pin_memory=pin_memory,
        # drop_last=True, # caution with small datasets
        # persistent_workers=True
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=LOADER_WORKERS,
        pin_memory=pin_memory
    )

    num_classes = len(dataset.labels)


    print(f"Device:         {device}")
    print(f"DataLoader:     {LOADER_WORKERS} workers")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"Pin Memory:     {pin_memory}")
    print(f"Labels:         {num_classes}")
    print(f"Training on:    {len(training_set)} samples, multiplier: {MULTIPLIER}")
    print(f"Validation on:  {len(validation_set)} samples, multiplier: {MULTIPLIER}")
    print()

    model = KanjiVGModel(input_size=3, hidden_size=256, num_classes=num_classes).to(device)

    # Graphen-Kompilierung für maximale Hardware-Auslastung (PyTorch 2.x Feature)
    # if hasattr(torch, 'compile'):
    #     print("Kompiliere Modell mit torch.compile...")
    #     model = torch.compile(model, mode="max-autotune")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    print("Training model...")
    for epoch in range(EPOCHS):

        epoch_start_time = time.perf_counter()

        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for batch_x, batch_y in training_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Autocast mit bfloat16 für maximale Tensor-Kerne-Geschwindigkeit
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_loss = running_loss / total if total != 0 else None
        epoch_acc = (correct / total) * 100 if total != 0 else None

        # Validierungsschritt am Ende jeder Epoche
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for val_X, val_y in validation_loader:
                val_X = val_X.to(device, non_blocking=True)
                val_y = val_y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
                    val_outputs = model(val_X)
                _, val_pred = torch.max(val_outputs, 1)
                val_total += val_y.size(0)
                val_correct += (val_pred == val_y).sum().item()

        val_acc = (val_correct / val_total) * 100
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoche {epoch+1:02d}/{EPOCHS} | "
            f"Elapsed: {epoch_duration:5.2f}s | "
            f"TL: {epoch_loss:1.4f} | "
            f"TA: {epoch_acc:6.2f}% | "
            f"VA: {val_acc:6.2f}% | "
            f"LR: {current_lr}")

    print(f"Saving model to '{MODEL_OUTPUT}'...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'labels': dataset.label_indices,
        'max_seq_len': MAX_POINTS
    }, MODEL_OUTPUT)

    now = time.perf_counter()
    print("Training successfully finished.")
    print("")
    print(f"Time for dataset preparation:   {(training_start_time - start_time):5.2f}s")
    print(f"Time for training ({EPOCHS} epochs):  {(now - training_start_time):5.2f}s")
    print(f"Time total:                     {(now - start_time):5.2f}s")

    # for i in range(0, len(dataset)):
    #     strokes, label = dataset[i]
    #     idx = i // MULTIPLIER
    #     save(f"images/{dataset.labels[idx]}-{i:04}.png", strokes)

#!/usr/bin/env python3

import os
import ast
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from kanji_nn.KanjiVGModel import *
import kanji_nn.tsv as tsv

def hex_to_char(hex_str):
    if hex_str == "unknown":
        return "?"
    try:
        return chr(int(hex_str[2:], 16))
    except ValueError:
        return "?"

# ---------------------------------------------------------
# MAIN EVALUATION PIPELINE
# ---------------------------------------------------------
def run_analysis(
        tsv_path,
        model_path,
        output_log_path="hiragana46_evaluation.log",
        num_samples_per_stroke=10
    ):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TSV file '{tsv_path}' not found.")

    print(f"Loading model '{model_path}'...")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    print(checkpoint.keys())
    labels = checkpoint['labels']
    max_len = checkpoint.get('max_seq_len', 64)
    inverse_labels = {v: k for k, v in labels.items()}

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = KanjiVGModel(input_size=3, hidden_size=256, num_classes=len(labels)).to(device)

    # Bereinige eventuelle Präfixe aus torch.compile ("_orig_mod.")
    state_dict = checkpoint['model_state_dict']
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model = model.to(device)
    model.eval()


    print(f"Modell ready on device: {device}\n")
    test_set = tsv.read(tsv_path)

    total_tsv_lines = len(test_set['labels'])
    total_tested = 0
    correct = 0
    skipped_not_in_model = 0

    read_characters = []
    tested_characters = []
    mismatches = []

    for i, label in enumerate(test_set['labels']):
        strokes = np.array(test_set['strokes'][i])
        literal = test_set['literals'][i]

        if label not in labels:
            skipped_not_in_model += 1
            print(f"[IGNORIERT] Gelesen: {literal} ({label}) -> Nicht im Modell")
            continue

        expected_label = labels[label]

        input_tensor = torch.tensor(strokes, dtype=torch.float32).unsqueeze(0).to(device)

        # Inferenz & Top-3 Berechnung via Softmax Wahrscheinlichkeiten
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            topk_probs, topk_indices = torch.topk(probabilities, 3, dim=1)

            topk_probs = topk_probs.squeeze(0).cpu().numpy()
            topk_indices = topk_indices.squeeze(0).cpu().numpy()

        total_tested += 1
        tested_characters.append(literal)

        best_predicted_idx = topk_indices[0]
        # print('best_predicted_idx', best_predicted_idx)
        # print('expected_label', expected_label)
        status_tag = "[RICHTIG]  " if best_predicted_idx == expected_label else "[FALSCH]   "
        if best_predicted_idx == expected_label:
            correct += 1
        else:
            pred_hex = inverse_labels.get(best_predicted_idx, "unknown")
            mismatches.append((literal, label, hex_to_char(pred_hex), pred_hex))

        # Extraktion der Top-3 Vorhersagen mit Prozentwerten
        top_strings = []
        for rank in range(3):
            idx = topk_indices[rank]
            prob = topk_probs[rank] * 100.0
            h_id = inverse_labels.get(idx, "unknown")
            char = hex_to_char(h_id)
            top_strings.append(f"#{rank+1}: {char} ({prob:4.1f}%)")

        log_line = f"{status_tag} Gelesen: {literal} ({label}) | Top-3 -> " + " | ".join(top_strings) + "\n"
        print(log_line.strip())

    # Zusammenfassung generieren
    summary = []

    print()
    print("ANALYSE- UND EVALUATIONBERICHT")
    print(f"Gelesene Zeilen aus TSV-Datei:           {total_tsv_lines}")
    print(f"Ignoriert (Nicht im gelernten Umfang):    {skipped_not_in_model}")
    print(f"Tatsächlich getestete Zeichen:           {total_tested}")

    if total_tested > 0:
        accuracy = (correct / total_tested) * 100
        print(f"Korrekt klassifiziert:                   {correct}")
        print(f"Fehlklassifiziert:                        {len(mismatches)}")
        print(f"FINALE GENAUIGKEIT (Accuracy):       {accuracy:4.2f}%")
    else:
        print("Fehler: Keine übereinstimmenden Klassen gefunden. Prüfe die Filterregeln.")


if __name__ == "__main__":
    TSV_FILE = "/Users/dehmer/Public/Data/hiragana_48_fast.tsv"
    MODEL_FILE = "models/hiragana_46.pt"
    OUTPUT_LOG = "hiragana_46_infer.log"

    run_analysis(TSV_FILE, MODEL_FILE, OUTPUT_LOG)
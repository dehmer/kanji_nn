def prepare_label_encoding(wkb_reader):
    labels = [entry.label for entry in wkb_reader.meta] # label offsets from CSV
    uniq_labels = sorted(list(set(labels)))
    encoded = {label: i for i, label in enumerate(uniq_labels)}
    encode = lambda idx: encoded[labels[idx]]
    label = lambda idx: labels[idx]
    return encode, label

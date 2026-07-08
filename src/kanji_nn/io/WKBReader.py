from dataclasses import dataclass
from pathlib import Path
import csv
import shapely.wkb
import numpy as np

filter_any = lambda label: True

@dataclass
class IndexEntry:
    label:  str # e.g. 'U+4E00'
    offset: int # [byte]
    length: int # [byte]

class WKBReader:
    def __init__(self, dir_name, dataset_name, filter=filter_any):
        self.index_path = Path(f"{dir_name}/{dataset_name}.csv")
        self.wkb_path   = Path(f"{dir_name}/{dataset_name}.wkb")
        self.wkb        = self.wkb = self.wkb_path.open("rb")

        # Load index to memory and close file immediately.
        with self.index_path.open(newline="") as f:
            self.meta = [
                # Note: We are dropping reduncant index column.
                IndexEntry(row["label"], int(row["offset"]), int(row["length"]))
                for row in csv.DictReader(f)
                if filter(row["label"])
            ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.wkb:
            self.wkb.close()
            self.wkb = None

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        entry = self.meta[idx]
        self.wkb.seek(entry.offset)
        buffer = self.wkb.read(entry.length)

        # Parse WKB (Big-Endian) fixed MultiLineString.
        # Shapely automatically detects the Big-Endian header marker from the buffer.
        mls = shapely.wkb.loads(buffer)
        strokes = [np.array(ls.coords) for ls in mls.geoms]

        return entry.label, strokes

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

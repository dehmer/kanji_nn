import numpy as np
import csv
import shapely.wkb
from shapely.geometry import LineString, MultiLineString

def write_wkb(kanji_data, output_dir, base_name):
    """
    Saves a dataset of kanji characters to a CSV label file (with binary offsets)
    and a Big-Endian WKB binary file.

    Parameters:
    - kanji_data: List of dicts, e.g., [{'label': 'U+4E0A', 'strokes': [np.array, np.array, ...]}]
    - csv_path: Target path for the character labels and offset maps
    - bin_path: Target path for the binary MultiLineString WKB data
    """

    csv_path=f"{output_dir}/{base_name}.csv"
    bin_path=f"{output_dir}/{base_name}.wkb"

    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file, open(bin_path, mode='wb') as bin_file:
        csv_writer = csv.writer(csv_file)

        # Updated CSV Header with offset data for random access
        csv_writer.writerow(["index", "label", "byte_offset", "byte_length"])

        for index, item in enumerate(kanji_data):
            label = item['label']
            strokes_list = item['strokes']

            # 1. Convert strokes into Shapely geometries
            lines = [LineString(stroke) for stroke in strokes_list]
            multi_line_string = MultiLineString(lines)

            # 2. Export to WKB using explicit Big-Endian byte order (0 = XDR/Big-Endian)
            wkb_bytes = shapely.wkb.dumps(multi_line_string, byte_order=0)

            # 3. Track the exact current position in the binary file before writing
            current_offset = bin_file.tell()
            wkb_length = len(wkb_bytes)

            # 4. Write data to both targets
            csv_writer.writerow([index, label, current_offset, wkb_length])
            bin_file.write(wkb_bytes)

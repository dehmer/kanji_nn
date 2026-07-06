import numpy as np
import csv
import shapely.wkb
from shapely.geometry import LineString, MultiLineString

def load_wkb(base_name):
    """
    Sequentially reads the CSV labels and WKB binary file together in one go.
    Yields one character at a time to minimize memory consumption.

    Yields:
    - dict: {'index': int, 'label': str, 'strokes': [np.array, np.array, ...]}
    """

    csv_path = f"data/wkb/{base_name}.csv"
    bin_path = f"data/wkb/{base_name}.wkb"

    with open(csv_path, mode='r', encoding='utf-8') as csv_file, open(bin_path, mode='rb') as bin_file:
        csv_reader = csv.reader(csv_file)

        # # Skip the header row
        # header = next(csv_reader, None)

        for row in csv_reader:
            if not row:
                continue

            # 1. Unpack CSV data
            idx = int(row[0])
            label = row[1]
            byte_length = int(row[3]) # Use length to isolate the geometry's chunk

            # 2. Read exactly the number of bytes belonging to this geometry
            wkb_bytes = bin_file.read(byte_length)

            # 3. Parse the Big-Endian WKB back into a Shapely MultiLineString
            # Shapely automatically detects the Big-Endian header marker from the bytes
            multi_line_string = shapely.wkb.loads(wkb_bytes)

            # 4. Extract the individual lines back into an array of NumPy coordinates
            # multi_line_string.geoms gives access to individual LineStrings
            strokes = [np.array(line.coords) for line in multi_line_string.geoms]

            yield {
                'index': idx,
                'label': label,
                'strokes': strokes
            }

import csv
import shapely.wkb
from shapely.geometry import LineString, MultiLineString

def write_wkb(csv_writer, bin_file, data):
    index = data["index"]
    label = data["label"]
    strokes = data["strokes"]

    # 1. Convert strokes into Shapely geometries
    lines = [LineString(stroke) for stroke in strokes]
    multi_line_string = MultiLineString(lines)

    # 2. Export to WKB using explicit Big-Endian byte order (0 = XDR/Big-Endian)
    wkb_bytes = shapely.wkb.dumps(multi_line_string, byte_order=0)

    # 3. Track the exact current position in the binary file before writing
    current_offset = bin_file.tell()
    wkb_length = len(wkb_bytes)

    # 4. Write data to both targets
    csv_writer.writerow([index, label, current_offset, wkb_length])
    bin_file.write(wkb_bytes)

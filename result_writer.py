import csv
from pathlib import Path

class ResultWriter:

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path

    def write_row(self, row: dict):
        """
        Writes a single row (dictionary) to the CSV file.
        - If the CSV file does not exist, writes the header first.
        - Appends rows to the file to preserve previous results.
        """
        write_header = not self.csv_path.exists()

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            if write_header:
                writer.writeheader()

            writer.writerow(row)


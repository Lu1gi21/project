import csv
from collections import Counter

region_counter = Counter()
total = 0

with open("vgsales.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            sales = {
                'NA': float(row['NA_Sales']),
                'EU': float(row['EU_Sales']),
                'JP': float(row['JP_Sales']),
                'Other': float(row['Other_Sales'])
            }
            dominant = max(sales, key=sales.get)
            region_counter[dominant] += 1
            total += 1
        except:
            continue

print("Dominant Region Distribution (%):")
for region, count in region_counter.items():
    print(f"{region}: {round((count / total) * 100, 2)}%")

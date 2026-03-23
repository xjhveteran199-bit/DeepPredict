# -*- coding: utf-8 -*-
import urllib.request, io, pandas as pd

urls = [
    ('https://raw.githubusercontent.com/selva86/datasets/master/energy_data.csv', 'energy_data.csv'),
    ('https://raw.githubusercontent.com/selva86/datasets/master/temperature.csv', 'temperature.csv'),
    ('https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv', 'shampoo.csv'),
    ('https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-temperature.csv', 'monthly-temperature.csv'),
    ('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv', 'daily-temp.csv'),
    ('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv', 'airline.csv'),
    ('https://raw.githubusercontent.com/jbrownlee/Datasets/master/士斯纳数据集/master/daily-total-female-births.csv', 'births.csv'),
]

for url, fname in urls:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        print(f'OK: {fname}, size={len(data)}')
        df = pd.read_csv(io.BytesIO(data))
        print(f'  Shape: {df.shape}, Cols: {df.columns.tolist()}')
    except Exception as e:
        print(f'Failed: {fname} - {e}')

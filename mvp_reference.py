import pandas as pd
import requests
from bs4 import BeautifulSoup

url = "https://www.basketball-reference.com/awards/mvp.html"

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

table = soup.find('table')
df = pd.read_html(str(table))[0]

df.to_csv('all_mvp_stats.csv', index=False)
print("MVP Data saved successfully!") 
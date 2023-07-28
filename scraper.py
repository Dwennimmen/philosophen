import requests
from bs4 import BeautifulSoup


def save_file(text, i):
    with open(r'dummy%s.txt' % i, 'w', encoding="utf-8" ) as f:
        f.write(str(text))
    f.close()
            

def get_content(link, index):
    page = requests.get(link)
    soup = BeautifulSoup(page.content, 'html.parser')
    #Find Elements by HTML Class Name
    text = soup.find_all("p")
    text = [para.get_text(" ", strip=True).strip() for para in text]
    i = index +1
    save_file(text, i)
    

import pandas as pd

#get file for websites that split the works into separate chapters
df = pd.read_excel('links.xlsx', header=None) # contains the links of the website


df = df.dropna(subset=[1])

result = df[1].to_list()


for e in result:
    index=result.index(e)
    get_content(e, index)

# url = "" # if only one website
# get_content(url, index)





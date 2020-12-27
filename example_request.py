import requests
from bs4 import BeautifulSoup
# from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium import webdriver

# url = r"https://www.wunderground.com/history/daily/pl/wroc%C5%82aw/EPWR/date/2020-12-16"

# chrome_options = Options()  
# chrome_options.add_argument("--headless") # Opens the browser up in background

# with webdriver.Chrome('./chromedriver_linux64/chromedriver', options=chrome_options) as browser:
#     browser.get(url)
#     page = browser.page_source

# with open('page', 'w+') as file:
#     print(page, file=file)


with open('page') as file:
    page = file.read()

# print(type(page))
soup = BeautifulSoup(page, 'html.parser')

table = soup.findAll('table')
# print(len(table))
our_table = table[1]

thead = our_table.findAll('thead')[0]
# print(thead)

names = []
units = []
data = []

ths = thead.findAll('th')

for th in ths:
    button = th.find('button')
    names += [button.contents[0]]

tbody = our_table.find('tbody')
trs = tbody.findAll('tr')

for tr in trs:
    d = []
    u = []
    for td in tr.findAll('td'):
        spans = td.findAll('span')
        if len(spans) < 2:
            d += spans[0].contents
            u += ['']
        else:
            tmp = td.find('span', {'class':"wu-value wu-value-to"})
            m = td.findAll('span', {'class':"ng-star-inserted"})[1]
            # print(m)
            if units == []:
                u += [m.contents[0]]
            d += [tmp.contents[0]]
    if units == []:
        units = u
    data += [d]

print(data)
print(units)
print(len(data))
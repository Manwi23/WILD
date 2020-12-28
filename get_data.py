# This is a script for collecting the data

# For a given timerange and location it downloads the data from the webpage, 
# gets the desired weather data and stores it in a directory

# The stored data is both the downloaded page and the scraped data (all_data.json)

from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from datetime import timedelta, date
import time
import random
import json
import sys
import os
from pathlib import Path

start = date(2018, 1, 1) # put your favourite date here
end = date(2019, 2, 1) # and another here

days = int((end - start).days)

all_data = []

# Wrocław is fixed here
url = r"https://www.wunderground.com/history/daily/pl/wroc%C5%82aw/EPWR/date/"

for i in range(days):
    cur = start + timedelta(days=i)
    print(cur)

    cur_url = url + str(cur)

    filename = str(cur.year)+"/"+str(cur)

    if not os.path.exists(str(cur.year)):
        os.makedirs(str(cur.year))

    while 1:
        try:

            # if the webpage was already downloaded, use it instead of downloading again

            my_file = Path(filename)
            if my_file.is_file():
                with open(my_file) as mf:
                    page = mf.read()
            else:
                chrome_options = Options()  
                chrome_options.add_argument("--headless") # Opens the browser up in background

                with webdriver.Chrome('./chromedriver_linux64/chromedriver', options=chrome_options) as browser:
                    browser.get(cur_url)
                    page = browser.page_source

            soup = BeautifulSoup(page, 'html.parser')

            table = soup.findAll('table')
            our_table = table[1]

            thead = our_table.findAll('thead')[0]

            names = ['date']
            units = []
            data = []

            ths = thead.findAll('th')

            for th in ths:
                button = th.find('button')
                names += [button.contents[0]]

            tbody = our_table.find('tbody')
            trs = tbody.findAll('tr')

            for tr in trs:
                d = [str(cur)]
                u = ['']
                for td in tr.findAll('td'):
                    spans = td.findAll('span')
                    if len(spans) < 2:
                        d += spans[0].contents
                        u += ['']
                    else:
                        tmp = td.find('span', {'class':"wu-value wu-value-to"})
                        m = td.findAll('span', {'class':"ng-star-inserted"})[1]
                        if units == []:
                            u += [m.contents[0]]
                        d += [tmp.contents[0]]
                if units == []:
                    units = u
                data += [d]

            # print(data)
            # print(units)
            # print(len(data))

            all_data += [(names, units, data)]
            break

        except IndexError as e:
            # it happens when the page didn't load for some reason

            t = random.random() * 10
            print("failed, sleeping for", t)
            sys.stdout.flush()
            time.sleep(t)
            pass

    with open(filename, 'w+') as file:
        print(page, file=file)


    # save after each webpage? probably makes sense if we download a lot and something might crash

    j = json.dumps(all_data)
    with open("2019/all_data.json", 'w+') as file:
        print(j, file=file)
# This is a script for collecting the data

# For a given timerange and location it downloads the data from the webpage, 
# gets the desired weather data and stores it in a directory

# The stored data is both the downloaded page and the scraped data (all_data + timestamp.json)

from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from datetime import timedelta, date
from pathlib import Path
import time
from time import sleep
import random
import json
import sys
import os
import csv
from get_places import get_city_address

def scrape_date_range(name, start=date(2018, 1, 1), end=date(2018, 1, 30), chromedriver='./chromedriver'):

    url = get_city_address(name, headless=True, wait_for_page=5, chromedriver=chromedriver)
    print(url)

    days = int((end - start).days)

    all_data = []
    just_data = []
    just_names = []
    just_units = []

    sleeping_time = 1

    # # Wrocław is fixed here
    # url = r"https://www.wunderground.com/history/daily/pl/wroc%C5%82aw/EPWR/date/"

    chrome_options = Options()  
    chrome_options.add_argument("--headless") # Opens the browser up in background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument('--disable-dev-shm-usage')


    for i in range(days):
        cur = start + timedelta(days=i)
        print(cur)

        cur_url = url + str(cur)
        cur_dir = "scraped/"+str(cur.year)
        filename = cur_dir+"/"+str(cur)

        if name != 'wroclaw':
            filename += '_'+name

        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

        while 1:
            try:

                # if the webpage was already downloaded, use it instead of downloading again

                my_file = Path(filename)
                if my_file.is_file():
                    with open(my_file) as mf:
                        page = mf.read()
                else:
                    with webdriver.Chrome(chromedriver, options=chrome_options) as browser:
                        browser.implicitly_wait(15)
                        browser.get(cur_url)
                        browser.find_element_by_tag_name("table")
                        # sleep(5)
                        page = browser.page_source

                soup = BeautifulSoup(page, 'html.parser')

                table = soup.findAll('table')
                our_table = table[1]

                thead = our_table.findAll('thead')[0]

                names = ['Date']
                # names = ['Place', 'Date'] (if one wanted to store place in data)
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
                    # d = [name.lower(), str(cur)] (if one wanted to store place in data)
                    u = ['']
                    # u = ['', ''] (if one wanted to store place in data)
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

                if just_names == []:
                    just_names = names
                if just_units == []:
                    just_units = units
                just_data += data

                sleeping_time = 1

                break

            except Exception as e:
                # it happens when the page didn't load for some reason

                t = sleeping_time + random.random()
                print(f"failed because of {e}, sleeping for", t)
                sys.stdout.flush()
                time.sleep(t)
                sleeping_time = 2*t
                pass

        with open(filename, 'w+') as file:
            print(page, file=file)

    with open("scraped/" +str(name) + str(start) + str(end) + ".csv", 'w+') as file:
        nu = [i + "|" + j for (i,j) in zip(just_names, just_units)]

        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(nu)

        for d in just_data:
            writer.writerow(d)


if __name__ == "__main__":
    
    scrape_date_range("wroclaw")

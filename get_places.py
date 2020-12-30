# This is a script for collecting the data

# For a given timerange and location it downloads the data from the webpage, 
# gets the desired weather data and stores it in a directory

# The stored data is both the downloaded page and the scraped data (all_data + timestamp.json)

from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from datetime import timedelta, date
from pathlib import Path
import time
import random
import json
import sys
import os
import csv

start = date(2018, 1, 1) # put your favourite date here
end = date(2018, 2, 1) # and another here


# places = [('pl', 'wrocław'), ('gb', 'london')]


days = int((end - start).days)

all_data = []
just_data = []
just_names = []
just_units = []

sleeping_time = 1

# Wrocław is fixed here
base = "https://www.wunderground.com"
url = r"https://www.wunderground.com/history/daily/pl/wroc%C5%82aw/EPWR/date/"
cur_url = url + "2018-01-01"

chrome_options = Options()  
# chrome_options.add_argument("--headless") # Opens the browser up in background

with webdriver.Chrome('./chromedriver_linux64/chromedriver', options=chrome_options) as browser:
    browser.get(cur_url)
    
    search = browser.find_element_by_id("wuSearch")
    # print(search)
    search.send_keys("london")
    search.send_keys(Keys.ENTER)

    page = browser.page_source

    soup = BeautifulSoup(page, 'html.parser')

    hrefs = soup.findAll("a", href=True)

    t = [href for href in hrefs if 'history' in str(href)]
    a = t[0]['href']

    browser.get(base + a)



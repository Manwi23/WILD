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
from time import sleep
import random
import json
import sys
import os
import csv

def get_city_address(city_name, headless=True, wait_for_page=0, chromedriver='./chromedriver'):

    base = "https://www.wunderground.com"

    chrome_options = Options()  
    if headless:
        chrome_options.add_argument("--headless") # Opens the browser up in background
        chrome_options.add_argument("--no-sandbox")

    with webdriver.Chrome(chromedriver, options=chrome_options) as browser:
        browser.get(base)
        search = browser.find_element_by_id("wuSearch")
        # print(search)
        search.send_keys(city_name)
        sleep(wait_for_page)
        search.send_keys(Keys.ENTER)
        sleep(wait_for_page)

        page = browser.page_source

        soup = BeautifulSoup(page, 'html.parser')

        hrefs = soup.findAll("a", href=True)

        t = [href for href in hrefs if 'history' in str(href)]
        a = t[0]['href']
        h, s, _ = a.partition('/date/')
        return base+h+s

if __name__ == "__main__":
    print(get_city_address("wroclaw", headless=True, wait_for_page=4))
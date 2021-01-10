from datetime import date
from get_data import scrape_date_range

print("scraping even older older")
scrape_date_range("wroclaw", date(2014, 1, 1), date(2014, 3, 31))
scrape_date_range("wroclaw", date(2015, 1, 1), date(2015, 3, 31))
scrape_date_range("wroclaw", date(2016, 1, 1), date(2016, 3, 31))
print("scraping even older")
scrape_date_range("wroclaw", date(2017, 1, 1), date(2017, 3, 31))
print("scraping older")
scrape_date_range("wroclaw", date(2018, 1, 1), date(2018, 3, 31))
print("scraping old")
scrape_date_range("wroclaw", date(2019, 1, 1), date(2019, 3, 31))
print("scraping new")
scrape_date_range("wroclaw", date(2020, 1, 1), date(2020, 3, 31))


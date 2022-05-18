from selenium import webdriver
from bs4 import BeautifulSoup
import scrapy

if __name__ == '__main__':
    # url = "https://realpython.github.io/fake-jobs/"
    # page = requests.get(url)
    # # soup = BeautifulSoup(page.content, 'html.parser')
    # elements = soup.select("div.column.is-half")
    # print('gr')

    url1 = "https://www.xe.gr/property/results?transaction_name=rent&item_type=re_residence&geo_place_id=ChIJsfGp3-CboRQRAHy54iy9AAQ&sorting=price_asc"
    page1 = requests.get(url1)
    soup1 = BeautifulSoup(page1.content, 'html.parser')
    elements1 = soup1.select("div[data-testid*='property-ad']")
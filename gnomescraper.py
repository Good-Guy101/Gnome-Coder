import time
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
import os

def tryURL(url):
	try:
		urlopen(url)
		return True
	except:
		return False

URL = 'https://www.homedepot.com/s/garden%20gnome%20statue?NCNI-5&Nao='
count = 0
DIR = "./gnomes/"

if(not os.path.exists(DIR)):
	os.makedirs(DIR)

driver = webdriver.Firefox()

for i in range(100):
	URLpage = URL + str(i*24)
	print(URLpage)
	if not tryURL(URLpage):
		print("End of Pages")
		break
	driver.get(URLpage)
	time.sleep(3)
	driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
	time.sleep(1)
	driver.execute_script("window.scrollTo(document.body.scrollHeight,0)")
	time.sleep(1)
	driver.execute_script("window.scrollTo(0,document.body.scrollHeight / 2)")
	time.sleep(1)
	page = driver.execute_script('return document.body.innerHTML')
	soup = BeautifulSoup(''.join(page), 'html.parser')
	images = soup.find_all('img', attrs={'class':'stretchy'})

	for item in images:
		if not tryURL(item['src']):
			print("Image Did Not Load")
			continue
		if(item['alt'] == "Available Shipping" or item['alt'] == "Available for pickup"):
			continue
		count += 1
		urllib.request.urlretrieve(item['src'], DIR + str(count) + ".jpg")
		print(item['src'])
		print(item['alt'])


print(count)
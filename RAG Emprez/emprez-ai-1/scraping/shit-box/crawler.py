from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from contextlib import contextmanager
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import json, codecs

'''
class Article:
    def __init__(self, title, url, content):
        self.title = title
        self.url = url
        self.content = content
'''


@contextmanager
def wait_for_page_load(driver, timeout=10):
    old_page = driver.find_element(By.TAG_NAME, 'html')
    yield
    WebDriverWait(driver, timeout).until(EC.staleness_of(old_page))

def wait_for_page(driver,href):
    WebDriverWait(driver, 10).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    driver.get(href)
    print(f'Page {href} loaded.')


def crawler(start_url):
    driver = webdriver.Chrome()
    driver.get(start_url)
    
    while True:
        print('Looking for links')
        links = driver.find_elements(By.TAG_NAME, 'a')
        print('Found ',links)
        for link in links:
            href = link.get_attribute('href')
            if href and ('.xml' in href or '.pdf' in href):
                print(f'Download {href}')
                # Add code to download the file
            else:
                if "/portal/fr/kb/" in href:
                    print(f'Click {href}')
#                    with wait_for_page_load(driver):
                    #wait_for_page(driver,href)
                    # with wait_for_page(driver):
                    #     driver.get(href)
                    WebDriverWait(driver, 10).until(
                        lambda d: d.execute_script("return document.readyState") == "complete"
                    )
                    driver.get(href)
                    print(f'Page {href} loaded.')
        
        # Check if there are more links to process
        if not links:
            break

    driver.quit()


@contextmanager
def dump_article(titles,soup,url):
    for title in titles:
            articleTitle = title.get_text()
    
        #print(f'Title: {articleTitle}')

    mydivs = soup.find_all("div", {"class": "articleContent"})
    for div in mydivs:
            #print('Found articleContent',div.get_text()) 
            #with open(articleTitle + ".txt", "w") as text_file:
            #    text_file.write(div.get_text())
        with open(articleTitle + ".json", 'wb') as f:
            article = {
                "title": articleTitle,
                "url": url,
                "content": div.get_text()
            }
                # article = Article(articleTitle,url,div.get_text()) 
            json.dump(article, codecs.getwriter('utf-8')(f), ensure_ascii=False)

        # div.decode_contents() 


def crawl_urls(site,url_list, crawled_urls, driver, url):
    """ get a set of urls and crawl each url recursively"""

    print('crawling ',url);
    # Once the url is parsed, add it to crawled url list
    crawled_urls.append(url)

    driver.get(url)
    html = driver.page_source.encode("utf-8")

    soup = BeautifulSoup(html,'html.parser')

    titles = soup.find_all("h1", {"data-id": "articleTitle"})
    articleTitle = 'NoTitle.' + url
    if titles:
        dump_article(titles,soup,url)
        '''
        for title in titles:
            articleTitle = title.get_text()
    
        #print(f'Title: {articleTitle}')

        mydivs = soup.find_all("div", {"class": "articleContent"})
        for div in mydivs:
            #print('Found articleContent',div.get_text()) 
            #with open(articleTitle + ".txt", "w") as text_file:
            #    text_file.write(div.get_text())
            with open(articleTitle + ".json", 'wb') as f:
                article = {
                    "title": articleTitle,
                    "url": url,
                    "content": div.get_text()
                }
                # article = Article(articleTitle,url,div.get_text()) 
                json.dump(article, codecs.getwriter('utf-8')(f), ensure_ascii=False)

            # div.decode_contents() 
        '''
    else:
        urls = soup.findAll("a")

        # Even if the url is not part of the same domain, it is still collected
        # But those urls not in the same domain are not parsed
        for a in urls:
            #print(f'Url {a}')
            if (a.get("href") and ("/portal/fr/kb" in a.get("href"))):
                foundLink = site + a.get("href")
               # print('foundLink ',foundLink)
                if (foundLink not in url_list):
                    print('Adding ',foundLink)
                    url_list.append(foundLink)

            # if (a.get("href")) and (a.get("href") not in url_list) and ("/portal/fr/kb/" in a.get("href")):
            #     print('Adding ',(site+a.get("href")))
            #     url_list.append(site + a.get("href"))

    # Recursively parse each url within same domain
    for page in set(url_list):  # set to remove duplicates

        # Check if the url belong to the same domain
        # And if this url is already parsed ignore it
#        if (urlparse(page).netloc == domain) and (page not in crawled_urls):
        if (page not in crawled_urls):

            # print this_url
            crawl_urls(site,url_list, crawled_urls, driver, page)

    # Once all urls are crawled return the list to calling function
    else:
        return crawled_urls, url_list


# Example usage
site = 'https://emprez.zohodesk.com'
start_url = 'https://emprez.zohodesk.com/portal/fr/kb'
start_url='https://emprez.zohodesk.com/portal/fr/kb/je-g%C3%A8re-la-paie'
start_url='https://emprez.zohodesk.com/portal/fr/kb/articles/gestion-des-d%C3%A9penses'
#crawler(start_url)

driver = webdriver.Chrome()

url_list = list()
crawled_urls = list()

url_list.append(start_url)

    # Initiate the crawling by passind the beginning url
crawled_urls, url_list = crawl_urls(site,url_list, crawled_urls, driver, start_url)



    # Finally quit the browser
driver.quit()

print ("FULL URLs LIST",url_list)
#print len(set(url_list))

print ("CRAWLED URLs LIST",crawled_urls)
#print len(set(crawled_urls))


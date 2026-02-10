from selenium import webdriver
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.by import By
from contextlib import contextmanager
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import json, codecs, os, time
from sanitize_filename import sanitize


@contextmanager
def dump_article(titles,soup,url,dir_json):
    for title in titles:
            articleTitle = title.get_text()
    
        #print(f'Title: {articleTitle}')

    mydivs = soup.find_all("div", {"class": "articleContent"})
    for div in mydivs:
            #print('Found articleContent',div.get_text()) 
            #with open(articleTitle + ".txt", "w") as text_file:
            #    text_file.write(div.get_text())
        try:
            os.makedirs(dir_json, exist_ok=True)
        except OSError as error:
            print(f"Can't create directory {dir_json} Error: {error}")

        with open(dir_json+sanitize(articleTitle + ".json"), 'wb') as f:
            article = {
                "title": articleTitle,
                "url": url,
                "content": div.get_text()
            }
                # article = Article(articleTitle,url,div.get_text()) 
            json.dump(article, codecs.getwriter('utf-8')(f), ensure_ascii=False)
            print (f'Article {articleTitle} saved.')

        # div.decode_contents() 


def crawl_urls(site,url_list, crawled_urls, driver, url,dir_json):
    """ get a set of urls and crawl each url recursively"""

    print('crawling ',url);
    # Once the url is parsed, add it to crawled url list
    crawled_urls.append(url)

    #print('Get Url')
    driver.get(url)
    print('Retrieving page')
    html = driver.page_source.encode("utf-8")
    # print(f'Page retrieved {len(html)} characters')
    # with open(sanitize(url) + ".html", "wb") as text_file:
    #     text_file.write(html)

    # html_content = driver.execute_script("return document.documentElement.outerHTML")
    # print(f'Page (html_content) retrieved {len(html_content)} characters')
    while True:
        time.sleep(5) # My network is slow when accessing Zoho KB.
        print('Verifying')
        if (driver.page_source.encode("utf-8")==html):
            break
        else:
            print('Waiting html is still generating')
        html = driver.page_source.encode("utf-8")

    print('Parsing now')

    soup = BeautifulSoup(html,'html.parser')

    titles = soup.find_all("h1", {"data-id": "articleTitle"})
    articleTitle = 'NoTitle.' + url
    if titles:
        dump_article(titles,soup,url,dir_json)
    else:
        urls = soup.findAll("a")

        # Even if the url is not part of the same domain, it is still collected
        # But those urls not in the same domain are not parsed
        for a in urls:
            #print(f'Url {a}')
            if (a.get("href") and ( ("/portal/fr/kb" in a.get("href")) or ("/portal/en/kb" in a.get("href")) )):
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
            crawl_urls(site,url_list, crawled_urls, driver, page,dir_json)

    # Once all urls are crawled return the list to calling function
    else:
        return crawled_urls, url_list

def start(driver, start_url,dir_json):
    url_list = list()
    crawled_urls = list()
    crawled_urls.append('https://emprez.zohodesk.com/portal/fr/kb') #To block
    crawled_urls.append('https://emprez.zohodesk.com/portal/en/kb') #To block
    url_list.append(start_url)
    # Initiate the crawling by passind the beginning url
    crawled_urls, url_list = crawl_urls(site,url_list, crawled_urls, driver, start_url,dir_json)
#print ("FULL URLs LIST",url_list)
    print(f'{len(set(url_list))} urls found')

#print ("CRAWLED URLs LIST",crawled_urls)
    print(f'{len(set(crawled_urls))} urls crawled')
    # print(len(set(crawled_urls)))



# Example usage
site = 'https://emprez.zohodesk.com'
start_url = 'https://emprez.zohodesk.com/portal/fr/kb'
start_url='https://emprez.zohodesk.com/portal/fr/kb/je-g%C3%A8re-la-paie'

fr_url_admin = 'https://emprez.zohodesk.com/portal/fr/kb/je-suis-admin'
fr_url_manager = 'https://emprez.zohodesk.com/portal/fr/kb/je-suis-responsable'
fr_url_payroll='https://emprez.zohodesk.com/portal/fr/kb/je-gère-la-paie'
fr_url_employee='https://emprez.zohodesk.com/portal/fr/kb/je-suis-un-employe'
fr_url_general='https://emprez.zohodesk.com/portal/fr/kb/emprez-1'
fr_url_emprez_bi='https://emprez.zohodesk.com/portal/fr/kb/performance-rh'


en_url_admin = 'https://emprez.zohodesk.com/portal/en/kb/i-am-an-administrator'
en_url_manager = 'https://emprez.zohodesk.com/portal/en/kb/je-suis-responsable'
en_url_payroll='https://emprez.zohodesk.com/portal/en/kb/je-gère-la-paie'
en_url_employee='https://emprez.zohodesk.com/portal/en/kb/je-suis-un-employe'
en_url_general='https://emprez.zohodesk.com/portal/en/kb/emprez-1'
en_url_emprez_bi='https://emprez.zohodesk.com/portal/en/kb/hr-performance'


#_start_url='https://emprez.zohodesk.com/portal/fr/kb/articles/gestion-des-d%C3%A9penses'
#crawler(start_url)


driver = webdriver.Chrome()
#driver.implicitly_wait(15); #15 sec
'''
dir_json = '.'
url_list = list()
crawled_urls = list()
url_list.append(start_url)
# Initiate the crawling by passind the beginning url
crawled_urls, url_list = crawl_urls(site,url_list, crawled_urls, driver, start_url,dir_json)
'''


#Test
#fr_url_admin = 'https://emprez.zohodesk.com/portal/fr/kb/je-suis-admin/rapport-de-gestion'
#fr_url_admin='https://emprez.zohodesk.com/portal/fr/kb/je-suis-responsable/premier-pas-vers-emprez'

'''
'''
start(driver,fr_url_admin,'fr/je_suis_administrateur/')
start(driver,fr_url_manager,'fr/je_suis_responsable/')
'''
start(driver,fr_url_payroll,'fr/je_fais_la_paie/')
start(driver,fr_url_employee,'fr/je_suis_un_employé/')
start(driver,fr_url_general,'fr/emprez_général/')
start(driver,fr_url_emprez_bi,'fr/emprez_business_intelligence/')

start(driver,en_url_admin,'en/i_am_administrator/')
start(driver,en_url_manager,'en/i_am_manager/')
start(driver,en_url_payroll,'en/payroll/')
start(driver,en_url_employee,'en/i_am_an_employee/')
start(driver,en_url_general,'en/emprez_general/')
start(driver,en_url_emprez_bi,'en/emprez_business_intelligence/')
#'''

    # Finally quit the browser
driver.quit()

print ('Done.')

#print ("FULL URLs LIST",url_list)
#print len(set(url_list))

#print ("CRAWLED URLs LIST",crawled_urls)
#print len(set(crawled_urls))


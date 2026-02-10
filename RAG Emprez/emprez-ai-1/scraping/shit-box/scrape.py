import requests
from bs4 import BeautifulSoup
from selenium import webdriver

def get_links_from_url(host,link,driver):
    try:
        # Send a GET request to the URL
        
        url = host + link
        print('Loading ',url);

        driver.get(url)
        page = driver.page_source

        # response = requests.get(url)
        # response.raise_for_status()  # Raise an error for bad status codes

        # print("READ ",response.text)
        #print("READ ",page)
        links=[]

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(page, 'html.parser')

        # Find all <a> tags (links) in the HTML
        for a_tag in soup.find_all('a', href=True):
            foundLink = a_tag['href']
            if "/portal/fr/kb/" in foundLink:
                if foundLink not in links:                
                    print('Found link ',foundLink)
                    links.append(foundLink)
                    get_links_from_url(host,foundLink,driver)

        return links

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")

# Example usage
host="https://emprez.zohodesk.com"
#links = []
driver = webdriver.Chrome()
#get_links_from_url(host,'/portal/fr/kb',driver,links)
allLinks = get_links_from_url(host,'/portal/fr/kb',driver)
'''
links2 = []
for link in links:
    if "portal/fr/kb/" in link: 
        print('Getting link: ',(host+link));
        get_links_from_url(host+link,driver,links2)

links = links + links2

if links:
    print(f"Found {len(links)} links:")
    for link in links:
        print(link)
else:
    print("No links found or an error occurred.")
'''
    
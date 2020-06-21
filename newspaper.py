import requests
import requests.exceptions
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import colorama

colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RED = colorama.Fore.RED
RESET = colorama.Fore.RESET

internal_urls = set()
external_urls = set()

total_urls_visited = 0

output = ''

filewrite = ''

def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_all_website_links(url):
    urls = set()
    domain_name = urlparse(url).netloc
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            continue
        href = urljoin(url, href)
        parsed_href = urlparse(href)
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
        if not is_valid(href):
            continue
        if href in internal_urls:
            continue
        if domain_name not in href:
            if href not in external_urls:
                print(f"{GRAY}[!] External link: {href}{RESET}")
                external_urls.add(href)
            continue
        print(f"{GREEN}[*] Internal link: {href}{RESET}")
        urls.add(href)
        internal_urls.add(href)
    return urls


def crawl(url, max_urls=50):
    global total_urls_visited
    total_urls_visited += 1
    links = get_all_website_links(url)
    for link in links:
        if total_urls_visited > max_urls:
            break
        crawl(link, max_urls=max_urls)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="News data Crawler/Extractor built in python by Sam Ehlers")
    parser.add_argument("url", help="The domain to extract data from.")
    parser.add_argument("-m", "--max-urls", help="Number of max URLs to crawl, default is 30.", default=30, type=int)

    args = parser.parse_args()
    url = args.url

    crawl(url, max_urls=int(30))

    for site in internal_urls:
        try:
            res = requests.get(site)
        except Exception as e:
            print(e)
        soup = BeautifulSoup(res.content, 'html.parser').find_all(text=True)

        blacklist = [
        	'[document]',
        	'noscript',
        	'header',
        	'html',
        	'meta',
        	'head',
        	'input',
        	'script',
        ]

        for t in soup:
        	if t.parent.name not in blacklist:
        		output += "{} ".format(t).strip()

        print(f"{RED}Writing page {site} contents to file: {(urlparse(url).netloc).strip()}_content.txt{RESET}")

        filewrite += output + '\n'

    with open(f"{(urlparse(url).netloc).strip()}_content.txt", "w") as f:
        print(filewrite, file=f)

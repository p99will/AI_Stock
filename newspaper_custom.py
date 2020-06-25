import requests, os, json, platform, colorama
import requests.exceptions
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET

plat = platform.system()
cwd = os.getcwd()
print(f'{GRAY}[*] Current working directory identified as {cwd}{RESET}')
path = ''
if plat == 'Windows':
    path = cwd + '\\articles'
else:
    path = cwd + '/articles'
if os.path.isdir(path):
    print(f'{GRAY}[*] Created folder \"articles\" at {path}{RESET}')
else:
    os.mkdir(path)
    print(f'{GRAY}[*] Created folder \"articles\" at {path}{RESET}')

internal_urls = set()
external_urls = set()

total_urls_visited = 0

output = ''

filewrite = ''

LINKCACHEFILE = 'chache'

def appendFile(filename,data):
    f=open(filename,'a')
    f.write(data + '\n')
    f.close()

def writeFile(filename, data):
    f=open(filename,'w')
    f.write(data)
    f.close()

def readFile(filename):
    f=open(filename,'r')
    data=f.read()
    f.close()
    return data

def makeCahce(data):
    for i in data:
        appendFile(LINKCACHEFILE,i)

def checkCache():
    if(os.path.isfile(LINKCACHEFILE)):
        print("Data cache being used.")
        data = readFile(LINKCACHEFILE).splitlines()
    else:
        print("No cache found.")
        data = ''
    return data

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

SUBFOLDER = "articles"
def writeJson(filename, data):
    data = json.dumps(data, indent=4)
    writeFile(SUBFOLDER + '/' + filename + '.json', data)

def crawl(url, max_urls=50):
    global total_urls_visited
    total_urls_visited += 1
    links = get_all_website_links(url)
    for link in links:
        if total_urls_visited > max_urls:
            break
        crawl(link, max_urls=max_urls)

allow_cache = True
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="News data Crawler/Extractor built in python by Sam Ehlers")
    parser.add_argument("url", help="The domain to extract data from.")
    parser.add_argument("-m", "--max-urls", help="Number of max URLs to crawl, default is 30.", default=30, type=int)

    args = parser.parse_args()
    url = args.url

    if(allow_cache):
        urls = checkCache()
        if(urls == ''):
            print("[*] Crawling")
            crawl(url, max_urls=int(30))
            makeCahce(internal_urls)
        else:
            internal_urls = set(urls)

    for site in internal_urls:
        try:
            res = requests.get(site)
        except Exception as e:
            print(e)

        raw_htmlBS = BeautifulSoup(res.content, 'html.parser') # Keep <div id="xxxx"> YES
        divBS = raw_htmlBS.find("div",attrs={'class':"content__article-body from-content-api js-article__body"})
        articleDate = raw_htmlBS.find("time", attrs={'class':'content__dateline-wpd js-wpd'})
        title = raw_htmlBS.find('title')
        if(title):
            title = title.string
            new_title = ''
            for char in title:
                if char.isalnum() or char == ' ':
                    new_title += char
            if(divBS == None or articleDate == None):
                print(f"{GRAY}[!] No news found at: {site}{RESET}")
            elif divBS != None:
                articleDate = articleDate.find_all(text=True)[0]
                print(articleDate)
                soup = divBS.find_all(text=True)

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

                print(f"{GREEN}[*] Writing page {site} contents to file: {new_title}.json")

                output = ''.join((c for c in str(output) if ord(c) < 128)) # Removes unicode
                toAppend = { 'text': output, 'date' : articleDate} # Adds date
                # print(toAppend)
                writeJson(new_title,toAppend)
                # filewrite.append(toAppend)
                # filewrite += output + '\n'
            else:
                continue
        else:
            print(f"{GRAY}[!] No title found, skipping {site}{RESET}")

    # with open(f"{(urlparse(url).netloc).strip()}_content.txt", "w",encoding="UTF-8") as f:
    #     print(filewrite, file=f)

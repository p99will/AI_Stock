from newspaper import Article
import newspaper

cnn_paper = newspaper.build('https://www.investopedia.com')

for article in cnn_paper.articles:
    print(article.url)

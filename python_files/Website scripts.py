from urllib.request import urlopen

url = "https://web.whatsapp.com"

page = urlopen(url)

html = page.read().decode("utf-8")

# print(html)

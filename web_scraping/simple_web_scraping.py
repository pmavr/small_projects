
from urllib.request import urlopen
import re


if __name__ == '__main__':
    url = "http://olympus.realpython.org/profiles/aphrodite"
    page = urlopen(url)

    html_bytes = page.read()
    html = html_bytes.decode("utf-8")

    start_index = html.find("<title>") + len("<title>")
    end_index = html.find("</title>")
    title = html[start_index:end_index]

    matching_cases = re.findall("a.c", "abc adbc dbbbc Abc", re.IGNORECASE)
    # re.search("a.c", "abc adbc dbbbc Abc", re.IGNORECASE).group() # group() returns first matching case

    string = "Everything is <replaced> if it's in <tags>."
    string1 = re.sub("<.*>", "ELEPHANTS", string)
    string2 = re.sub("<.*?>", "ELEPHANTS", string)
    print('gr')
#!/usr/bin/env python
import urllib
import json
import sys

API_KEY = "AIzaSyAPSxj_e8L5mVAxVO4M8iS0jtosO6_MIY0"
TRANSLATE_URL = "https://www.googleapis.com/language/translate/v2?key=" + API_KEY  # &q=hello%20world&source=en&target=de


def unicode_urlencode(params):
    if isinstance(params, dict):
        params = params.items()
    return urllib.urlencode([(k, isinstance(v, unicode) and v.encode('utf-8') or v) for k, v in params])


def make_request(url):
    return urllib.urlopen(url).read()


def quick_translate(text, target, source):
    source_text = text.replace("\n", "<br>")
    translated_html = translate(source_text, target, source)["data"]["translations"][0]["translatedText"]
    translated_text = translated_html.replace(" <br> ", "\n").replace("<br> ", "\n").replace(" <br>", "\n")
    return translated_text


def translate(text, target, source):
    query_params = {"q": text, "source": source, "target": target}
    url = TRANSLATE_URL + "&" + unicode_urlencode(query_params)
    return json.loads(make_request(url))


if __name__ == '__main__':
    if len(sys.argv) == 3:
        source, dest = sys.argv[1:]
        text = sys.stdin.read().rstrip()
    else:
        source, dest, text = sys.argv[1:]

    print quick_translate(text, dest, source).rstrip()

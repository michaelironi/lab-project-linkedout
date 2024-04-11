from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

proxies = {
    'http': 'http://brd-customer-hl_80709a30-zone-ds_dc:rif7p5j3fx5n@brd.superproxy.io:22225',
    'https': 'http://brd-customer-hl_80709a30-zone-ds_dc:rif7p5j3fx5n@brd.superproxy.io:22225',
}

def get_posts(html):
    soup = BeautifulSoup(html, 'html.parser')
    posts = []

    for item in soup.select('li.profile-creator-shared-feed-update__container'):
        title_elem = item.select_one('.update-components-text')
        title = title_elem.get_text(strip=True) if title_elem else ''

        attribution_elem = item.select_one('.update-components-actor__name')
        attribution = attribution_elem.get_text(strip=True) if attribution_elem else ''

        img_elem = item.select_one('.ivm-view-attr__img--aspect-fill')
        img_url = img_elem['src'] if img_elem else ''

        link_elem = item.select_one('.update-components-actor__meta-link')
        link = urljoin('https://www.linkedin.com', link_elem['href']) if link_elem else ''

        description_elem = item.select_one('.feed-shared-inline-show-more-text')
        description = description_elem.get_text(strip=True) if description_elem else ''

        created_at_elem = item.select_one('.update-components-actor__sub-description')
        created_at = created_at_elem.get_text(strip=True) if created_at_elem else ''

        reactions_elem = item.select_one('.social-details-social-counts__reactions-count')
        reactions = int(reactions_elem.get_text(strip=True)) if reactions_elem else 0

        post = {
            'title': title,
            'attribution': attribution,
            'img_url': img_url,
            'link': link,
            'description': description,
            'created_at': created_at,
            'reactions': reactions
        }
        posts.append(post)

    return posts


if __name__ == '__main__':
    url = 'https://www.linkedin.com/in/daniel-sinitsky/recent-activity/all/'
    try:
        res  = requests.get(url, proxies=proxies, verify=False)
        res.raise_for_status()
        if res .status_code == 200:
            posts = get_posts(res.text)
            for post in posts:
                print(post)
        else:
            print(f'Failed to fetch URL: {res .status_code}')
    except requests.exceptions.RequestException as e:
        print(f'Error occurred: {e}')
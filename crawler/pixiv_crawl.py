#!/usr/bin/env python3
#
"""
usage: pixiv_crawl.py [-h] --target TARGET [--freq FREQ] --limit LIMIT [--path PATH]
                      [--username USERNAME] [--password PASSWORD] [--disable-headless] [--clean]     

Pixiv Crawler with undetected-chromedriver

options:
  -h, --help            show this help message and exit
  --target TARGET, -t TARGET
                        CSV file with a 'keyword' field
  --freq FREQ, -f FREQ  Number of images to crawl per minute
  --limit LIMIT, -l LIMIT
                        Number of images per keyword
  --path PATH, -p PATH  Dirctionary to save images
  --username USERNAME, -u USERNAME
                        Pixiv username
  --password PASSWORD, -pw PASSWORD
                        Pixiv password
  --disable-headless    Disable headless mode
  --clean, -c           Clean saved seen_urls and cookie
  
Remarks:
    1. to fetching more than ten pages, provides username and password
    2. target.csv should have a field name "keyword"
    3. freq does not matter, the script it self is slow enough
    4. when user and password are provided, login is refreshed. otherwise, saved cookie will be loaded if exists
"""

import argparse
import csv
import os
import time
import random
import requests
import urllib.parse
import undetected_chromedriver as uc
import pickle
from selenium.webdriver.common.by import By
from tqdm import tqdm

# hyper parameters
sample_scale = 0.5
min_delay_scale = 0.2
last_pause_time = None
interval_btw_pause = 180  # seconds
interval_of_pause = 30 # seconds
last_delay_time = None
delay_per_page = 5 # seconds

max_retry = 3


# workaround to surpress insignificant quit exception
# see https://github.com/ultrafunkamsterdam/undetected-chromedriver/issues/955
class Chrome(uc.Chrome):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def quit(self):
        try:
            super().quit()
        except OSError:
            pass


def sleep_scheduler(avg_delay):
    global last_pause_time, last_delay_time
    
    if last_pause_time is None:
        last_pause_time = time.time()
    if last_delay_time is None:
        last_delay_time = time.time()
        
    sample_std = avg_delay * sample_scale  # Standard deviation as 50% of average delay
    min_delay = avg_delay * min_delay_scale  # Minimum delay as 20% of average delay
    delay = max(random.gauss(avg_delay, sample_std), min_delay)
    
    # perform delay (deducted time already elapsed since last delay) 
    delay = max(delay - (time.time() - last_delay_time), 0)
    if delay > 0:
        time.sleep(delay)
    last_delay_time = time.time()

    if time.time() - last_pause_time >= interval_btw_pause:        # Pause for a longer duration
        print(f"\nLong pause for {interval_of_pause} sec")
        time.sleep(interval_of_pause)
        last_pause_time = time.time()
        
def mimic_user_interaction(driver): 
    # change focus
    driver.execute_script("window.focus();")
    driver.execute_script("document.dispatchEvent(new Event('visibilitychange'));")
    
    # driver.find_element(By.TAG_NAME, "body").click()  # Click on the leftmost side
    time.sleep(1)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 3);") # Scroll to middle 1/3
    time.sleep(1)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 3 * 2);")  # Scroll to middle 2/3
    time.sleep(1)
    driver.execute_script("window.scrollTo(0, 0);")  # Scroll to top
    time.sleep(1)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # Scroll to bottom
    time.sleep(1)

    time.sleep(delay_per_page - 5) # Wait for the rest of the delay
    
def get_image_urls_from_page(driver, keyword, page):
    # get page url with no r18
    base_url = "https://www.pixiv.net/tags/{}/illustrations?p={}&mode=safe"
    encoded_keyword = urllib.parse.quote(keyword)
    url = base_url.format(encoded_keyword, page)
    print(f"\nFetching: {url}")
    driver.get(url)
    
    # Emulate some user interactions to ensure the page loads completely
    mimic_user_interaction(driver)
    
    print(f"Fetching complete")
    
    # Check if the URL is correct (redirect may happen if not logined)
    if url != driver.current_url:
        print(f"[Warning]: current URL '{driver.current_url}' have changed")

    imgs = driver.find_elements(By.TAG_NAME, "img")
    image_urls = []
    for img in imgs:
        src = img.get_attribute("src")
        def criteria(src):
            return src and "i.pximg.net" in src and "img-master" in src
        if criteria(src):
            image_urls.append(src)
    
    result = list(dict.fromkeys(image_urls))
    print(f"Found {len(result)} images on page {page} for '{keyword}'")
    return result

def download_image(url, filepath):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://www.pixiv.net/"
    }
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=10)
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        # print(f"Downloaded: {filepath}")
    except Exception as e:
        print(f"Failed to download image {url}: {e}, retry")
        return False
    return True
    
def get_extension_from_url(url):
    path = urllib.parse.urlparse(url).path
    _, ext = os.path.splitext(path)
    return ext.split('?')[0] if ext else ".jpg"

def read_seen_urls_from_file(seen_urls_path):
    seen_urls = set()
    seen_urls_file = open(seen_urls_path, "a+")
    seen_urls_file.seek(0)
    for line in seen_urls_file:
        seen_urls.add(line.strip())
    return seen_urls, seen_urls_file

def get_uc_driver(headless):
    options = uc.ChromeOptions()
    if headless:
        options.add_argument('headless')  # Run in headless mode
    driver = Chrome(options=options)
    return driver
        
def login_to_pixiv(username, password, cookie_path):
    if os.path.exists(cookie_path):
        print("Cookie already exists. Skip interactive logging")
        
    login_url = "https://accounts.pixiv.net/login"
    
    options = uc.ChromeOptions()
    # create interactive driver
    driver = Chrome(options=options)
    driver.get(login_url)
    
    # Wait for the login page to load
    time.sleep(2)
    
    # Find and fill the username and password fields
    username_field = driver.find_element(By.CSS_SELECTOR, "input[type='text']")
    password_field = driver.find_element(By.CSS_SELECTOR, "input[type='password']")
    
    username_field.send_keys(username)
    password_field.send_keys(password)
    
    # Find and click the login button with more specific selector
    login_button = driver.find_element(By.CSS_SELECTOR, "button.charcoal-button[data-variant='Primary'][data-full-width='true']")
    login_button.click()
    
    # Wait for the login process to complete
    time.sleep(2)
    
    # Check if login was successful
    if "pixiv.net" in driver.current_url:
        # Save cookies to file
        with open(cookie_path, "wb") as f:
            pickle.dump(driver.get_cookies(), f)
            for cookie in driver.get_cookies():
                print(cookie)
        driver.quit()
        print("Login successful")
    else:
        print("Login failed")
        driver.quit()
        exit(1)

def load_cookies(driver, cookie_path):
    if os.path.exists(cookie_path):
        with open(cookie_path, "rb") as f:
            cookies = pickle.load(f)
            driver.get("https://www.pixiv.net") # navigate to proper domain
            for cookie in cookies:
                driver.add_cookie(cookie)
        print("Cookies loaded successfully")
    else:
        print("No cookie file found")
        raise FileNotFoundError(f"Cookie file {cookie_path} not found")
    
def clean_saved_info(seen_urls_path, cookie_path):
    if os.path.exists(seen_urls_path):
        os.remove(seen_urls_path)
        print(f"Cleaned saved seen_urls: {seen_urls_path}")
    else:
        print(f"Seen URL {seen_urls_path} does not exist. Skip clean")
        
    if os.path.exists(cookie_path):
        os.remove(cookie_path)
        print(f"Cleaned saved cookies: {cookie_path}")
    else:
        print(f"Cookies {cookie_path} does not exist. Skip clean")
        
def skip_exisiting_data(path_keyword, downloaded, limit):
    # skip existing data
    skip_cnt = 0
    while downloaded < limit:
        ext_set = {"jpg", "jpeg", "png", "webp"}
        exists = False
        for ext in ext_set:
            filepath = os.path.join(path_keyword, f"{downloaded+1}.{ext}")
            if os.path.exists(filepath):
                exists = True
                break
        if exists:
            downloaded += 1
            skip_cnt += 1
        else:
            break
    return downloaded, skip_cnt
def main():
    parser = argparse.ArgumentParser(description="Pixiv Crawler with undetected-chromedriver")
    parser.add_argument('--target', '-t', required=True, help="CSV file with a 'keyword' field")
    parser.add_argument('--freq', '-f', type=int, default=60, help="Number of images to crawl per minute")
    parser.add_argument('--limit', '-l', type=int, required=True, help="Number of images per keyword")
    parser.add_argument('--path', '-p', type=str, default="data/", help="Dirctionary to save images")
    parser.add_argument('--username', '-u', type=str, default='', help="Pixiv username")
    parser.add_argument('--password', '-pw', type=str, default='', help="Pixiv password")
    parser.add_argument('--disable-headless', action='store_true', help="Disable headless mode")
    parser.add_argument('--clean', '-c', action='store_true', help="Clean saved seen_urls and cookie")
    
    args = parser.parse_args()
    
    # Precompute paths
    seen_urls_path = os.path.join(args.path, "seen_urls.txt")
    cookie_path = os.path.join(args.path, "pixiv_cookies.txt")
    
    if args.clean:
        clean_saved_info(seen_urls_path, cookie_path)
    
    # Compute delay between downloads (in seconds)
    delay = 60.0 / args.freq

    # Ensure the data directory exists.
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    
    # Get driver
    driver = get_uc_driver(not args.disable_headless)
    
    if args.username and args.password:
        print("Logging in to Pixiv and save cookies ...")
        login_to_pixiv(args.username, args.password, cookie_path)
        load_cookies(driver, cookie_path)
    else:
        print("No login credentials provided. Attempting to load cookies ...")
        try:
            load_cookies(driver, cookie_path)
        except FileNotFoundError:
            print("Proceeding without login. If this is not desired, CTRL+C now")
        
    
    assert os.path.exists(args.target), f"File {args.target} does not exist"

    with open(args.target, newline='', encoding='utf-8') as csvfile:
        seen_urls, seen_urls_file = read_seen_urls_from_file(seen_urls_path)
        
        reader = csv.DictReader(csvfile)
        for row in reader: # process each keyword
            keyword = row.get("keyword")
            
            
            if not keyword:
                print("keyword is None, skip")
                continue
            
            args.path_keyword = os.path.join("data", keyword)
            if not os.path.exists(args.path_keyword):
                os.makedirs(args.path_keyword)
            
            print(f"Processing keyword: {keyword}")
            downloaded = 0
            page = 1

            # Continue paging until the limit is reached.
            
            # progress bar
            pbar = tqdm(total=args.limit, desc=f"Downloading images for '{keyword}'")

            # skip existing data at beginning
            downloaded, skip_cnt = skip_exisiting_data(args.path_keyword, downloaded, args.limit)
            pbar.update(skip_cnt)
            
            # actual fetching
            while downloaded < args.limit:
                
                retry = 0
                while retry < max_retry:
                    image_urls = get_image_urls_from_page(driver, keyword, page)
                    if len(image_urls) == 0:
                        print(f"No images found on page {page} for '{keyword}'. Retry")
                        retry += 1
                    else:
                        break
                    
                if retry == max_retry:
                    print(f"Failed to fetch images after {max_retry} attempts. Skip.")
                    break
                
                seen_cnt = 0
                for url in image_urls:
                    if url in seen_urls:
                        seen_cnt += 1
                        continue
                    
                    # skip existing data
                    downloaded, skip_cnt = skip_exisiting_data(args.path_keyword, downloaded, args.limit)
                    pbar.update(skip_cnt)
                    
                    if downloaded >= args.limit:
                        break
                    
                    # add to seen_urls
                    seen_urls.add(url)
                    seen_urls_file.write(url + "\n")
                    
                    # save to path/keyword/
                    ext = get_extension_from_url(url)
                    filepath = os.path.join(args.path_keyword, f"{downloaded+1}{ext}")
                    
                    succeed = download_image(url, filepath)
                    
                    if succeed:
                        downloaded += 1
                        pbar.update(1)
                    
                    sleep_scheduler(delay)
                
                # report seen_urls count
                if seen_cnt == len(image_urls):
                    print(f"\nAll images on page {page} for '{keyword}' have been seen. Skip")
                elif seen_cnt:
                    print(f"\nSkipped {seen_cnt} already seen images (but not all) on page {page} for '{keyword}'")
                else:
                    print(f"")
                page += 1
            
            pbar.close()
            print("\n")
    print("> All keywords successfully processed. Quit")
    driver.quit()
    exit(0)
        
if __name__ == "__main__":
    main()
    
"""
Example:
In main workspace directory
python crawler/pixiv_crawl.py -l 1000 -f 100 -t crawler/th_name_pretest.csv -p data
"""

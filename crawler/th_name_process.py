import random
import time
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import undetected_chromedriver as uc

options = uc.ChromeOptions()
# options.add_argument("--disable-blink-features=AutomationControlled")
# options.add_argument("headless")  # 无头模式

driver = uc.Chrome(options=options)

def browser_based_scraping(tag):
    driver.get(f"https://www.pixiv.net/tags/{tag}/artworks")
    
    for i in range(10):
        # 执行人类化操作
        driver.execute_script("window.scrollBy(0, 500)")
        time.sleep(random.uniform(0.5, 2))
    
    # 获取页面源码
    html = driver.page_source
    return html

def get_illust_description_by_tag(tag):
    # url = f"https://www.pixiv.net/tags/{tag}/artworks"
    # headers = {
    #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    #     "Referer": "https://www.pixiv.net/"
    # }
    
    try:
        # response = requests.get(url, headers=headers)
        # response.raise_for_status()
        # soup = BeautifulSoup(response.text, "html.parser")

        html = browser_based_scraping(tag)
        soup = BeautifulSoup(html, "html.parser")
        
        # 从meta标签的description中提取数量
        meta_description = soup.find("meta", {"name": "description"})
        if not meta_description:
            print("未找到描述信息")
            return -1
            
        description = meta_description.get("content", "")
        return description
            
    except Exception as e:
        print(f"请求失败: {e}")
        return ""

def find_count(description):
    # 匹配模式：寻找 "XXX件" 的格式
    match = re.search(r"((\d+[,]?)*\d+)件", description)
    
    if match:
        count_str = match.group(1).replace(",", "")  # 处理千分位逗号
        return int(count_str)
    else:
        print(f"未在描述中找到插画数量：{description}")
        return -1

def find_related_tag(description: str, related_tag):
    return related_tag in description

def process_name(name: str):
    '''
    Yields processed names.
    '''
    if '/' in name:
        for new_name in name.split('/'):
            process_name(new_name)
    else:
        yield name # Original name
        spliter = ["　", "・", " ", '·']
        for sp in spliter:
            if sp in name:
                yield name.replace(sp, "")
                for s in name.split(sp):
                    yield s.strip()

def filter_tags_by_count_and_related_tags_saving_to(tags : pd.DataFrame, related_tag, save_csv):
    '''
    Filter tags each row by finding max count with related tags.
    '''
    with open(save_csv, 'w', newline='', encoding='utf-8') as csvfile:
        print("Name,Tag,Cnt", file=csvfile)
        filtered_tags = []
        for index, data in tags.iterrows():
            zh_name = tags['CH'][index]
            print(f"Processing {zh_name}:")
            target_tag = zh_name
            max_cnt = 0
            for raw_name in data:
                for name in process_name(raw_name):
                    # time.sleep(random.uniform(30, 60))
                    des = get_illust_description_by_tag(name)
                    cnt = find_count(des)
                    print(f"Checking {name}...counts:{cnt}")
                    print(des)
                    if (find_related_tag(des, related_tag) and cnt > max_cnt):
                        max_cnt = cnt
                        target_tag = name
            filtered_tags.append({"Name": zh_name, "Tag": target_tag, "Cnt": max_cnt})
            print(f"Chose {target_tag} as the most related tag for {zh_name}!")
            print(f"{zh_name},{target_tag},{max_cnt}", file=csvfile)
            csvfile.flush()
        return filtered_tags

def preprocess(touhou_tag, raw_tag_csv, target_tag_csv):
    raw_tags = pd.read_csv(raw_tag_csv)
    filter_tags_by_count_and_related_tags_saving_to(raw_tags, touhou_tag, target_tag_csv)

touhou_tag = '东方Project'
raw_tag_csv = 'th_name_raw.csv'
target_tag_csv = 'th_name_processed.csv'

if __name__ == "__main__":
    preprocess(touhou_tag, raw_tag_csv, target_tag_csv)
    driver.quit()
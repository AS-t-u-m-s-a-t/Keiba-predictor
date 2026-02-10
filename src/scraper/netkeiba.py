"""netkeiba.comからレースデータをスクレイピングするモジュール

このモジュールは競馬データをnetkeiba.comから取得します。
サーバーに負荷をかけないよう、リクエスト間隔を設定しています。
"""

import time
import csv
from typing import List, Dict, Optional
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from .parser import RaceParser


class NetkeibaScaper:
    """netkeiba.comからデータをスクレイピングするクラス"""
    
    BASE_URL = "https://db.netkeiba.com"
    
    def __init__(self, delay: float = 1.0, output_dir: str = "data/raw"):
        """
        Args:
            delay: リクエスト間の待機時間（秒）
            output_dir: データ保存先ディレクトリ
        """
        self.delay = delay
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parser = RaceParser()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_race_page(self, race_id: str) -> Optional[str]:
        """レースページのHTMLを取得
        
        Args:
            race_id: レースID（例: 202305010101）
            
        Returns:
            HTML文字列。失敗時はNone
        """
        url = f"{self.BASE_URL}/race/{race_id}"
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching race {race_id}: {e}")
            return None
    
    def scrape_race(self, race_id: str) -> Optional[Dict]:
        """単一レースのデータをスクレイピング
        
        Args:
            race_id: レースID
            
        Returns:
            レースデータの辞書。失敗時はNone
        """
        html = self.fetch_race_page(race_id)
        if html is None:
            return None
        
        try:
            race_data = self.parser.parse_race(html, race_id)
            return race_data
        except Exception as e:
            print(f"Error parsing race {race_id}: {e}")
            return None
    
    def scrape_races(self, race_ids: List[str], output_file: str = "races.csv") -> List[Dict]:
        """複数レースをスクレイピングしてCSVに保存
        
        Args:
            race_ids: レースIDのリスト
            output_file: 出力ファイル名
            
        Returns:
            スクレイピングしたデータのリスト
        """
        all_data = []
        output_path = self.output_dir / output_file
        
        print(f"Starting to scrape {len(race_ids)} races...")
        
        for i, race_id in enumerate(race_ids, 1):
            print(f"Scraping race {i}/{len(race_ids)}: {race_id}")
            
            race_data = self.scrape_race(race_id)
            if race_data and 'results' in race_data:
                all_data.extend(race_data['results'])
        
        if all_data:
            self._save_to_csv(all_data, output_path)
            print(f"Saved {len(all_data)} records to {output_path}")
        
        return all_data
    
    def _save_to_csv(self, data: List[Dict], filepath: Path):
        """データをCSVファイルに保存
        
        Args:
            data: 保存するデータのリスト
            filepath: 保存先のパス
        """
        if not data:
            return
        
        fieldnames = list(data[0].keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    
    def generate_race_ids(self, year: int, month: int, track_code: str = "05", 
                         day: int = 1, race_num: int = 12) -> List[str]:
        """レースIDのリストを生成
        
        Args:
            year: 年（4桁）
            month: 月
            track_code: 競馬場コード（01:札幌, 02:函館, 03:福島, 04:新潟, 05:東京, 
                                      06:中山, 07:中京, 08:京都, 09:阪神, 10:小倉）
            day: 開催日数
            race_num: レース数
            
        Returns:
            レースIDのリスト
        """
        race_ids = []
        for d in range(1, day + 1):
            for r in range(1, race_num + 1):
                # フォーマット: YYYYMMTTDDRRR
                # YYYY: 年, MM: 月, TT: 競馬場, DD: 日, RRR: レース番号
                race_id = f"{year}{month:02d}{track_code}{d:02d}{r:02d}"
                race_ids.append(race_id)
        return race_ids

"""HTMLパース処理モジュール

netkeiba.comのHTMLをパースしてデータを抽出します。
"""

import re
from typing import Dict, List, Optional
from bs4 import BeautifulSoup


class RaceParser:
    """レースページをパースするクラス"""
    
    def parse_race(self, html: str, race_id: str) -> Dict:
        """レースページからデータを抽出
        
        Args:
            html: レースページのHTML
            race_id: レースID
            
        Returns:
            レース情報と結果を含む辞書
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # レース情報を取得
        race_info = self._parse_race_info(soup, race_id)
        
        # レース結果を取得
        results = self._parse_race_results(soup, race_info)
        
        return {
            'race_info': race_info,
            'results': results
        }
    
    def _parse_race_info(self, soup: BeautifulSoup, race_id: str) -> Dict:
        """レース基本情報を抽出
        
        Args:
            soup: BeautifulSoupオブジェクト
            race_id: レースID
            
        Returns:
            レース情報の辞書
        """
        race_info = {'race_id': race_id}
        
        # レース名
        try:
            race_name = soup.find('div', class_='RaceName')
            if race_name:
                race_info['race_name'] = race_name.get_text(strip=True)
        except:
            race_info['race_name'] = ''
        
        # レース条件（距離、馬場、天候など）
        try:
            race_data = soup.find('div', class_='RaceData01')
            if race_data:
                text = race_data.get_text(strip=True)
                
                # 距離を抽出（例: "芝2000m" → 2000）
                distance_match = re.search(r'(\d+)m', text)
                if distance_match:
                    race_info['distance'] = int(distance_match.group(1))
                
                # コース種別（芝/ダート）
                if '芝' in text:
                    race_info['course_type'] = '芝'
                elif 'ダート' in text or 'ダ' in text:
                    race_info['course_type'] = 'ダート'
                else:
                    race_info['course_type'] = ''
        except:
            race_info['distance'] = 0
            race_info['course_type'] = ''
        
        # 天候・馬場状態
        try:
            race_data2 = soup.find('div', class_='RaceData02')
            if race_data2:
                text = race_data2.get_text()
                
                # 天候
                weather_match = re.search(r'天候:(\S+)', text)
                if weather_match:
                    race_info['weather'] = weather_match.group(1)
                
                # 馬場状態
                track_match = re.search(r'馬場:(\S+)', text)
                if track_match:
                    race_info['track_condition'] = track_match.group(1)
        except:
            race_info['weather'] = ''
            race_info['track_condition'] = ''
        
        return race_info
    
    def _parse_race_results(self, soup: BeautifulSoup, race_info: Dict) -> List[Dict]:
        """レース結果を抽出
        
        Args:
            soup: BeautifulSoupオブジェクト
            race_info: レース情報
            
        Returns:
            各馬の結果のリスト
        """
        results = []
        
        # 結果テーブルを取得
        result_table = soup.find('table', class_='RaceTable01')
        if not result_table:
            return results
        
        rows = result_table.find_all('tr')
        
        for row in rows[1:]:  # ヘッダー行をスキップ
            cols = row.find_all('td')
            if len(cols) < 10:
                continue
            
            try:
                result = race_info.copy()
                
                # 着順
                result['finish_position'] = self._clean_text(cols[0].get_text())
                
                # 枠番
                result['frame_number'] = self._clean_text(cols[1].get_text())
                
                # 馬番
                result['horse_number'] = self._clean_text(cols[2].get_text())
                
                # 馬名
                horse_link = cols[3].find('a')
                if horse_link:
                    result['horse_name'] = horse_link.get_text(strip=True)
                    result['horse_id'] = self._extract_id(horse_link.get('href', ''))
                
                # 性齢
                result['sex_age'] = self._clean_text(cols[4].get_text())
                
                # 斤量
                result['weight'] = self._clean_text(cols[5].get_text())
                
                # 騎手
                jockey_link = cols[6].find('a')
                if jockey_link:
                    result['jockey_name'] = jockey_link.get_text(strip=True)
                    result['jockey_id'] = self._extract_id(jockey_link.get('href', ''))
                
                # タイム
                result['time'] = self._clean_text(cols[7].get_text())
                
                # 着差
                result['margin'] = self._clean_text(cols[8].get_text())
                
                # 人気
                if len(cols) > 10:
                    result['popularity'] = self._clean_text(cols[10].get_text())
                
                # オッズ
                if len(cols) > 11:
                    result['odds'] = self._clean_text(cols[11].get_text())
                
                results.append(result)
                
            except Exception as e:
                print(f"Error parsing row: {e}")
                continue
        
        return results
    
    def _clean_text(self, text: str) -> str:
        """テキストをクリーニング
        
        Args:
            text: 元のテキスト
            
        Returns:
            クリーニングされたテキスト
        """
        return text.strip().replace('\n', '').replace('\xa0', '')
    
    def _extract_id(self, url: str) -> str:
        """URLからIDを抽出
        
        Args:
            url: URL文字列
            
        Returns:
            抽出されたID
        """
        match = re.search(r'/(\d+)/?', url)
        if match:
            return match.group(1)
        return ''

import os
import json
from math import sqrt
from datetime import timedelta, datetime

import pandas as pd
import requests

from batch_classes import Pickup, Dropoff, Order, LocationType

import warnings
warnings.filterwarnings(action='ignore')

# Excel 데이터를 Location 및 Order 객체로 변환하는 함수
def preprocess_excel_to_objects(file_path: str, sheet_name: str, order_num=20):
    # SameDay Orders(8 Hours SLA)
    # Instant Orders(1-4 Hours SLA)
    # Mixed Orders(SameDay+Instant)
    
    if sheet_name == 'SameDay Orders(8 Hours SLA)':
        distance_threshold = 5
    elif sheet_name == 'Instant Orders(1-4 Hours SLA)':
        distance_threshold = 10
    elif sheet_name == 'Mixed Orders(SameDay+Instant)':
        distance_threshold = 5
    
    # Excel 파일에서 특정 시트 읽기
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Order, Location 객체 리스트 저장
    orders = []
    locations = []

    for index, row in df.iterrows():
        # 각 행의 데이터를 기반으로 Pickup과 Dropoff 객체 생성
        order_id = row['Order ID']
        weight = row['Weight(KG)']
        
        pickup_location_xy = (row['Pickup Latitude'], row['Pickup Longitude'])
        dropoff_location_xy = (row['Dropoff Latitude'], row['Dropoff Longitude'])
        travel_time = calculate_travel_time(pickup_location_xy, dropoff_location_xy)

        base_date = datetime.today().date()  # 오늘 날짜를 기준으로 사용
        time_window = (
            datetime.combine(base_date, row['Pickup Time']),
            datetime.combine(base_date, row['Delivery Time'])
        )
        pickup_time_window = (
            datetime.combine(base_date, row['Pickup Time']),
            datetime.combine(base_date, row['Delivery Time']) - travel_time
        )
        dropoff_time_window = (
            datetime.combine(base_date, row['Pickup Time']) + travel_time,
            datetime.combine(base_date, row['Delivery Time'])
        )
        
        # Pickup과 Dropoff 객체 생성 및 추가
        pickup = Pickup(order_id=order_id, location_xy=pickup_location_xy, time_window=pickup_time_window, weight=weight)
        dropoff = Dropoff(order_id=order_id, location_xy=dropoff_location_xy, time_window=dropoff_time_window)
        locations.append(pickup)
        locations.append(dropoff)
        
        # Order 객체 생성 및 추가
        order = Order(pickup=pickup, dropoff=dropoff, time_window=time_window)
        orders.append(order)

    orders.sort(key=lambda order: order.time_window[0])
    orders = orders[:order_num]

    locations = []
    for order in orders:
        locations.append(order.pickup)
        locations.append(order.dropoff)

    return orders, locations


def calculate_travel_time(pickup_location_xy, dropoff_location_xy, speed_kmh=50):
    """
    두 좌표 간 직선 거리와 이동 시간을 계산하는 함수.
    
    Parameters:
        pickup_location_xy (tuple): 픽업 지점의 (위도, 경도)
        dropoff_location_xy (tuple): 드롭오프 지점의 (위도, 경도)
        speed_kmh (float): 이동 속도 (km/h). 기본값은 50km/h.
    
    Returns:
        timedelta: 이동 시간
    """
    # 1도는 111km로 가정하여 위도와 경도의 차이를 거리로 변환
    lat_diff = abs(pickup_location_xy[0] - dropoff_location_xy[0]) * 111
    lon_diff = abs(pickup_location_xy[1] - dropoff_location_xy[1]) * 111
    
    # 피타고라스 정리를 사용하여 직선 거리 계산
    distance_km = sqrt(lat_diff**2 + lon_diff**2)
    
    # 이동 시간 계산 (시간 단위)
    travel_time_hours = distance_km / speed_kmh
    return timedelta(hours=travel_time_hours)

def split_list(lst, n):
    """리스트를 n개씩 나누는 함수"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_distance_matrix(sheet_name, locations, API_KEY):
    response_list = []

    matrix = {}
    if os.path.exists('matrix.json'):
        with open('matrix.json', 'r', encoding='utf-8') as f:
            matrix = json.load(f)
            matrix = convert_to_float(matrix)

        if isinstance(matrix, tuple):
            matrix = matrix[0]
            
        matrix_keys_to_add = [loc_to_matrix_id(sheet_name, loc) for loc in locations]
        keys_not_added = False
        for origin_matrix_id in matrix_keys_to_add:
            if origin_matrix_id not in matrix or keys_not_added:
                keys_not_added = True
                break
            else:
                for destination_matrix_id in matrix_keys_to_add:
                    if origin_matrix_id != destination_matrix_id:
                        if destination_matrix_id not in matrix[origin_matrix_id]:
                            keys_not_added = True
                            break
                            
        if not keys_not_added:
            return matrix, None

    locations_xy = [loc.location for loc in locations]
    loc_batches = list(split_list(locations, 10))

    for origin_batch in loc_batches:
        origin_batch_xy = [loc.location for loc in origin_batch]
        for destination_batch in loc_batches:
            destination_batch_xy = [loc.location for loc in destination_batch]

            is_not_added = False
            origin_matrix_ids = [loc_to_matrix_id(sheet_name, loc) for loc in origin_batch]
            for origin_matrix_id in origin_matrix_ids:
                if origin_matrix_id not in matrix or is_not_added:
                    is_not_added = True
                    break
                else:
                    destination_matrix_ids = [loc_to_matrix_id(sheet_name, loc) for loc in destination_batch]
                    for destination_matrix_id in destination_matrix_ids:
                        if destination_matrix_id not in matrix[origin_matrix_id]:
                            is_not_added = True
                            break

            if is_not_added:
            
                origins_str = "|".join([f"{lat},{lon}" for lat, lon in origin_batch_xy])
                destinations_str = "|".join([f"{lat},{lon}" for lat, lon in destination_batch_xy])
            
                url = f"https://maps.googleapis.com/maps/api/distancematrix/json"
                params = {
                    "origins": origins_str,
                    "destinations": destinations_str,
                    "key": API_KEY,
                }
            
                departure_time = "2026-03-22 14:00:00"  # general time, it needed to be current or future time
            
                dt = datetime.strptime(departure_time, "%Y-%m-%d %H:%M:%S")
                unix_timestamp = int(dt.timestamp())
                params["departure_time"] = unix_timestamp
            
                print("Sending google-api request...")
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    response_list.append(response)
                    matrix = parse_distance_matrix(matrix, sheet_name, response.json(), origin_batch, destination_batch, is_store=True)
                else:
                    print(f"Error: {response.status_code}, {response.text}")
                    return None, None

    return matrix, response_list

def loc_to_matrix_id(sheet_name, loc):
    if sheet_name == "SameDay Orders(8 Hours SLA)":
        sheet_i = 1
    elif sheet_name == "Instant Orders(1-4 Hours SLA)":
        sheet_i = 2
    else:
        sheet_i = 3
    
    if loc.location_type == LocationType.PICKUP:
        location_i = 1
    else:
        location_i = 2
        
    return 10000 * sheet_i + 1000 * location_i + loc.order_id

def parse_distance_matrix(matrix, sheet_name, response, origin_batch, destination_batch, is_store=True):
    origins = response['origin_addresses']
    destinations = response['destination_addresses']
    rows = response['rows']
    
    for i, _ in enumerate(origin_batch):
        origin = loc_to_matrix_id(sheet_name, origin_batch[i])

        if origin not in matrix:
            matrix[origin] = {}
        for j, _ in enumerate(destination_batch):
            destination = loc_to_matrix_id(sheet_name, destination_batch[j])
            
            element = rows[i]['elements'][j]
            if element['status'] == 'OK':
                matrix[origin][destination] = {
                    'distance': element['distance']['value'] / 1000,  # 거리(km)
                    'duration_traffic': element['duration_in_traffic']['value'] / 3600,  # 시간(hours)
                    'duration_general' : element['duration']['value'] / 3600,  # 시간(hours)
                }

    if is_store:
        if os.path.exists('matrix.json'):
            with open('matrix.json', 'r', encoding='utf-8') as f:
                overwrited_matrix = json.load(f)
            
            overwrited_matrix = convert_to_float(overwrited_matrix)
            if isinstance(overwrited_matrix, tuple):
                overwrited_matrix = overwrited_matrix[0]
            for origin, destinations in matrix.items():
                if origin not in overwrited_matrix:
                    overwrited_matrix[origin] = {} 
            
                for destination, values in destinations.items():
                    if destination not in overwrited_matrix[origin]:
                        # overwrited_matrix에 없는 키-쌍만 업데이트
                        overwrited_matrix[origin][destination] = values

        else:
            overwrited_matrix = matrix
        
        with open('matrix.json', 'w', encoding='utf-8') as f:
            json.dump(overwrited_matrix, f, indent=4)
    
    return matrix

def convert_to_float(d):
    new_dict = {}
    for key, value in d.items():
        # 키를 float으로 변환
        try:
            new_key = float(key)
        except ValueError:
            new_key = key
        
        if isinstance(value, dict):
            # 값이 딕셔너리인 경우 재귀 호출
            new_value = convert_to_float(value)
        else:
            # 값이 딕셔너리가 아닌 경우 float으로 변환
            try:
                new_value = float(value)
            except ValueError:
                new_value = value

        # 새로운 딕셔너리에 추가
        new_dict[new_key] = new_value

    return new_dict
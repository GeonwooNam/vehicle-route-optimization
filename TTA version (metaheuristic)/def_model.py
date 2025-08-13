import copy
import time
import math
from datetime import timedelta
from datetime import datetime

import numpy as np
import pandas as pd
from random import *
from tqdm import tqdm
import matplotlib.pyplot as plt

from itertools import permutations
from sklearn.cluster import KMeans


## 1. 편의 기능
# 1-1. 경로가 시간 제약 준수하는지 확인
def time_checker(tour, travel_time, due_time, tour_start_time):
    
    # 1. 경유지가 없을 경우 True 반환
    if len(tour) <= 2:
        return True
    
    time = tour_start_time
    
    # 2. 각 경유지에서 시간 확인
    for i in range(1, len(tour)):
        travel_duration = travel_time[tour[i-1]][tour[i]]
        
        # 2-1. 경로가 존재하지 않으면 중단
        if travel_duration == float("inf"):
            return False
        
        time += pd.to_timedelta(travel_duration, unit='m')
        
        # 2-2. 시간 제한을 넘으면 False 반환
        if time > due_time[tour[i]]:
            return False
    
    return True


# 1-2. 주문 마감기한 계산
def calculate_duetime(row):
   
    if row['service_type'] == 'sameday_8_hours':
        return row['midmile_pickup_done_at'] + timedelta(hours=8)
    elif row['service_type'] == 'sameday_ev':
        # Set duetime to 23:59 on the same day
        return row['midmile_pickup_done_at'].replace(hour=23, minute=59, second=59)
    elif row['service_type'] == 'instant_4_hours':
        return row['midmile_pickup_done_at'] + timedelta(hours=4)
    elif row['service_type'] == 'instant_2_hours':
        return row['midmile_pickup_done_at'] + timedelta(hours=2)
    elif row['service_type'] == 'sameday_5_hours':
        return row['midmile_pickup_done_at'] + timedelta(hours=5)
    elif row['service_type'] == 'regular':
        return row['midmile_pickup_done_at'] + timedelta(days=1)  # Assuming a day later for regular
    else:
        return pd.NaT  # If unknown service_type, return Not-a-Time


# 1-3. 유클리드 거리 매트릭스, 시간 매트릭스 계산
def euclidean_distance_duration_matrix(df):
    
    # 주문 목적지에 허브 3곳 좌표를 추가
    coords = np.array([df['consignee_longitude'], df['consignee_latitude']]).astype(float).T    
    hub_xy = [
        {"hub": 'Blitz Head Office', 'latitude': -6.274828818342444, 'longitude': 106.79793647301315},
        {"hub": 'Blitz Station - Jakarta Timur', 'latitude': -6.214863672845071, 'longitude': 106.91482253872306},
        {"hub": 'Blitz Station - Jakarta Barat', 'latitude': -6.171775094163178, 'longitude': 106.72794162707773},
    ]
    hub_coords = np.array([[hub['longitude'], hub['latitude']] for hub in hub_xy])
    coords = np.vstack((coords, hub_coords))
    
    lon = coords[:, 0]
    lat = coords[:, 1]
    
    distance_matrix = np.sqrt((lon[:, np.newaxis] - lon[np.newaxis, :])**2 + (lat[:, np.newaxis] - lat[np.newaxis, :])**2) * 111  # 1도에 약 111km
    duration_matrix = distance_matrix / 0.5  # 오토바이 평균 시속 30km/h으로 가정 = 0.5km/m

    return distance_matrix, duration_matrix, coords


# 1-4. 한 경로의 주문 용량 합 계산
def demand_sum(tour, demand):
    if isinstance(tour, int):
        return demand[tour]
    elif isinstance(tour, list):
        return sum([demand[i] for i in tour])


# 1-5. 한 경로의 길이 계산
def tour_distance(tour, distance):
    # 마지막 목적지 -> 물류센터로의 거리는 미포함
    if tour[0] == tour[-1]:
        tour = tour[:-1]
        
    if len(tour) <= 1:
        return 0
    elif len(tour) > 1:
        return sum([distance[tour[i - 1]][tour[i]] for i in range(1, len(tour))])


# 1-6. 솔루션의 전체 경로의 총 길이 계산
def total_distance(tours, distance):    
    return sum([tour_distance(tour, distance) for tour in tours])


# 1-7. 매트릭스 계산시 필요한 허브-숫자 변환
def blitz_hub_to_i(hub_str):
    
    if hub_str == 'Blitz Head Office':
        return -3
    elif hub_str == 'Blitz Station - Jakarta Timur':
        return -2
    elif hub_str == 'Blitz Station - Jakarta Barat':
        return -1


# 1-8. 주문 묶음의 주문 출발 시각 설정
def create_timestamp_from_time_block(time_str):
    date_part, time_range = time_str.split(' ')
    start_time, end_time = time_range.split('-')
    
    new_time = f"{date_part} {end_time}:00"
    new_dt = pd.Timestamp(datetime.strptime(new_time, "%Y-%m-%d %H:%M"))
    
    return new_dt


## 2. VND
# 1. 2-opt 탐색
def two_opt_search(tour_list, distance, travel_time, due_time, tour_start_time):
    best_imp = 0
    tour_indices, position1, position2 = [], [], []

    # 각 투어에 대해 2-opt 개선 시도
    for tour_idx, tour in enumerate(tour_list):
        if len(tour) >= 5:
            
            for i in range(len(tour) - 3):
                for j in range(i + 2, len(tour) - 1):
                    
                    # 새로운 경로 생성 (2-opt 적용)
                    new_tour = tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]
                    
                    # 시간 제약 확인
                    if time_checker(new_tour, travel_time, due_time, tour_start_time):
                        # 개선도 계산
                        imp = distance[tour[i]][tour[j]] + distance[tour[i + 1]][tour[j + 1]] - distance[tour[i]][tour[i + 1]] - distance[tour[j]][tour[j + 1]]
                        
                        # 개선이 있으면 업데이트
                        if imp < 0:
                            tour_indices.append(tour_idx)
                            position1.append(i)
                            position2.append(j)
                            best_imp += imp

    return tour_indices, position1, position2, best_imp


# 2. Or-opt 탐색
def or_opt_search(tour_list, distance, travel_time, due_time, K, tour_start_time):
    best_imp = 0
    tour_indices, position1, position2, position3 = [], [], [], []

    # 각 투어에 대해 or-opt 개선 시도
    for tour_idx, tour in enumerate(tour_list):
        if len(tour) >= K + 3:
        
            for i in range(len(tour) - K - 1):
                j = i + K
                for k in range(len(tour) - 1):
                    if (k < i) or (j < k):
                        # 새로운 경로 생성 (Or-opt 적용)
                        if k < i:
                            new_tour = tour[:k + 1] + tour[i + 1:j + 1] + tour[k + 1:i + 1] + tour[j + 1:]
                        else:
                            new_tour = tour[:i + 1] + tour[j + 1:k + 1] + tour[i + 1:j + 1] + tour[k + 1:]
                        
                        # 시간 제약 확인
                        if time_checker(new_tour, travel_time, due_time, tour_start_time):
                            # 개선도 계산
                            imp = (distance[tour[i]][tour[j + 1]] + distance[tour[k]][tour[i + 1]] + distance[tour[j]][tour[k + 1]]) \
                                   - (distance[tour[i]][tour[i + 1]] + distance[tour[j]][tour[j + 1]] + distance[tour[k]][tour[k + 1]])
                            
                            # 개선이 있으면 업데이트
                            if imp < 0:
                                tour_indices.append(tour_idx)
                                position1.append(i)
                                position2.append(j)
                                position3.append(k)
                                best_imp += imp

    return tour_indices, position1, position2, position3, best_imp


# 3. Inter Relocation
def relocate_search(tour_list, distance, travel_time, due_time, tour_start_time, demand, capacity):
    best_imp = 0
    best_t1, best_t2 = -1, -1
    best_cust, best_position = -1, -1

    # 각 투어 쌍에 대해 리로케이션(재배치) 시도
    for t1, tour1 in enumerate(tour_list):
        for t2, tour2 in enumerate(tour_list):
            if t1 != t2:
            
                for i in range(1, len(tour1) - 1):
                    if demand[tour1[i]] + demand_sum(tour2, demand) <= capacity:
    
                        for j in range(len(tour2) - 1):
                            # 새로운 투어2 생성 (재배치 적용)
                            new_tour2 = tour2[:j + 1] + [tour1[i]] + tour2[j + 1:]
        
                            # 시간 제약 확인
                            if time_checker(new_tour2, travel_time, due_time, tour_start_time):
                                # 거리 계산
                                original_distance = (tour_distance(tour1, distance) + tour_distance(tour2, distance))
                                new_tour1 = tour1[:i] + tour1[i + 1:]
                                new_distance = (tour_distance(new_tour1, distance) + tour_distance(new_tour2, distance))
                                
                                # 개선이 있으면 업데이트
                                imp = new_distance - original_distance
                                if imp < best_imp:
                                    best_imp = imp
                                    best_t1, best_t2 = t1, t2
                                    best_cust, best_position = i, j

    return best_t1, best_t2, best_cust, best_position, best_imp


# 4. Inter Exchange
def exchange_search(tour_list, distance, travel_time, due_time, tour_start_time, demand, capacity):
    best_imp = 0
    best_t1, best_t2 = -1, -1
    best_position1, best_position2 = -1, -1

    # 각 투어 쌍에 대해 교환 시도
    for t1, tour1 in enumerate(tour_list[:-1]):
        for t2 in range(t1 + 1, len(tour_list)):
            tour2 = tour_list[t2]

            for i in range(1, len(tour1) - 1):
                for j in range(1, len(tour2) - 1):
                    # 수요 및 용량 제약 확인
                    tour1_new_demand = demand[tour2[j]] + demand_sum(tour1, demand) - demand[tour1[i]]
                    tour2_new_demand = demand[tour1[i]] + demand_sum(tour2, demand) - demand[tour2[j]]

                    if (tour1_new_demand <= capacity) and (tour2_new_demand <= capacity):
                        # 새로운 경로 생성 (교환 적용)
                        new_tour1 = tour1[:i] + [tour2[j]] + tour1[i + 1:]
                        new_tour2 = tour2[:j] + [tour1[i]] + tour2[j + 1:]

                        # 시간 제약 확인
                        if time_checker(new_tour1, travel_time, due_time, tour_start_time) and \
                           time_checker(new_tour2, travel_time, due_time, tour_start_time):
                            
                            # 교환 비용 계산
                            ex_cost1 = (distance[tour1[i - 1]][tour2[j]] + distance[tour2[j]][tour1[i + 1]]
                                        - distance[tour1[i - 1]][tour1[i]] - distance[tour1[i]][tour1[i + 1]])
                            ex_cost2 = (distance[tour2[j - 1]][tour1[i]] + distance[tour1[i]][tour2[j + 1]]
                                        - distance[tour2[j - 1]][tour2[j]] - distance[tour2[j]][tour2[j + 1]])

                            imp = ex_cost1 + ex_cost2
                            
                            # 개선이 있으면 업데이트
                            if imp < best_imp:
                                best_imp = imp
                                best_t1, best_t2 = t1, t2
                                best_position1, best_position2 = i, j

    return best_t1, best_t2, best_position1, best_position2, best_imp


# 5. CROSS
def CROSS_search(tour_list, distance, travel_time, due_time, tour_start_time, demand, capacity):
    best_imp = 0
    best_t1, best_t2 = -1, -1
    best_node11, best_node12 = -1, -1
    best_node21, best_node22 = -1, -1

    # 각 투어 쌍에 대해 CROSS 교환 시도
    for t1, tour1 in enumerate(tour_list[:-1]):
        for t2 in range(t1 + 1, len(tour_list)):
            tour2 = tour_list[t2]

            for i in range(1, len(tour1) - 2):
                for k in range(i + 1, len(tour1) - 1):
                    for j in range(1, len(tour2) - 2):
                        for l in range(j + 1, len(tour2) - 1):
                            # 수요 및 용량 제약 확인
                            tour1_new_demand = (demand_sum(tour2[j:l + 1], demand) +
                                                demand_sum(tour1, demand) - demand_sum(tour1[i:k + 1], demand))
                            tour2_new_demand = (demand_sum(tour1[i:k + 1], demand) +
                                                demand_sum(tour2, demand) - demand_sum(tour2[j:l + 1], demand))

                            if (tour1_new_demand <= capacity) and (tour2_new_demand <= capacity):
                                # 새로운 경로 생성 (CROSS 적용)
                                new_tour1 = tour1[:i] + tour2[j:l + 1] + tour1[k + 1:]
                                new_tour2 = tour2[:j] + tour1[i:k + 1] + tour2[l + 1:]

                                # 시간 제약 확인
                                if time_checker(new_tour1, travel_time, due_time, tour_start_time) and \
                                   time_checker(new_tour2, travel_time, due_time, tour_start_time):
                                    
                                    # 교환 비용 계산
                                    CROSS1 = (distance[tour1[i - 1]][tour2[j]] + distance[tour2[l]][tour1[k + 1]] +
                                              distance[tour2[j - 1]][tour1[i]] + distance[tour1[k]][tour2[l + 1]])
                                    CROSS2 = (distance[tour1[i - 1]][tour1[i]] + distance[tour1[k]][tour1[k + 1]] +
                                              distance[tour2[j - 1]][tour2[j]] + distance[tour2[l]][tour2[l + 1]])
                                    CROSS_cost = round(CROSS1 - CROSS2, 10)

                                    # 개선이 있으면 업데이트
                                    if CROSS_cost < best_imp:
                                        best_imp = CROSS_cost
                                        best_t1, best_t2 = t1, t2
                                        best_node11, best_node12 = i, k
                                        best_node21, best_node22 = j, l

    return best_t1, best_t2, best_node11, best_node12, best_node21, best_node22, best_imp


# 6. ICROSS
def ICROSS_search(tour_list, distance, travel_time, due_time, tour_start_time, demand, capacity):
    best_imp = 0
    best_t1, best_t2 = -1, -1
    best_node11, best_node12 = -1, -1
    best_node21, best_node22 = -1, -1

    for t1, tour1 in enumerate(tour_list[:-1]):
        for t2 in range(t1 + 1, len(tour_list)):
            tour2 = tour_list[t2]

            for i in range(1, len(tour1) - 2):
                for k in range(i + 1, len(tour1) - 1):
                    for j in range(1, len(tour2) - 2):
                        for l in range(j + 1, len(tour2) - 1):
                            # 수요 계산
                            tour1_new_demand = demand_sum(tour2[j:l + 1], demand) + demand_sum(tour1, demand) - demand_sum(tour1[i:k + 1], demand)
                            tour2_new_demand = demand_sum(tour1[i:k + 1], demand) + demand_sum(tour2, demand) - demand_sum(tour2[j:l + 1], demand)

                            if (tour1_new_demand <= capacity) and (tour2_new_demand <= capacity):
                                # 새로운 경로 생성 (역순 포함)
                                new_tour1 = tour1[:i] + tour2[j:l + 1][::-1] + tour1[k + 1:]
                                new_tour2 = tour2[:j] + tour1[i:k + 1][::-1] + tour2[l + 1:]

                                # 시간 제약 확인
                                if time_checker(new_tour1, travel_time, due_time, tour_start_time) and \
                                   time_checker(new_tour2, travel_time, due_time, tour_start_time):

                                    # ICROSS 비용 계산
                                    ICROSS1 = (distance[tour1[i - 1]][tour2[l]] + distance[tour2[j]][tour1[k + 1]] +
                                               distance[tour2[j - 1]][tour1[k]] + distance[tour1[i]][tour2[l + 1]])
                                    ICROSS2 = (distance[tour1[i - 1]][tour1[i]] + distance[tour1[k]][tour1[k + 1]] +
                                               distance[tour2[j - 1]][tour2[j]] + distance[tour2[l]][tour2[l + 1]])
                                    ICROSS3 = total_distance([tour1[i:k + 1][::-1]], distance) - total_distance([tour1[i:k + 1]], distance)
                                    ICROSS4 = total_distance([tour2[j:l + 1][::-1]], distance) - total_distance([tour2[j:l + 1]], distance)
                                    ICROSS_cost = ICROSS1 - ICROSS2 + ICROSS3 + ICROSS4

                                    # 개선이 있는 경우 업데이트
                                    if ICROSS_cost < best_imp:
                                        best_imp = ICROSS_cost
                                        best_t1, best_t2 = t1, t2
                                        best_node11, best_node12 = i, k
                                        best_node21, best_node22 = j, l

    return best_t1, best_t2, best_node11, best_node12, best_node21, best_node22, best_imp


# 7. GENI
def GENI_search(tour_list, distance, travel_time, due_time, tour_start_time, demand, capacity):
    best_imp = 0
    best_t1, best_t2 = -1, -1
    best_node1, best_node21, best_node22 = -1, -1, -1

    for t1, tour1 in enumerate(tour_list):
        for t2, tour2 in enumerate(tour_list):
            if t1 != t2 and len(tour2) >= 4:
                
                for i in range(1, len(tour1) - 1):
                    for j in range(0, len(tour2) - 3):
                        for k in range(j + 2, len(tour2) - 1):
                            # 수요 계산
                            tour2_new_demand = demand[tour1[i]] + demand_sum(tour2, demand)

                            if tour2_new_demand <= capacity:
                                # 새로운 경로 생성
                                new_tour1 = tour1[:i] + tour1[i + 1:]
                                new_tour2 = tour2[:j + 1] + [tour1[i]] + [tour2[k]] + tour2[j + 1:k] + tour2[k + 1:]

                                # 시간 제약 확인
                                if time_checker(new_tour1, travel_time, due_time, tour_start_time) and \
                                   time_checker(new_tour2, travel_time, due_time, tour_start_time):

                                    # GENI 비용 계산
                                    GENI1 = round(distance[tour1[i - 1]][tour1[i + 1]] -
                                                  distance[tour1[i - 1]][tour1[i]] -
                                                  distance[tour1[i]][tour1[i + 1]], 10)
                                    GENI2 = round(distance[tour2[j]][tour1[i]] + distance[tour1[i]][tour2[k]] +
                                                  distance[tour2[k]][tour2[j + 1]] +
                                                  distance[tour2[k - 1]][tour2[k + 1]] -
                                                  distance[tour2[j]][tour2[j + 1]] -
                                                  distance[tour2[k - 1]][tour2[k]] -
                                                  distance[tour2[k]][tour2[k + 1]], 10)
                                    GENI_cost = GENI1 + GENI2

                                    # 개선이 있는 경우 업데이트
                                    if GENI_cost < best_imp:
                                        best_imp = GENI_cost
                                        best_t1, best_t2 = t1, t2
                                        best_node1, best_node21, best_node22 = i, j, k

    return best_t1, best_t2, best_node1, best_node21, best_node22, best_imp


# 8. 2-opt*
def two_optstar_search(tour_list, distance, travel_time, due_time, tour_start_time, demand, capacity):
    best_imp = 0
    best_t1, best_t2 = -1, -1
    best_pos1, best_pos2 = -1, -1

    for t1, tour1 in enumerate(tour_list[:-1]):
        for t2, tour2 in enumerate(tour_list[t1 + 1:], start=t1 + 1):
            
            for i in range(len(tour1) - 1):
                for j in range(len(tour2) - 1):
                    # 새로운 경로 생성
                    new_tour1 = tour1[:i + 1] + tour2[j + 1:]
                    new_tour2 = tour2[:j + 1] + tour1[i + 1:]
                    
                    # 수요 계산
                    if (demand_sum(new_tour1, demand) <= capacity) and (demand_sum(new_tour2, demand) <= capacity):
                        # 시간 제약 확인
                        if time_checker(new_tour1, travel_time, due_time, tour_start_time) and \
                           time_checker(new_tour2, travel_time, due_time, tour_start_time):
                            
                            # 2-opt* 비용 계산
                            twoopts_cost = round(
                                distance[tour1[i]][tour2[j + 1]] + distance[tour2[j]][tour1[i + 1]] -
                                distance[tour1[i]][tour1[i + 1]] - distance[tour2[j]][tour2[j + 1]], 10
                            )

                            # 개선이 있는 경우 업데이트
                            if twoopts_cost < best_imp:
                                best_imp = twoopts_cost
                                best_t1, best_t2 = t1, t2
                                best_pos1, best_pos2 = i, j

    return best_t1, best_t2, best_pos1, best_pos2, best_imp

# 9. 개선방법 적용
def apply_improved_method(original_Sub_tour, result, improvement_idx):
    Sub_tour = copy.deepcopy(original_Sub_tour)
    if improvement_idx == 0:  # 2-opt
        Tour, Position1, Position2 = result[0], result[1], result[2]
        for t in range(len(Tour)):
            tour = Sub_tour[Tour[t]]
            New_tour = (
                tour[0:Position1[t] + 1]
                + tour[Position1[t] + 1:Position2[t] + 1][::-1]
                + tour[Position2[t] + 1:]
            )
            Sub_tour[Tour[t]] = New_tour

    elif improvement_idx in [1,2,3]:  # Or-opt
        Tour, Position1, Position2, Position3 = result[0], result[1], result[2], result[3]
        
        for t in range(len(Tour)):
            tour = Sub_tour[Tour[t]]
            if Position3[t] < Position1[t]:
                New_tour = (
                    tour[: Position3[t] + 1]
                    + tour[Position1[t] + 1 : Position2[t] + 1]
                    + tour[Position3[t] + 1 : Position1[t] + 1]
                    + tour[Position2[t] + 1 :]
                )

            else:
                New_tour = (
                    tour[:Position1[t] + 1]
                    + tour[Position2[t] + 1:Position3[t] + 1]
                    + tour[Position1[t] + 1:Position2[t] + 1]
                    + tour[Position3[t] + 1:]
                )

            Sub_tour[Tour[t]] = New_tour

    elif improvement_idx == 8:  # 2-opt*
        Tour1, Tour2, Position1, Position2 = result[0], result[1], result[2], result[3]
        New_tour1 = Sub_tour[Tour1][:Position1 + 1] + Sub_tour[Tour2][Position2 + 1:]
        New_tour2 = Sub_tour[Tour2][:Position2 + 1] + Sub_tour[Tour1][Position1 + 1:]
        Sub_tour[Tour1] = New_tour1
        Sub_tour[Tour2] = New_tour2

    elif improvement_idx == 5:  # Relocation
        Tour1, Tour2, Customer, Insert_Position = result[0], result[1], result[2], result[3]
        Sub_tour[Tour2].insert(Insert_Position + 1, Sub_tour[Tour1][Customer])
        del Sub_tour[Tour1][Customer]

    elif improvement_idx == 6:  # Exchange
        Tour1, Tour2, Position1, Position2 = result[0], result[1], result[2], result[3]
        New_tour1 = Sub_tour[Tour1][:Position1] + [Sub_tour[Tour2][Position2]] + Sub_tour[Tour1][Position1 + 1:]
        New_tour2 = Sub_tour[Tour2][:Position2] + [Sub_tour[Tour1][Position1]] + Sub_tour[Tour2][Position2 + 1:]
        Sub_tour[Tour1] = New_tour1
        Sub_tour[Tour2] = New_tour2

    elif improvement_idx == 7:  # CROSS
        Tour1, Tour2, Node11, Node12, Node21, Node22 = result[0], result[1], result[2], result[3], result[4], result[5]
        New_tour1 = Sub_tour[Tour1][:Node11] + Sub_tour[Tour2][Node21:Node22 + 1] + Sub_tour[Tour1][Node12 + 1:]
        New_tour2 = Sub_tour[Tour2][:Node21] + Sub_tour[Tour1][Node11:Node12 + 1] + Sub_tour[Tour2][Node22 + 1:]
        Sub_tour[Tour1] = New_tour1
        Sub_tour[Tour2] = New_tour2

    elif improvement_idx == 4:  # GENI
        Tour1, Tour2, Node1, Node21, Node22 = result[0], result[1], result[2], result[3], result[4]
        New_tour1 = Sub_tour[Tour1][:Node1] + Sub_tour[Tour1][Node1 + 1:]
        New_tour2 = (
            Sub_tour[Tour2][:Node21 + 1]
            + [Sub_tour[Tour1][Node1]]
            + [Sub_tour[Tour2][Node22]]
            + Sub_tour[Tour2][Node21 + 1:Node22]
            + Sub_tour[Tour2][Node22 + 1:]
        )
        Sub_tour[Tour1] = New_tour1
        Sub_tour[Tour2] = New_tour2

    elif improvement_idx == 9:  # ICROSS
        Tour1, Tour2, Node11, Node12, Node21, Node22 = result[0], result[1], result[2], result[3], result[4], result[5]
        New_tour1 = (
            Sub_tour[Tour1][:Node11]
            + Sub_tour[Tour2][Node21:Node22 + 1][::-1]
            + Sub_tour[Tour1][Node12 + 1:]
        )
        New_tour2 = (
            Sub_tour[Tour2][:Node21]
            + Sub_tour[Tour1][Node11:Node12 + 1][::-1]
            + Sub_tour[Tour2][Node22 + 1:]
        )
        Sub_tour[Tour1] = New_tour1
        Sub_tour[Tour2] = New_tour2

    return Sub_tour


# 10. VND 파라미터 설정
def set_vnd_parameters(vnd_type, vnd_no_imp_cnt):
    OPT_THRESHOLD = 8
    
    if vnd_type == 'small':
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 8, 20, False
    elif vnd_type == 'big':
        if vnd_no_imp_cnt < OPT_THRESHOLD:
            return [0, 1, 2, 3], 4, 30, False
        return [4, 5, 6, 7], 2, 30, True


# 11. 단일 VND 과정
def single_vnd_process(Sub_tour, distance, travel_time, due_time, tour_start_time, is_stage_2):
    method_len = 10  # Total number of improvement methods
    start_improvement_idx = randint(0, method_len)

    for i in range(method_len):
        improvement_idx = (start_improvement_idx + i * 3) % method_len
        if is_stage_2:
            improvement_idx += 4

        methods = {
            0: lambda st, d, tt, dt, ts: two_opt_search(st, d, tt, dt, ts),
            1: lambda st, d, tt, dt, ts: or_opt_search(st, d, tt, dt, 1, ts),
            2: lambda st, d, tt, dt, ts: or_opt_search(st, d, tt, dt, 2, ts),
            3: lambda st, d, tt, dt, ts: or_opt_search(st, d, tt, dt, 3, ts),
            4: lambda st, d, tt, dt, ts: GENI_search(st, d, tt, dt, ts, demand, capacity),
            5: lambda st, d, tt, dt, ts: relocate_search(st, d, tt, dt, ts, demand, capacity),
            6: lambda st, d, tt, dt, ts: exchange_search(st, d, tt, dt, ts, demand, capacity),
            7: lambda st, d, tt, dt, ts: CROSS_search(st, d, tt, dt, ts, demand, capacity),
            8: lambda st, d, tt, dt, ts: two_optstar_search(st, d, tt, dt, ts, demand, capacity),
            9: lambda st, d, tt, dt, ts: ICROSS_search(st, d, tt, dt, ts, demand, capacity)
        }
    
        return methods[improvement_idx](Sub_tour, distance, travel_time, due_time, tour_start_time)

        if result:
            return result, improvement_idx

    return None, None


# 12. VND 메인코드
def VND(Sub_tour, distance, travel_time, due_time, tour_start_time, demand, capacity, vnd_no_imp_cnt, vnd_type):
    
    start_time = time.time()
    tabu_list = []
    Improved = True
    iteration = 0

    # 1. VND 유형에 따라 매개변수 설정
    Neighbor_order, ITERATION_THRESHOLD, VND_TIME_LIMIT, is_stage_2 = set_vnd_parameters(vnd_type, vnd_no_imp_cnt)
    method_len = len(Neighbor_order)

    # 2. 개선 진행
    while Improved and iteration < ITERATION_THRESHOLD and time.time() - start_time < VND_TIME_LIMIT:
        # 2-1. VND
        start_improvement_idx = randint(0, method_len)
        result, improvement, improvement_idx = None, None, 0
        for i in range(method_len):
            improvement_idx = (start_improvement_idx+i*3)%method_len
            if is_stage_2:
                improvement_idx += 4
            
            if improvement_idx == 0:
                result = two_opt_search(Sub_tour, distance, travel_time, due_time, tour_start_time)
            elif improvement_idx == 1:
                result = or_opt_search(Sub_tour, distance, travel_time, due_time, 1, tour_start_time)
            elif improvement_idx == 2:
                result = or_opt_search(Sub_tour, distance, travel_time, due_time, 2, tour_start_time)
            elif improvement_idx == 3:
                result = or_opt_search(Sub_tour, distance, travel_time, due_time, 3, tour_start_time)
            elif improvement_idx == 8:  # 2-opt*
                result = two_optstar_search(Sub_tour, distance, travel_time, due_time, tour_start_time, demand, capacity)
            elif improvement_idx == 5:  # Relocation
                result = relocate_search(Sub_tour, distance, travel_time, due_time, tour_start_time, demand, capacity)
            elif improvement_idx == 6:  # Exchange
                result = exchange_search(Sub_tour, distance, travel_time, due_time, tour_start_time, demand, capacity)
            elif improvement_idx == 7:  # CROSS
                result = CROSS_search(Sub_tour, distance, travel_time, due_time, tour_start_time, demand, capacity)
            elif improvement_idx == 9:  # ICROSS
                result = ICROSS_search(Sub_tour, distance, travel_time, due_time, tour_start_time, demand, capacity)
            elif improvement_idx == 4:  # GENI
                result = GENI_search(Sub_tour, distance, travel_time, due_time, tour_start_time, demand, capacity)

            if round(result[-1], 5) < -2:
                break
            if i >= method_len-1:
                Improved = False

        # 2-2. move
        if Improved:
            Sub_tour_improved = apply_improved_method(Sub_tour, result, improvement_idx)            
            if Sub_tour_improved not in tabu_list:
                tabu_list.append(Sub_tour_improved)
                Sub_tour = Sub_tour_improved
                # print(improvement_idx)
            else:
                Improved = False

        iteration += 1

    return Sub_tour, improvement_idx


## 3. VNS
# 1. K-means 클러스터링으로 초기해 계산
def initial_solution_by_kmeans(coords, order_n_list, start_hub_i):
    
    n_clusters = len(order_n_list) // 5
    order_coords = [coords[i] for i in order_n_list]
    
    # 1. 클러스터별 주문 할당
    kmeans = KMeans(n_clusters=n_clusters).fit(order_coords)
    
    clustered_orders = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(kmeans.labels_):
        clustered_orders[label].append(order_n_list[idx])

    # 2. 각 클러스터에 출발 허브 추가
    initial_solution = [[start_hub_i] + cluster + [start_hub_i] for cluster in clustered_orders]
    
    # 3. 빈 경로 3개 추가
    initial_solution.extend([[start_hub_i, start_hub_i]] * 3)

    return initial_solution


# 2. VNS 파라미터 설정
def set_vns_parameters(order_count):
    if order_count < 40:
        return 'small', 20
    elif order_count > 80:
        return 'big', 150
    return 'big', 30


# 3. 개수가 적은 경우 순열로 탐색
def handle_small_instances(order_n_list, start_hub_i, distance_matrix):
    min_distance = float('inf')
    final_solution = []

    for tour in permutations(order_n_list):
        tour = [start_hub_i] + list(tour)
        total_dist = tour_distance(tour, distance_matrix)

        if total_dist < min_distance:
            min_distance = total_dist
            final_solution = tour

    # print(f"cost : {min_distance}, permutation")
    return [final_solution]


# 4. 섞기 함수 - sub
def shaking(Input_tour, travel_time, duetime, demand, Neighbor_Str, tour_start_time):
    capacity = 180
    
    n = len(Input_tour) - 1
    Sub_tour = [tour[:] for tour in Input_tour]
    # copy.deepcopy(Input_tour)
    shaking_start = time.time()

    if Neighbor_Str == 0:  # 2-opt
        while True:
          tour_list = [i for i in range(0, n) if len(Sub_tour[i]) >= 5]
          if len(tour_list) == 0:
            break
          Tour = choice(tour_list)

          Position1 = randint(0, len(Sub_tour[Tour]) - 4)
          Position2 = randint(Position1 + 2, len(Sub_tour[Tour]) - 2)

          New_tour = Sub_tour[Tour][0:Position1 + 1] + Sub_tour[Tour][Position1 + 1:Position2 + 1][::-1] + Sub_tour[
                                                                                                                Tour][
                                                                                                            Position2 + 1:]
          time_check = time_checker(New_tour, travel_time, duetime, tour_start_time)

          shaking_end = time.time()

          if time_check:
              Sub_tour[Tour] = New_tour
              break
          elif shaking_end - shaking_start > 0.5:
              break

    elif Neighbor_Str == 1 or Neighbor_Str == 2 or Neighbor_Str == 3:  # Or-opt
        K = Neighbor_Str
        while True:
            tour_list = [i for i in range(0, n) if len(Sub_tour[i]) >= K + 3]
            if len(tour_list) == 0:
              break
            Tour = choice(tour_list)

            Position1 = randint(0, len(Sub_tour[Tour]) - K - 2)
            Position2 = Position1 + K
            Position3 = randint(0, len(Sub_tour[Tour]) - 2)

            for Position3 in range(0, len(Sub_tour[Tour]) - 2):
              if Position1 > Position3 or Position3 > Position2:
                break

            if Position3 < Position1:
                New_tour = Sub_tour[Tour][:Position3 + 1] + Sub_tour[Tour][Position1 + 1:Position2 + 1] + Sub_tour[
                                                                                                              Tour][
                                                                                                          Position3 + 1:Position1 + 1] + \
                           Sub_tour[Tour][Position2 + 1:]
            else:
                New_tour = Sub_tour[Tour][:Position1 + 1] + Sub_tour[Tour][Position2 + 1:Position3 + 1] + Sub_tour[
                                                                                                              Tour][
                                                                                                          Position1 + 1:Position2 + 1] + \
                           Sub_tour[Tour][Position3 + 1:]

            time_check = time_checker(New_tour, travel_time, duetime, tour_start_time)

            shaking_end = time.time()

            if time_check:
                Sub_tour[Tour] = New_tour
                break
            elif shaking_end - shaking_start > 0.5:
                break

    elif Neighbor_Str == 4 and n > 0:  # 2-optstar
        while True:
            tour_list = [i for i in range(0, n) if len(Sub_tour[i]) > 2]
            if len(tour_list) < 2:
              break
            tour_a, tour_b = sample(tour_list, 2)
            
            Tour1 = min(tour_a, tour_b)
            Tour2 = max(tour_a, tour_b)
            Position1 = randint(1, len(Sub_tour[Tour1]) - 2)
            Position2 = randint(1, len(Sub_tour[Tour2]) - 2)

            New_tour1 = Sub_tour[Tour1][:Position1 + 1] + Sub_tour[Tour2][Position2 + 1:]
            New_tour2 = Sub_tour[Tour2][:Position2 + 1] + Sub_tour[Tour1][Position1 + 1:]

            time_check1 = time_checker(New_tour1, travel_time, duetime, tour_start_time)
            time_check2 = time_checker(New_tour2, travel_time, duetime, tour_start_time)
            new_tour1_demand = demand_sum(New_tour1, demand)
            new_tour2_demand = demand_sum(New_tour2, demand)

            shaking_end = time.time()

            if time_check1 and time_check2 and new_tour1_demand <= capacity and new_tour2_demand <= capacity:
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2
                break
            elif shaking_end - shaking_start > 0.5:
                break


    elif Neighbor_Str == 5 and n > 0:  # Relocation

        while True:
            tour_list = [i for i in range(0, n) if len(Sub_tour[i]) > 2]
            if len(tour_list) < 2:
              break
                
            Tour1 = choice(tour_list)
            tour_list.remove(Tour1)
            Tour2 = choice(tour_list)
            
            Customer = randint(1, len(Sub_tour[Tour1]) - 2)
            Insert_Position = randint(0, len(Sub_tour[Tour2]) - 2)

            newtour = Sub_tour[Tour2][:Insert_Position + 1] + [Sub_tour[Tour1][Customer]] + Sub_tour[Tour2][
                                                                                            Insert_Position + 1:]
            time_check = time_checker(newtour, travel_time, duetime, tour_start_time)
            tour2_demand = demand_sum(Sub_tour[Tour1][Customer], demand) + sum([demand_sum(c, demand) for c in Sub_tour[Tour2]])

            shaking_end = time.time()

            if time_check and tour2_demand <= capacity:
                Sub_tour[Tour2].insert(Insert_Position + 1, Sub_tour[Tour1][Customer])
                del Sub_tour[Tour1][Customer]
                break
            elif shaking_end - shaking_start > 0.5:
                break

    elif Neighbor_Str == 6 and n > 0:  # Exchange

        while True:
            tour_list = [i for i in range(0, n) if len(Sub_tour[i]) > 2]
            if len(tour_list) < 2:
              break
            tour_a, tour_b = sample(tour_list, 2)
            
            Tour1 = min(tour_a, tour_b)
            Tour2 = max(tour_a, tour_b)
            
            Position1 = randint(1, len(Sub_tour[Tour1]) - 2)
            Position2 = randint(1, len(Sub_tour[Tour2]) - 2)

            New_tour1 = Sub_tour[Tour1][:Position1] + [Sub_tour[Tour2][Position2]] + Sub_tour[Tour1][Position1 + 1:]
            New_tour2 = Sub_tour[Tour2][:Position2] + [Sub_tour[Tour1][Position1]] + Sub_tour[Tour2][Position2 + 1:]

            time_check1 = time_checker(New_tour1, travel_time, duetime, tour_start_time)
            time_check2 = time_checker(New_tour2, travel_time, duetime, tour_start_time)
            new_tour1_demand = demand_sum(New_tour1, demand)
            new_tour2_demand = demand_sum(New_tour2, demand)

            shaking_end = time.time()

            if time_check1 and time_check2 and new_tour1_demand <= capacity and new_tour2_demand <= capacity:
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2
                break
            elif shaking_end - shaking_start > 0.5:
                break

    elif Neighbor_Str == 7 and n > 0:  # CROSS

        while True:
            tour_list = [i for i in range(0, n) if len(Sub_tour[i])>=4]
            if len(tour_list) < 2:
              break
            Tour1 = choice(tour_list)
            tour_list.remove(Tour1)
            Tour2 = choice(tour_list)

            Node11 = randint(1, len(Sub_tour[Tour1]) - 3)
            Node12 = randint(Node11, len(Sub_tour[Tour1]) - 2)
            Node21 = randint(1, len(Sub_tour[Tour2]) - 3)
            Node22 = randint(Node21, len(Sub_tour[Tour2]) - 2)

            New_tour1 = Sub_tour[Tour1][:Node11] + Sub_tour[Tour2][Node21:Node22 + 1] + Sub_tour[Tour1][Node12 + 1:]
            New_tour2 = Sub_tour[Tour2][:Node21] + Sub_tour[Tour1][Node11:Node12 + 1] + Sub_tour[Tour2][Node22 + 1:]

            time_check1 = time_checker(New_tour1, travel_time, duetime, tour_start_time)
            time_check2 = time_checker(New_tour2, travel_time, duetime, tour_start_time)
            new_tour1_demand = demand_sum(New_tour1, demand)
            new_tour2_demand = demand_sum(New_tour2, demand)

            shaking_end = time.time()

            if time_check1 and time_check2 and new_tour1_demand <= capacity and new_tour2_demand <= capacity:
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2
                break
            elif shaking_end - shaking_start > 0.5:
                break

    elif Neighbor_Str == 8 and n > 0:  # ICROSS

        while True:
            tour_list = [i for i in range(0, n) if len(Sub_tour[i])>=4]
            if len(tour_list) < 2:
              break
            Tour1 = choice(tour_list)
            tour_list.remove(Tour1)
            Tour2 = choice(tour_list)

            Node11 = randint(1, len(Sub_tour[Tour1]) - 3)
            Node12 = randint(Node11, len(Sub_tour[Tour1]) - 2)
            Node21 = randint(1, len(Sub_tour[Tour2]) - 3)
            Node22 = randint(Node21, len(Sub_tour[Tour2]) - 2)

            New_tour1 = Sub_tour[Tour1][:Node11] + Sub_tour[Tour2][Node21:Node22 + 1][::-1] + Sub_tour[Tour1][
                                                                                              Node12 + 1:]
            New_tour2 = Sub_tour[Tour2][:Node21] + Sub_tour[Tour1][Node11:Node12 + 1][::-1] + Sub_tour[Tour2][
                                                                                              Node22 + 1:]

            time_check1 = time_checker(New_tour1, travel_time, duetime, tour_start_time)
            time_check2 = time_checker(New_tour2, travel_time, duetime, tour_start_time)
            new_tour1_demand = demand_sum(New_tour1, demand)
            new_tour2_demand = demand_sum(New_tour2, demand)

            shaking_end = time.time()

            if time_check1 and time_check2 and new_tour1_demand <= capacity and new_tour2_demand <= capacity:
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2
                break
            elif shaking_end - shaking_start > 0.5:
                break


    elif Neighbor_Str == 9 and n > 0:  # GENI

        while True:
            
            tour_list_1 = [i for i in range(0, n) if len(Sub_tour[i])>2]
            if len(tour_list_1) == 0:
                break
            Tour1 = choice(tour_list_1)
            
            tour_list = [i for i in range(0, n) if len(Sub_tour[i])>=4 and i != Tour1]
            if len(tour_list) == 0:
              break
            Tour2 = choice(tour_list)

            Node1 = randint(1, len(Sub_tour[Tour1]) - 2)
            Node21 = randint(0, len(Sub_tour[Tour2]) - 4)
            Node22 = randint(Node21 + 2, len(Sub_tour[Tour2]) - 2)

            New_tour1 = Sub_tour[Tour1][:Node1] + Sub_tour[Tour1][Node1 + 1:]
            New_tour2 = Sub_tour[Tour2][:Node21 + 1] + [Sub_tour[Tour1][Node1]] + [Sub_tour[Tour2][Node22]] + Sub_tour[
                                                                                                                  Tour2][
                                                                                                              Node21 + 1:Node22] + \
                        Sub_tour[Tour2][Node22 + 1:]
            
            time_check1 = time_checker(New_tour1, travel_time, duetime, tour_start_time)
            time_check2 = time_checker(New_tour2, travel_time, duetime, tour_start_time)
            new_tour1_demand = demand_sum(New_tour1, demand)
            new_tour2_demand = demand_sum(New_tour2, demand)

            shaking_end = time.time()

            if time_check1 and time_check2 and new_tour1_demand <= capacity and new_tour2_demand <= capacity:
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2
                break
            elif shaking_end - shaking_start > 0.5:
                break

    elif Neighbor_Str == 10 and n > 0:  #λ-interchange

        while True:
            tour_list = [i for i in range(0, n) if len(Sub_tour[i])>=4]
            if len(tour_list) < 2:
              break
            Tour1 = choice(tour_list)
            tour_list.remove(Tour1)
            Tour2 = choice(tour_list)

            Node11 = randint(1, len(Sub_tour[Tour1]) - 3)
            Node12 = randint(Node11, len(Sub_tour[Tour1]) - 2)
            Node21 = randint(1, len(Sub_tour[Tour2]) - 3)
            Node22 = randint(Node21, len(Sub_tour[Tour2]) - 2)

            New_tour1 = Sub_tour[Tour1][:Node11] + Sub_tour[Tour2][Node21:Node22 + 1] + Sub_tour[Tour1][Node12 + 1:]
            New_tour2 = Sub_tour[Tour2][:Node21] + Sub_tour[Tour1][Node11:Node12 + 1] + Sub_tour[Tour2][Node22 + 1:]

            time_check1 = time_checker(New_tour1, travel_time, duetime, tour_start_time)
            time_check2 = time_checker(New_tour2, travel_time, duetime, tour_start_time)
            new_tour1_demand = demand_sum(New_tour1, demand)
            new_tour2_demand = demand_sum(New_tour2, demand)

            shaking_end = time.time()

            if time_check1 and time_check2 and new_tour1_demand <= capacity and new_tour2_demand <= capacity:
                Sub_tour[Tour1] = New_tour1
                Sub_tour[Tour2] = New_tour2
                break
            elif shaking_end - shaking_start > 0.5:
                break

    return Sub_tour
    

# 5. 섞기 함수 - main
def shake_solution(x, shaking_method_idx, duration_matrix, due_time, demand, current_time):
    Shaking_method = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    x_prime = x
    while x_prime == x:
        x_prime = shaking(x_prime, duration_matrix, due_time, demand, Shaking_method[shaking_method_idx],
                          current_time)
        shaking_method_idx = (shaking_method_idx + 1) % len(Shaking_method)
    return x_prime


# 6. VNS 메인코드
def VNS(df, time_block, blitz_hub_str, distance_matrix, duration_matrix, due_time, demand, capacity, coords):
    # 1. 주문 필터링
    order_list = df[(df.time_block == time_block) & (df.blitz_hub == blitz_hub_str)]
    if order_list.empty:
        return []

    # print(f'\n{time_block} - {blitz_hub_str}')
    
    tour_start_time = create_timestamp_from_time_block(time_block)
    start_hub_i = blitz_hub_to_i(blitz_hub_str)
    order_n_list = order_list.index.tolist()
    vnd_type, TIME_LIMIT = set_vns_parameters(order_list.shape[0])

    # 2. 주문 개수 적은 경우 순열로 직접 탐색
    if len(order_n_list) <= 5:
        return handle_small_instances(order_n_list, start_hub_i, distance_matrix)

    # 3. 초기해 설정
    initial_solution = initial_solution_by_kmeans(coords, order_n_list, start_hub_i)
    # print(f"cost : {total_distance(initial_solution, distance_matrix)} (initial)")

    # 4. VNS 프로세스
    VNS_IMP_THRESHOLD = 3
    VND_IMP_THRESHOLD = 9

    improvement_map = {
        0: '2-opt',
        1: 'or-opt',
        2: 'or-opt',
        3: 'or-opt',
        4: 'GENI',
        5: 'Relocation',
        6: 'Exchange',
        7: 'CROSS',
        8: '2-opt*',
        9: 'ICROSS'
    }
    
    x = initial_solution
    shaking_method_idx = 0
    vns_no_imp_cnt = 0
    start_time = time.time()

    while vns_no_imp_cnt < VNS_IMP_THRESHOLD and time.time() - start_time < TIME_LIMIT:
        x_prime = shake_solution(x, shaking_method_idx, duration_matrix, due_time, demand, tour_start_time)
        best_x_double_prime = None
        vnd_no_imp_cnt = 0
    
        while vnd_no_imp_cnt < VND_IMP_THRESHOLD and time.time() - start_time < TIME_LIMIT:
            x_double_prime, improvement_idx = VND(x_prime, distance_matrix, duration_matrix, due_time, tour_start_time, demand, capacity, 
                                                  vnd_no_imp_cnt, vnd_type)
            if total_distance(x_double_prime, distance_matrix) < total_distance(x_prime, distance_matrix):
                best_x_double_prime = x_double_prime
            vnd_no_imp_cnt += 1

        if best_x_double_prime and total_distance(best_x_double_prime, distance_matrix) < total_distance(x, distance_matrix):
            x = best_x_double_prime
            improvement_algo = improvement_map.get(improvement_idx, '')
            # print(f"cost : {total_distance(x, distance_matrix)}, {improvement_algo}\t{time.time() - start_time:>15.2f}s")
            vns_no_imp_cnt = 0  # Reset counter on improvement
        else:
            vns_no_imp_cnt += 1  # Increment counter if no improvement

    # print(f"duration : {time.time() - start_time:.2f} seconds.")

    return [tour for tour in x if len(tour)>2]
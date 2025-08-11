import os
import pickle
import json

from datetime import datetime, timedelta
from math import sqrt

import requests

from batch_classes import LocationType, Order

class RouteOptimizer:
    MAX_CAPACITY = 30  # 최대 용량 (kg)
    MAX_DISTANCE = 120  # 최대 거리 (km)
    DELAY_MINUTE_PER_TASK = 5  # 매 pickup/dropoff마다 소요되는 지연 시간 (분)
    FIRST_ORDER_PICKUP_DURATION = 10  # 첫 주문이 created된 이후 pickup까지 걸리는 시간 (분)
    
    def __init__(
        self, sheet_name, all_orders, location_list, matrix, 
        order_pool=None, max_capacity=None, max_distance=None, delay_minute_per_task=None, first_order_pickup_duration=None
    ):
        self.batch_locations = location_list  # 초기 location 리스트
        self.order_pool = set(all_orders)  # 전체 주문 집합
        if order_pool:
            self.order_pool = order_pool
        self.matrix = matrix
        
        self.sheet_name = sheet_name
        if sheet_name == 'SameDay Orders(8 Hours SLA)':
            self.distance_threshold = 5
            self.problem_type = 1
        elif sheet_name == 'Instant Orders(1-4 Hours SLA)':
            self.distance_threshold = 10
            self.problem_type = 2
        elif sheet_name == 'Mixed Orders(SameDay+Instant)':
            self.distance_threshold = 5
            self.problem_type = 3
        
        self.failed_attempts = set()  # 삽입 시도 실패한 주문 ID 집합
        self.added_order_id = None  # 추가할 주문 ID
        self.current_total_distance = float('inf')
        self.order_dict = {order.order_id: order for order in all_orders}

        self.query_count = 0
        
        if max_capacity:
            self.MAX_CAPACITY = max_capacity
        if max_distance:
            self.MAX_DISTANCE = max_distance
        if delay_minute_per_task:
            self.DELAY_MINUTE_PER_TASK = delay_minute_per_task
        if first_order_pickup_duration:
            self.FIRST_ORDER_PICKUP_DURATION = first_order_pickup_duration

    def optimize_route(self):
        # 초기해 생성 (time window대로 정렬)
        self.batch_locations.sort()
        succeed = self.satisfy_constraints()

        succeed = self.calculate_cost_and_check_constraints_in_optimization(succeed)

        if not succeed:
            self.failed_attempts.add(self.added_order_id)
        
        return succeed

    def optimize_existing_route(self, new_order: Order):
        succeed = self.satisfy_constraints()
        new_order_created_time = new_order.time_window[0]
        
        succeed = self.calculate_cost_and_check_constraints_in_optimization(succeed)

        if not succeed:
            self.failed_attempts.add(self.added_order_id)
        
        return succeed

    def calculate_cost_and_check_constraints_in_optimization(
        self, succeed,
    ):
        for target_i in range(1, len(self.batch_locations)):
            target = self.batch_locations[target_i]

            start_i, end_i = self.find_start_end_i(target)
            
            for insert_i in range(start_i, end_i+1):
                if insert_i != target_i and insert_i != target_i + 1:
        
                    original_state = self.batch_locations[:]
                    
                    delete_cost = self.calculate_delete_cost(target_i)
                    insert_cost = self.calculate_insert_cost(insert_i, target)
        
                    if insert_cost + delete_cost < 0:
                        self.move_location(target, target_i, insert_i)
                        
                        if (
                            self.satisfy_constraints()
                            and self.check_pickup_dropoff_order()
                            and self.current_total_distance + self.distance_threshold > self.calculate_total_distance()
                        ):  
                            succeed = True
                            self.update_arrival_time()
                            
                            # 처음부터 다시 최적화
                            target_i = 1
                            insert_i = 0
                            break
                            
                        else:  # 실패
                            self.batch_locations = original_state

        return succeed

    def add_order_to_existing_batch(self, new_order: Order):
        self.batch_locations.extend([new_order.pickup, new_order.dropoff])
        self.order_dict[new_order.order_id] = new_order
    
    def find_start_end_i(self, target):
        if target.location_type == LocationType.PICKUP:
            start_i = 0
            end_i = 0
            for i in range(len(self.batch_locations)):
                if (
                    self.batch_locations[i].location_type == LocationType.DROPOFF 
                    and self.batch_locations[i].order_id == target.order_id
                ):
                    end_i = i
                    break
        elif target.location_type == LocationType.DROPOFF:
            start_i = 0
            for i in range(len(self.batch_locations)):
                if (
                    self.batch_locations[i].location_type == LocationType.PICKUP 
                    and self.batch_locations[i].order_id == target.order_id
                ):
                    start_i = i
                    break
            start_i += 1
            end_i = len(self.batch_locations)

        return start_i, end_i
    
    def move_location(self, target, target_i, insert_i):
        """
        리스트 내 특정 요소를 제거하고 새로운 위치에 삽입하는 함수.
        
        :param target: 이동할 요소 (Location 객체)
        :param target_i: 현재 위치의 인덱스
        :param insert_i: 삽입할 위치의 인덱스
        """
        
        # 1. 현재 위치에서 요소 제거 (target_i를 직접 활용)
        if 0 <= target_i < len(self.batch_locations) and self.batch_locations[target_i] == target:
            self.batch_locations.pop(target_i)
        else:
            raise ValueError("target_i와 target이 일치하지 않거나 범위를 벗어났습니다.")
    
        # 2. 새로운 위치에 요소 삽입 (insert_i 범위 확인)
        if target_i < insert_i:
            insert_i -= 1
        if 0 <= insert_i <= len(self.batch_locations):
            self.batch_locations.insert(insert_i, target)
        else:
            # insert_i 범위 오류 처리
            raise ValueError("삽입할 위치 insert_i가 유효하지 않습니다.")

    def satisfy_capacity_constraint(self):        
        current_capacity = 0
        for location in self.batch_locations:
            try:
                order = self.order_dict[location.order_id]
            except KeyError:
                print(f"{location.order_id} not in order_dict")
                order = self.order_dict[location.order_id]
            order_capacity = order.weight

            if location.location_type == LocationType.PICKUP:
                current_capacity += order_capacity
            elif location.location_type == LocationType.DROPOFF:
                current_capacity -= order_capacity

            if current_capacity > self.MAX_CAPACITY:
                return False

        return True
            
    
    def satisfy_constraints(self):
        # 1. 최대 거리
        total_distance = self.calculate_total_distance()
        if total_distance > self.MAX_DISTANCE:
            return False
        
        # 2. 최대 용량
        if not self.satisfy_capacity_constraint():
            return False
            
        # 3. time window
        current_time = self.batch_locations[0].time_window[0] + timedelta(minutes=self.FIRST_ORDER_PICKUP_DURATION)
        
        for i, loc in enumerate(self.batch_locations[1:]):
            
            current_i = i + 1
            distance, duration, duration_td = self.calculate_distance_and_duration(self.batch_locations[current_i-1], self.batch_locations[current_i], current_time)
            ori = duration_td
            duration_td += timedelta(minutes=self.DELAY_MINUTE_PER_TASK)
            current_time += duration_td
            
            if current_time < loc.time_window[0] or current_time > loc.time_window[1]:
                return False

        # 4. pickup -> dropoff 순서
        location_index = {}
        for i, loc in enumerate(self.batch_locations):
            order_id = loc.order_id
            if order_id not in location_index:
                location_index[order_id] = [-1, -1]

            if loc.location_type == LocationType.PICKUP:
                location_index[order_id][0] = i
            elif loc.location_type == LocationType.DROPOFF:
                location_index[order_id][1] = i

        for order_id, (pickup_index, dropoff_index) in location_index.items():
            if pickup_index > dropoff_index:
                return False

        return True

    def check_pickup_dropoff_order(self):
        # pickup -> dropoff 순서
        location_index = {}
        for i, loc in enumerate(self.batch_locations):
            order_id = loc.order_id
            if order_id not in location_index:
                location_index[order_id] = [-1, -1]

            if loc.location_type == LocationType.PICKUP:
                location_index[order_id][0] = i
            elif loc.location_type == LocationType.DROPOFF:
                location_index[order_id][1] = i

        for order_id, (pickup_index, dropoff_index) in location_index.items():
            if pickup_index > dropoff_index:
                return False

        return True

    def enforce_pickup_dropoff(self):
        # pickup -> dropoff 순서 지키도록 swap
        location_index = {}
        for i, loc in enumerate(self.batch_locations):
            order_id = loc.order_id
            if order_id not in location_index:
                location_index[order_id] = [-1, -1]

            if loc.location_type == LocationType.PICKUP:
                location_index[order_id][0] = i
            elif loc.location_type == LocationType.DROPOFF:
                location_index[order_id][1] = i

        for order_id, index_list in location_index.items():
            pickup_index, dropoff_index = index_list

            # 만약 dropoff가 pickup보다 먼저 나오면 위치를 교환
            if pickup_index > dropoff_index:
                # 위치 교환
                self.batch_locations[pickup_index], self.batch_locations[dropoff_index] = (
                    self.batch_locations[dropoff_index],
                    self.batch_locations[pickup_index],
                )
    
    def calculate_total_distance(self):
        current_time = self.batch_locations[0].time_window[0] + timedelta(minutes=self.FIRST_ORDER_PICKUP_DURATION)
        total_distance = 0
        for i in range(0, len(self.batch_locations)-1):
            distance, duration, duration_td = self.calculate_distance_and_duration(self.batch_locations[i], self.batch_locations[i+1], current_time)
            total_distance += distance
            duration_td += timedelta(minutes=self.DELAY_MINUTE_PER_TASK)
            current_time += duration_td
        return total_distance

    def calculate_batch_total_distance(self, batch):
        batch_locations = batch.route
        
        current_time = batch_locations[0].time_window[0] + timedelta(minutes=self.FIRST_ORDER_PICKUP_DURATION)
        total_distance = 0
        for i in range(0, len(batch_locations)-1):
            distance, duration, duration_td = self.calculate_distance_and_duration(batch_locations[i], batch_locations[i+1], current_time)
            total_distance += distance
            duration_td += timedelta(minutes=self.DELAY_MINUTE_PER_TASK)
            current_time += duration_td
            
        return total_distance

    def calculate_total_weight(self):
        return sum(loc.weight for loc in self.batch_locations)

    def calculate_total_duration(self):
        current_time = self.batch_locations[0].time_window[0] + timedelta(minutes=self.FIRST_ORDER_PICKUP_DURATION)
        start_time = current_time
        total_distance = 0
        for i in range(0, len(self.batch_locations)-1):
            distance, duration, duration_td = self.calculate_distance_and_duration(self.batch_locations[i], self.batch_locations[i+1], current_time)
            duration_td += timedelta(minutes=self.DELAY_MINUTE_PER_TASK)
            current_time += duration_td
        return current_time - start_time
    
    def i_to_location(self, i):
        if i < 0 or i >= len(self.batch_locations):
            return None
        else:
            return self.batch_locations[i]

    def calculate_delete_cost(self, target_i):
        # assume excluding B: A -> B -> C to A -> C
        
        distance_a_to_b = self.calculate_distance_using_i(target_i-1, target_i)
        distance_b_to_c = self.calculate_distance_using_i(target_i, target_i+1)
        distance_a_to_c = self.calculate_distance_using_i(target_i-1, target_i+1)

        return distance_a_to_c - (distance_a_to_b + distance_b_to_c)

    def calculate_insert_cost(self, insert_i, target):
        # assume inserting C: A -> B to A -> C -> B

        A_loc = self.i_to_location(insert_i-1)
        B_loc = self.i_to_location(insert_i)

        if not A_loc:  # 맨 앞에 추가하는 경우
            distance_c_to_b = self.calculate_distance(target, B_loc)
            return distance_c_to_b

        if not B_loc:  # 맨 뒤에 추가하는 경우
            distance_a_to_c = self.calculate_distance(A_loc, target)
            return distance_a_to_c

        else:
            distance_a_to_b = self.calculate_distance_using_i(insert_i-1, insert_i)
            distance_a_to_c = self.calculate_distance(A_loc, target)
            distance_c_to_b = self.calculate_distance(target, B_loc)
            return distance_a_to_c + distance_c_to_b - distance_a_to_b

    def update_arrival_time(self):
        current_time = self.batch_locations[0].time_window[0] + timedelta(minutes=self.FIRST_ORDER_PICKUP_DURATION)
        self.batch_locations[0].arrival_time = current_time
        for i, loc in enumerate(self.batch_locations[1:]):
            distance, duration, duration_td = self.calculate_distance_and_duration(self.batch_locations[i], self.batch_locations[i+1], current_time)
            duration_td += timedelta(minutes=self.DELAY_MINUTE_PER_TASK)
            current_time += duration_td
            loc.arrival_time = current_time
        # self.batch_locations[-1].arrival_time = current_time

    def choose_order_to_add(self):
        """
        거리 정보가 가장 짧은 후보 주문을 선택하여 추가.
        """
        for (order1_id, order2_id), _ in self.sorted_distances:
            if (order2_id in [order.order_id for order in self.candidate_orders]) \
            and (order2_id not in [order_id for order_id in self.failed_attempts]):
                self.added_order_id = order2_id
                return order2_id

        return None

    def check_time_if_is_peek(self, current_time, rush_hour_start_1=6, rush_hour_end_1=9, rush_hour_start_2=16, rush_hour_end_2=19):
        hour = current_time.hour
        if rush_hour_start_1 <= hour < rush_hour_end_1 or rush_hour_start_2 <= hour < rush_hour_end_2:  # Traffic jam hour
            return True
        return False

    def calculate_distance(self, loc_a, loc_b, current_time=None):
        distance, _, _ = self.calculate_distance_and_duration(loc_a, loc_b, current_time)
        return distance
    
    def calculate_distance_using_i(self, a_idx, b_idx, current_time=None):
        if a_idx < 0 or b_idx < 0 or a_idx >= len(self.batch_locations) or b_idx >= len(self.batch_locations):
            return 0
        
        loc_a = self.batch_locations[a_idx]
        loc_b = self.batch_locations[b_idx]
        distance, _, _ = self.calculate_distance_and_duration(loc_a, loc_b, current_time)
        return distance
    
    def calculate_distance_and_duration(self, loc_1, loc_2, current_time):
        if current_time:
            is_peek_time = self.check_time_if_is_peek(current_time)
        else:
            is_peek_time = True
            
        if is_peek_time:
            duration_str = "duration_traffic"
        else:
            duration_str = "duration_general"

        matrix_id_1 = self.loc_to_matrix_id(loc_1)
        matrix_id_2 = self.loc_to_matrix_id(loc_2)
        
        if matrix_id_1 == matrix_id_2:
            distance, duration, duration_timedelta = 0, 0, timedelta(hours=0)
        else:
            try:
                distance = self.matrix[matrix_id_1][matrix_id_2]['distance']
                duration = self.matrix[matrix_id_1][matrix_id_2][duration_str]
                duration_timedelta = timedelta(hours=duration)
            except (KeyError, TypeError, IndexError):

                print(type(self.matrix))
                print(matrix_id_1, matrix_id_2)
                
                distance = self.matrix[matrix_id_1][matrix_id_2]['distance']
                duration = self.matrix[matrix_id_1][matrix_id_2][duration_str]
                duration_timedelta = timedelta(hours=duration)
    
        return distance, duration, duration_timedelta

    def convert_to_float(self, d):
        new_dict = {}
        for key, value in d.items():
            # 키를 float으로 변환
            try:
                new_key = float(key)
            except ValueError:
                new_key = key
            
            if isinstance(value, dict):
                # 값이 딕셔너리인 경우 재귀 호출
                new_value = self.convert_to_float(value)
            else:
                # 값이 딕셔너리가 아닌 경우 float으로 변환
                try:
                    new_value = float(value)
                except ValueError:
                    new_value = value
    
            # 새로운 딕셔너리에 추가
            new_dict[new_key] = new_value
    
        return new_dict

    def loc_to_matrix_id(self, loc):
        if self.sheet_name == "SameDay Orders(8 Hours SLA)":
            sheet_i = 1
        elif self.sheet_name == "Instant Orders(1-4 Hours SLA)":
            sheet_i = 2
        else:
            sheet_i = 3
        
        if loc.location_type == LocationType.PICKUP:
            location_i = 1
        else:
            location_i = 2

        if isinstance(loc.order_id, str):
            id_i = int(loc.order_id[4:]) + 100
        else:
            id_i = loc.order_id
            
        return 10000 * sheet_i + 1000 * location_i + id_i
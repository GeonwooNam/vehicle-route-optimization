from math import sqrt
from datetime import datetime, timedelta, time
from batch_classes import LocationType, Location, Dropoff, Pickup, Order, Batch
from typing import List
import random

import numpy as np
import pandas as pd

import folium
from folium.plugins import BeautifyIcon

from route_optimization_api import RouteOptimizer

class BatchManager(RouteOptimizer):
    # 하나의 Batch에 담을 수 있는 주문 고려, 생성 과정 총괄
    
    def __init__(
        self, sheet_name, all_orders, locations, matrix,
        max_capacity=None, max_distance=None, delay_minute_per_task=None, first_order_pickup_duration=None
    ):
        super().__init__(
            sheet_name,
            all_orders,
            locations,
            matrix,
            max_capacity=max_capacity, 
            max_distance=max_distance, 
            delay_minute_per_task=delay_minute_per_task,
            first_order_pickup_duration=first_order_pickup_duration,
        )
        self.num_orders = len(all_orders)
        self.location_pool = locations
        self.failed_pool = set()  # 선택 가능한 주문 집합
        self.batch_orders = set()  # 현재 배치에 포함된 주문 집합
        self.batch_list = []
        self._batch_id_counter = 1

        self.color_hex_list = [
            "#FF0000", "#FF1493", "#0047AB", "#008000", 
            "#FF00FF", "#00008B", "#8000FF",
            "#FF6347", "#87CEEB", "#FFFDD0", 
            "#FF7F50", "#800080", "#A52A2A", "#0000FF",
        ]
        self.new_order_id = 1
        self.info_of_new_order_and_batch = {}
    
    def sort_possible_distances(self):
        sorted_distances = []

        if not self.current_batch:  # current_batch가 공집합인 경우
            # 모든 주문 간 거리 값을 정렬
            for order1 in [orde for order in self.order_pool]:
                for order2 in [order for order in self.order_pool]:
                    if order1.order_id != order2.order_id:
                        distance = self.distance_between_orders(order1, order2)
                        sorted_distances.append(((order1, order2), distance))
                        
        else:  # current_batch가 비어있지 않은 경우
            # current_batch와 candidate_orders 사이의 거리만 정렬
            for order1 in self.current_batch:
                for order2 in self.candidate_orders:
                    distance = self.distance_between_orders(order1, order2)
                    sorted_distances.append(((order1, order2), distance))

        sorted_distances.sort(key=lambda x: x[1])

        return sorted_distances

    def find_most_urgent_location(self):
        return next(
            loc for loc in self.location_pool
        )
    
    def add_to_batch(self, order):
        """
        주문을 current_batch에 추가하고 candidate_orders에서 제거.
        """
        if order in self.candidate_orders:
            self.current_batch.add(order)
            self.candidate_orders.remove(order)
            return True
        return False

    def reset_failed_pool(self):
        self.failed_pool = set()

    def add_to_failed_pool(self, order):
        self.failed_pool.add(order)

    def add_order_to_batch(self, order):
        self.batch_orders.add(order)
        
    def delete_batch_from_pool(self):
        batch_order_ids = {order.order_id for order in self.batch_orders}  # 집합으로 변환
        orders_to_remove = {
            order for order in self.order_pool 
            if order.order_id in batch_order_ids
        }  # 제거할 주문 수집
    
        # 한 번에 제거
        self.order_pool -= orders_to_remove

        self.batch_orders = set()
        self.batch_locations = []

    def add_current_batch_to_list(self):
        # 중복 제거
        location_list = [[order.pickup, order.dropoff] for order in self.batch_orders]
        self.batch_locations = [location for sublist in location_list for location in sublist]
        
        self.enforce_pickup_dropoff()
        self.optimize_route()
        self.enforce_pickup_dropoff()
        self.update_arrival_time()
        
        distance = self.calculate_total_distance()
        duration = self.calculate_total_duration()
        start_time = self.batch_locations[0].time_window[0] + timedelta(minutes=self.FIRST_ORDER_PICKUP_DURATION)
        end_time = start_time + duration

        accumulated_distances, accumulated_weights = self.calculate_batch_metrics(self.batch_locations)
        
        self.batch_list.append(
            Batch(
                batch_id=self._batch_id_counter,
                route_locations=self.batch_locations,
                route_orders=self.batch_orders,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                distance=distance,
                accumulated_distances=accumulated_distances,
                accumulated_weights=accumulated_weights,
            )
        )
        self._batch_id_counter += 1

    def calculate_batch_metrics(self, batch_route: List[Location]):
        """
        배치 리스트에서 각 배치의 누적 거리와 용량을 계산하여 반환하는 함수.
        
        Returns:
            accumulated_distances (list): 각 배치의 누적 거리 리스트.
            accumulated_weights (list): 각 배치의 누적 용량 리스트.
        """
        accumulated_distances = [0,]
        accumulated_weights = []
    
        total_distance = 0
        current_capacity = 0.00000001
          
        for i in range(len(batch_route)):
            location = batch_route[i]

            # 누적 거리
            if i > 0:
                loc1 = batch_route[i-1]
                loc2 = batch_route[i]
                
                total_distance += self.calculate_distance(loc1, loc2)
                accumulated_distances.append(total_distance)

            # 누적 용량
            order = self.order_dict[location.order_id]
            order_capacity = order.weight

            if location.location_type == LocationType.PICKUP:
                current_capacity += order_capacity
            elif location.location_type == LocationType.DROPOFF:
                current_capacity -= order_capacity
            accumulated_weights.append(current_capacity)
    
        return accumulated_distances, accumulated_weights

    def gather_locations_from_order(self, order_to_add):
        order_list = list(self.batch_orders)
        order_list.append(order_to_add)
        location_list = [[order.pickup, order.dropoff] for order in order_list]
        flattened_location_list = [location for sublist in location_list for location in sublist]
        return flattened_location_list

    def distance_between_orders(self, order_a, order_b):
        pickup_a = self.loc_to_matrix_id(order_a.pickup)
        dropoff_a = self.loc_to_matrix_id(order_a.dropoff)
        pickup_b = self.loc_to_matrix_id(order_b.pickup)
        dropoff_b = self.loc_to_matrix_id(order_b.dropoff)

        distance_list = []
        for a_id in [pickup_a, dropoff_a]:
            for b_id in [pickup_b, dropoff_b]:
                distance = self.matrix[a_id][b_id]["distance"]
                distance_list.append(distance)
                distance = self.matrix[b_id][a_id]["distance"]
                distance_list.append(distance)

        if len(distance_list) > 0:
            return sum(distance_list) / len(distance_list)
        else:
            return 0
    
    def find_close_order(self):
        """
        현재 배치에 포함된 주문으로부터 평균 거리가 가장 가까운 주문을 반환.
        """
        closest_order = None
        min_distance = float('inf')  # 초기 최소 거리 값을 무한대로 설정
    
        # 현재 배치에 포함된 모든 주문과 후보 주문 간 거리 계산
        for candidate_order in self.order_pool - self.batch_orders - self.failed_pool:
            distance_list = []
            for batch_order in self.batch_orders:
                # 두 주문 간의 평균 거리 가져오기
                distance = self.distance_between_orders(candidate_order, batch_order)
                distance_list.append(distance)

            avg_distance = sum(distance_list) / len(distance_list)
            # 최소 거리 업데이트
            if avg_distance < min_distance:
                min_distance = avg_distance
                closest_order = candidate_order
    
        return closest_order

    def calculate_avg_distance_with_batch_orders(self, order):
        distance_list = []
        for batch_order in self.batch_orders:
            # 두 주문 간의 평균 거리 가져오기
            distance = self.distance_between_orders(order, batch_order)
            distance_list.append(distance)

        avg_distance = sum(distance_list) / len(distance_list)
        return avg_distance

    def delete_location_from_order(self, target_order):
        self.batch_locations = [
            loc for loc in self.batch_locations if loc.order_id != target_order.order_id
        ]

    def make_map(self, batch_id=1):
        return folium.Map(location=self.batch_list[batch_id-1].route[0].location, zoom_start=14)
    
    def add_single_route_to_map(self, batch_id=1, make_map=True, existing_map=None, number_version=True):
        if make_map:
            map_ = self.make_map(batch_id=batch_id)
        else:
            map_ = existing_map
        
        batch_id -= 1
        batch = self.batch_list[batch_id]

        # 2. 경로를 선으로 연결
        coords = [location.location for location in batch.route]
    
        # 3. 선 그리기
        folium.PolyLine(locations=coords, color=self.color_hex_list[batch_id], weight=2.5, opacity=0.8, tooltip=f"Batch ID: {batch_id+1}").add_to(map_)

        accumulated_distances = batch.accumulated_distances
        accumulated_weights = batch.accumulated_weights

        if len(accumulated_distances) != len(batch.route):
            accumulated_distances, accumulated_weights = self.calculate_batch_metrics(batch.route)

        new_order_colors = {}
        available_colors = set(self.color_hex_list) - {self.color_hex_list[batch_id]}

        for order_idx, location in enumerate(batch.route):
            order = self.order_dict[location.order_id]
            start_window = order.time_window[0]
            end_window = order.time_window[1]

            if isinstance(location.order_id, str) and location.order_id.startswith("new_"):
                if location.order_id not in new_order_colors:
                    new_order_colors[location.order_id] = random.choice(list(available_colors))
                    available_colors -= {new_order_colors[location.order_id]}
    
                color = new_order_colors[location.order_id]
            else:
                color = self.color_hex_list[batch_id]

            tooltip_info = (
                f"Batch ID: {batch_id+1}<br><br>"
                f"Sequence number: {order_idx+1}<br>"
                f"Order ID: {location.order_id}<br>"
                f"Location type: {'PICKUP' if location.location_type == LocationType.PICKUP else 'DROPOFF'}<br>"
            )
            
            # location_type이 dropoff일 때만 무게를 표시
            if location.location_type == LocationType.PICKUP:
                tooltip_info += f"Weight: {location.weight} kg<br>"
            
            tooltip_info += (
                f"<br><span style='display:inline-block; width:95px;'>Pickup time:</span>{start_window.strftime('%H:%M:%S')}<br>"
                f"<span style='display:inline-block; width:95px;'>Arrival time:</span>{location.arrival_time.strftime('%H:%M:%S')}<br>"
                f"<span style='display:inline-block; width:95px;'>Delivery deadline:</span>{end_window.strftime('%H:%M:%S')}<br>"
                f"<br>Accumulated Distance: {accumulated_distances[order_idx]:.2f} km<br>"
                f"Accumulated Weight: {accumulated_weights[order_idx]:.2f} kg<br>"
            )

            if number_version:
                folium.Marker(
                    location=location.location,
                    icon=BeautifyIcon(
                        icon='arrow-down',
                        icon_shape='circle',
                        border_width=2,
                        border_color=color,  # 테두리 색상을 color로 설정
                        number=location.order_id,
                        background_color=color if location.location_type == LocationType.PICKUP else 'white',
                        text_color='white' if location.location_type == LocationType.PICKUP else color,  # 숫자 색상
                    ),
                    tooltip=tooltip_info
                ).add_to(map_)

            else:
                folium.CircleMarker(
                    location=location.location,
                    radius=7,  # 점의 크기
                    color=color,  # 점의 테두리 색상
                    fill=True,  # PICKUP일 때만 내부 색상 채우기
                    fill_color=color,  # 내부 색상
                    fill_opacity=0.8 if location.location_type == LocationType.PICKUP else 0,  # 내부 투명도
                    tooltip=tooltip_info  # 마우스 오버 시 표시될 정보
                ).add_to(map_)


        return map_
        
    
    def add_all_route_to_map(self, number_version=False):
        map_ = self.make_map(batch_id=1)
        
        for idx, batch in enumerate(self.batch_list):
            map_ = self.add_single_route_to_map(batch_id=idx+1, make_map=False, existing_map=map_, number_version=number_version)
            
        return map_

    def get_batch_metrics_table(self, batch_no=1):
        batch = self.batch_list[batch_no-1]
        data = []

        batch.accumulated_distances, batch.accumulated_weights = self.calculate_batch_metrics(batch.route)
    
        for order_idx, location in enumerate(batch.route):
            order = self.order_dict[location.order_id]
            start_window = order.time_window[0]
            end_window = order.time_window[1]
            
            row = {
                "ID": location.order_id,
                "Type": "PICKUP" if location.location_type == LocationType.PICKUP else "DROPOFF",
                "Weight": f"{location.weight} kg" if location.location_type == LocationType.PICKUP else "",
                "Arrival": location.arrival_time.strftime('%H:%M:%S'),
                "Deadline": end_window.strftime('%H:%M:%S'),
                "(Created)": start_window.strftime('%H:%M:%S'),
                "Acc Dist": f"{batch.accumulated_distances[order_idx]:.2f} km",
                "Acc Weight": f"{batch.accumulated_weights[order_idx]:.2f} kg",
            }
            data.append(row)
        
        return pd.DataFrame(data)
        
    def store_info_of_new_order_and_batch(self, added_batch, new_order):
        pickup_index = next(
            (i for i, loc in enumerate(added_batch.route) 
             if loc.order_id == new_order.order_id and loc.location_type == LocationType.PICKUP),
            None
        )
        dropoff_index = next(
            (i for i, loc in enumerate(added_batch.route) 
             if loc.order_id == new_order.order_id and loc.location_type == LocationType.DROPOFF),
            None
        )
        
        self.info_of_new_order_and_batch = {
            "batch_id": added_batch.batch_id,
            "new_order": new_order,
            "pickup_index": pickup_index,
            "dropoff_index": dropoff_index,
        }

    def timedelta_to_time_string(self, time_delta):
        total_seconds = int(time_delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        # 결과 출력
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    
    def optimize_stats(self):
        distance_list = [self.calculate_batch_total_distance(batch) for batch in self.batch_list]
        duration_list = [batch.duration for batch in self.batch_list]

        total_duration = sum(duration_list, timedelta())
        average_duration = total_duration / len(duration_list)

        print(f"Number of batch : {len(self.batch_list)}\n")
        
        print(f"Total distance   : {sum(distance_list):.2f} km")
        print(f"Average distance : {sum(distance_list)/len(distance_list):.2f} km")
        print(f"Maximum distance : {max(distance_list):.2f} km")
        print(f"Minimum distance : {min(distance_list):.2f} km")
        print('\n')
        
        print(f"Total duration   : {self.timedelta_to_time_string(total_duration)}")
        print(f"Average duration : {self.timedelta_to_time_string(average_duration)}")
        print(f"Maximum duration : {self.timedelta_to_time_string(max(duration_list))}")
        print(f"Minimum duration : {self.timedelta_to_time_string(min(duration_list))}")
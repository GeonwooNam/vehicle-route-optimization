from typing import List
from enum import Enum
from datetime import datetime, timedelta, time
from functools import total_ordering

class LocationType(Enum):
    PICKUP = 1
    DROPOFF = 2

@total_ordering
class Location:
    """기본 목적지 데이터를 관리하는 클래스"""
    def __init__(self, order_id, location_type, location_xy, time_window):

        # 기본 정보
        self.order_id = order_id  # 주문 ID
        self.location_type = location_type  # 위치 타입: pickup 또는 dropoff
        self.weight = 0  # 주문 무게

        # 시간 정보
        self.arrival_time = time_window[1]  # 현재 가능한 도착 시간
        self.time_window = time_window  # 가능 시간 범위

        # 위치 정보
        self.location = location_xy  # 튜플 형태 (x, y)

    def __eq__(self, other):
        if not isinstance(other, Location):
            return NotImplemented
        return self.arrival_time == other.arrival_time

    def __lt__(self, other):
        if not isinstance(other, Location):
            return NotImplemented
        return self.arrival_time < other.arrival_time

class Dropoff(Location):
    """주문 데이터(Dropoff)를 관리하는 클래스"""
    def __init__(self, order_id, location_xy, time_window):
        super().__init__(order_id, LocationType.DROPOFF, location_xy, time_window)

class Pickup(Location):
    """허브 데이터(Pickup)를 관리하는 클래스"""
    def __init__(self, order_id, location_xy, time_window, weight):
        super().__init__(order_id, LocationType.PICKUP, location_xy, time_window)
        self.weight = weight

class Order:
    """Pickup과 Dropoff 쌍을 관리하는 클래스"""
    def __init__(self, pickup: Pickup, dropoff: Dropoff, time_window: tuple, order_id=None):
        if order_id:
            self.order_id = order_id
        else: 
            self.order_id = pickup.order_id
            
        self.pickup = pickup
        self.dropoff = dropoff
        self.time_window = time_window
        self.weight = pickup.weight

class Batch:
    """하나의 배치에 대한 정보를 관리하는 클래스"""
    
    def __init__(
        self,
        batch_id: int,
        route_locations: List[Location],
        route_orders: List[Order], 
        start_time: datetime,
        end_time: datetime, 
        duration: timedelta,
        distance: float,
        accumulated_distances: List[float],
        accumulated_weights: List[float],
    ):
        self.batch_id = batch_id        
        self.route = route_locations
        self.orders = route_orders
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.distance = distance
        self.accumulated_distances = accumulated_distances
        self.accumulated_weights = accumulated_weights
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Box:
    bl: Point
    tr: Point

@dataclass
class Node:
    level: int
    mbr: Box
    value: Box
    space: int
    childern: list

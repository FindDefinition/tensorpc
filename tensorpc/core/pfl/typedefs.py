import enum 

class BoolOpType(enum.IntEnum):
    AND = 0
    OR = 1


class BinOpType(enum.IntEnum):
    ADD = 0
    SUB = 1
    MULT = 2
    DIV = 3
    MOD = 4
    POW = 5
    LSHIFT = 6
    RSHIFT = 7
    BIT_OR = 8
    BIT_XOR = 9
    BIT_AND = 10
    FLOOR_DIV = 11
    MATMUL = 12


class UnaryOpType(enum.IntEnum):
    INVERT = 0
    NOT = 1
    UADD = 2
    USUB = 3


class CompareType(enum.IntEnum):
    EQUAL = 0
    NOT_EQUAL = 1
    LESS = 2
    LESS_EQUAL = 3
    GREATER = 4
    GREATER_EQUAL = 5
    IS = 6
    IS_NOT = 7
    IN = 8
    NOT_IN = 9

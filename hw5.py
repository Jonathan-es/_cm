import random
from group import Group

# --- Finite field additive group (mod p) ---
class FiniteFieldAddGroup(Group):
    def __init__(self, p):
        self.p = p
        self._identity = 0

    @property
    def identity(self):
        return self._identity

    def operation(self, a, b):
        return (a + b) % self.p

    def inverse(self, val):
        return (-val) % self.p

    def include(self, element):
        return isinstance(element, int) and 0 <= element < self.p

    def random_generate(self):
        return random.randint(0, self.p - 1)


# --- Finite field multiplicative group (mod p, exclude 0) ---
class FiniteFieldMulGroup(Group):
    def __init__(self, p):
        self.p = p
        self._identity = 1

    @property
    def identity(self):
        return self._identity

    def operation(self, a, b):
        return (a * b) % self.p

    def inverse(self, val):
        if val == 0:
            raise ValueError("Zero has no multiplicative inverse")
        # Extended Euclidean Algorithm to find modular inverse
        t, new_t = 0, 1
        r, new_r = self.p, val
        while new_r != 0:
            quotient = r // new_r
            t, new_t = new_t, t - quotient * new_t
            r, new_r = new_r, r - quotient * new_r
        if r > 1:
            raise ValueError(f"{val} has no inverse modulo {self.p}")
        return t % self.p

    def include(self, element):
        return isinstance(element, int) and 1 <= element < self.p

    def random_generate(self):
        return random.randint(1, self.p - 1)


# --- Finite field object ---
class FiniteField:
    def __init__(self, p):
        if p < 2:
            raise ValueError("p must be a prime number >= 2")
        self.p = p
        self.add_group = FiniteFieldAddGroup(p)
        self.mul_group = FiniteFieldMulGroup(p)

    def add(self, a, b):
        return self.add_group.operation(a, b)

    def subtract(self, a, b):
        return self.add_group.operation(a, self.add_group.inverse(b))

    def multiply(self, a, b):
        return self.mul_group.operation(a, b)

    def divide(self, a, b):
        return self.mul_group.operation(a, self.mul_group.inverse(b))

    def additive_inverse(self, val):
        return self.add_group.inverse(val)

    def multiplicative_inverse(self, val):
        return self.mul_group.inverse(val)

    def element(self, val):
        """Wrap an int as a finite field element"""
        return FiniteFieldElement(self, val)


# --- Finite field element with operator overloading ---
class FiniteFieldElement:
    def __init__(self, field, value):
        self.field = field
        self.value = value % self.field.p

    def __repr__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, FiniteFieldElement):
            return self.value == other.value and self.field == other.field
        return False

    def __add__(self, other):
        if isinstance(other, FiniteFieldElement):
            if self.field != other.field:
                raise ValueError("Cannot operate across fields")
            return self.field.element(self.field.add(self.value, other.value))
        if isinstance(other, int):
            return self.field.element(self.field.add(self.value, other))
        raise TypeError("Unsupported operand type for +")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, FiniteFieldElement):
            if self.field != other.field:
                raise ValueError("Cannot operate across fields")
            return self.field.element(self.field.subtract(self.value, other.value))
        if isinstance(other, int):
            return self.field.element(self.field.subtract(self.value, other))
        raise TypeError("Unsupported operand type for -")

    def __rsub__(self, other):
        if isinstance(other, int):
            return self.field.element(self.field.subtract(other, self.value))
        raise TypeError("Unsupported operand type for -")

    def __mul__(self, other):
        if isinstance(other, FiniteFieldElement):
            if self.field != other.field:
                raise ValueError("Cannot operate across fields")
            return self.field.element(self.field.multiply(self.value, other.value))
        if isinstance(other, int):
            return self.field.element(self.field.multiply(self.value, other))
        raise TypeError("Unsupported operand type for *")

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, FiniteFieldElement):
            if self.field != other.field:
                raise ValueError("Cannot operate across fields")
            return self.field.element(self.field.divide(self.value, other.value))
        if isinstance(other, int):
            return self.field.element(self.field.divide(self.value, other))
        raise TypeError("Unsupported operand type for /")

    def __rtruediv__(self, other):
        if isinstance(other, int):
            return self.field.element(self.field.divide(other, self.value))
        raise TypeError("Unsupported operand type for /")
